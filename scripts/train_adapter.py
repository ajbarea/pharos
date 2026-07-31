#!/usr/bin/env python3
"""Is the analyst's decision rule learnable by gradient, where examples failed?

Finding 5 measured the cheap proxy for the design's central premise and it came back
negative: withholding the rule and supplying up to eight labelled examples closed
none of the gap to the rule-given ceiling of F1 1.000. That result explicitly did not
settle the question, because in-context learning and gradient learning are different
mechanisms and the first failing does not establish that the second will. This script
runs the second.

**What it answers.** Whether a LoRA trained on labelled verdicts recovers a rule that
was never stated to it. The rule is a fixed conjunction of three facts out of fifteen,
so it is learnable in principle; whether it is learnable from this signal, at this
scale, is the measurement.

**What it does not answer.** The design learns from an analyst's accept, revise, and
reject decisions. Those are Pharos build-order step 3 and do not exist yet. This
trains on clean definitional labels instead, which is a strictly easier problem. A
negative result here would therefore be strong evidence against the premise; a
positive result is necessary but not sufficient for it.

Three things keep the comparison honest.

**The prompt is byte-identical to the one finding 5 used.** `INSTRUCTION_NO_RULE`
and the report formatting are imported from `measure_rule_learnability`, not
reimplemented, so a difference in score cannot be a difference in prompt.

**The split is by event, and events are disjoint.** One task per event, so an index
split is an event split. Nothing in evaluation was trained on.

**The baseline is the same checkpoint, unmodified.** The published sweep numbers came
from Q4-quantized weights served by Ollama; this trains bf16 Hugging Face weights.
Comparing across that difference would confound quantization with training, so the
script evaluates the base model itself under the same prompt and reports both.

    uv sync --extra train
    uv run python scripts/train_adapter.py --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_rule_learnability import (
    INSTRUCTION_NO_RULE,
    _reports_block,
    parse_verdict,
)

from pharos.generate import GeneratorConfig, generate
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import get_logger, log_execution_context, progress, record

LOG = get_logger()

#: Hyperparameters from current practice: rank 16 with alpha 32 keeps alpha/r at 2,
#: which is the ratio the 2026 guidance converges on, and 1e-4 with cosine decay over
#: a few epochs is the standard range for adapter SFT on a small model.
#: research(2026-08): r=16/alpha=32, lr 1e-4 cosine, 3 epochs, target attention+FFN,
#: never embeddings or layer norms.
#: F1 reachable when the rule IS stated and the prompt structures the check
#: (finding 3). The anchor every number here is measured against.
CEILING_F1 = 1.000

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def verdict_text(significant: bool) -> str:
    """The completion the model must produce. Deliberately the shortest thing that
    `parse_verdict` accepts, so the measurement is of the decision and not of prose."""
    return f"VERDICT: {'SIGNIFICANT' if significant else 'ROUTINE'}"


def prompt_for(task: TriageTask) -> str:
    """Byte-identical to the prompt finding 5 scored, so the two are comparable."""
    return f"{_reports_block(task)}\n\n{INSTRUCTION_NO_RULE}"


def build_examples(tasks: list[TriageTask]) -> list[dict[str, str]]:
    return [{"prompt": prompt_for(t), "completion": verdict_text(t.significant)} for t in tasks]


def _class_balance(tasks: list[TriageTask]) -> dict[str, int]:
    positive = sum(1 for t in tasks if t.significant)
    return {"significant": positive, "routine": len(tasks) - positive}


def tokenize_masked(example: dict[str, str], tokenizer, max_len: int):
    """Tokenize, masking the prompt out of the loss.

    Only the completion is supervised. Training on the prompt would teach the model to
    generate maritime reports, which is not the task and would swamp the few tokens
    that carry the decision.
    """
    prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(
        example["completion"] + tokenizer.eos_token, add_special_tokens=False
    )["input_ids"]

    input_ids = (prompt_ids + completion_ids)[-max_len:]
    # Recompute how much of the (possibly truncated) sequence is prompt.
    n_completion = min(len(completion_ids), len(input_ids))
    labels = [-100] * (len(input_ids) - n_completion) + input_ids[-n_completion:]
    return {"input_ids": input_ids, "labels": labels}


def evaluate(model, tokenizer, tasks: list[TriageTask], *, label: str) -> dict[str, object]:
    """Greedy-decode a verdict per task and score it. No sampling, so this is stable."""
    import torch

    model.eval()
    tp = fp = tn = fn = unparsed = 0
    for index, task in enumerate(tasks):
        text = prompt_for(task)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3072).to(
            model.device
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        verdict = parse_verdict(answer)

        if verdict is None:
            unparsed += 1
        elif verdict and task.significant:
            tp += 1
        elif verdict and not task.significant:
            fp += 1
        elif not verdict and task.significant:
            fn += 1
        else:
            tn += 1

        if (index + 1) % 10 == 0:
            progress("adapter.eval_progress", label=label, done=index + 1, total=len(tasks))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + tn + fn + unparsed
    prevalence = (tp + fn) / total if total else 0.0
    return {
        "label": label,
        "n": total,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "unparsed": unparsed,
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "majority_accuracy": round(max(prevalence, 1 - prevalence), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--events", type=int, default=1200, help="corpus size to draw tasks from")
    parser.add_argument("--eval-tasks", type=int, default=60)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    log_execution_context()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    if not torch.cuda.is_available():
        print("no CUDA device; this experiment needs a GPU node", file=sys.stderr)
        return 1

    # ---- data ---------------------------------------------------------------
    reports = generate(GeneratorConfig(seed=args.seed, n_events=args.events))
    all_tasks = build_triage_tasks(reports)
    evaluation = all_tasks[: args.eval_tasks]
    training = all_tasks[args.eval_tasks :]
    leaked = {t.event_id for t in evaluation} & {t.event_id for t in training}
    if leaked:
        raise SystemExit(
            f"{len(leaked)} events leaked between train and eval: {sorted(leaked)[:5]}"
        )

    print(f"train {len(training)} tasks, eval {len(evaluation)} tasks, events disjoint")
    print(f"train balance {_class_balance(training)}")
    print(f"eval  balance {_class_balance(evaluation)}")
    progress(
        "adapter.data",
        n_train=len(training),
        n_eval=len(evaluation),
        train_balance=_class_balance(training),
        eval_balance=_class_balance(evaluation),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )

    # ---- baseline: the same checkpoint, untrained ---------------------------
    base_result = None
    if not args.skip_base_eval:
        print("\n>>> baseline: this checkpoint, rule withheld, no training")
        base_result = evaluate(model, tokenizer, evaluation, label="base")
        print(json.dumps(base_result, indent=2))
        record("adapter.base_f1", base_result["f1"], model=args.model)

    # ---- train --------------------------------------------------------------
    lora = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(TARGET_MODULES),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nlora: {trainable} trainable of {total} ({100 * trainable / total:.3f}%)")
    if not 0 < trainable < total:
        raise SystemExit(
            f"peft did not freeze the base: {trainable} trainable of {total}. "
            "Training every parameter is not the experiment."
        )

    rows = [tokenize_masked(e, tokenizer, args.max_len) for e in build_examples(training)]

    def collate(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        pad = tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor(
                [b["input_ids"] + [pad] * (longest - len(b["input_ids"])) for b in batch]
            ),
            "labels": torch.tensor(
                [b["labels"] + [-100] * (longest - len(b["labels"])) for b in batch]
            ),
            "attention_mask": torch.tensor(
                [[1] * len(b["input_ids"]) + [0] * (longest - len(b["input_ids"])) for b in batch]
            ),
        }

    out_dir = Path("adapter-out")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=5,
            save_strategy="no",
            bf16=True,
            report_to=[],
            seed=args.seed,
        ),
        train_dataset=rows,
        data_collator=collate,
    )
    train_out = trainer.train()
    final_loss = train_out.training_loss
    print(f"\ntraining loss {final_loss:.4f}")
    record("adapter.train_loss", final_loss, model=args.model)

    # ---- evaluate the adapter ----------------------------------------------
    print("\n>>> adapter: same prompt, same held-out events")
    tuned_result = evaluate(model, tokenizer, evaluation, label="adapter")
    print(json.dumps(tuned_result, indent=2))
    record("adapter.f1", tuned_result["f1"], model=args.model)

    # ---- verdict against the two anchors -----------------------------------
    print("\n" + "=" * 72)
    if base_result:
        print(f"base     F1 {base_result['f1']:.3f}  acc {base_result['accuracy']:.3f}")
    print(f"adapter  F1 {tuned_result['f1']:.3f}  acc {tuned_result['accuracy']:.3f}")
    print(f"ceiling  F1 {CEILING_F1:.3f}  (rule stated, checklist prompt)")
    if base_result:
        gap = CEILING_F1 - base_result["f1"]
        closed = (tuned_result["f1"] - base_result["f1"]) / gap if gap > 0 else 0.0
        print(f"\ngap closed: {closed:+.1%} of the distance from base to ceiling")
        print(
            "Interpret against finding 5, where in-context examples closed none of it.\n"
            "A gain here means the rule is gradient-learnable from clean labels, which\n"
            "is necessary but NOT sufficient for learning it from analyst decisions."
        )
    print("=" * 72)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "provenance": run_provenance(model=args.model, seed=args.seed),
                    "hyperparameters": {
                        "lora_rank": LORA_RANK,
                        "lora_alpha": LORA_ALPHA,
                        "lora_dropout": LORA_DROPOUT,
                        "target_modules": list(TARGET_MODULES),
                        "epochs": args.epochs,
                        "learning_rate": args.lr,
                        "effective_batch": args.batch_size * args.grad_accum,
                    },
                    "data": {
                        "n_train": len(training),
                        "n_eval": len(evaluation),
                        "train_balance": _class_balance(training),
                        "eval_balance": _class_balance(evaluation),
                        "events_disjoint": True,
                    },
                    "training_loss": final_loss,
                    "base": base_result,
                    "adapter": tuned_result,
                    "ceiling_f1": CEILING_F1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
