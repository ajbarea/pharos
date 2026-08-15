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
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


from pharos.analyst import DEFAULT_ENSEMBLE, AnalystPolicy
from pharos.generate import GeneratorConfig, generate
from pharos.prompting import INSTRUCTION_NO_RULE, parse_verdict, reports_block
from pharos.provenance import run_provenance
from pharos.tasks import TriageTask, build_triage_tasks
from pharos.telemetry import get_logger, log_execution_context, progress, record
from pharos.validity import check_classification

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


@dataclass(frozen=True, slots=True)
class EvalResult:
    """One evaluation pass, typed.

    This was a plain dict, which meant every downstream arithmetic on it was
    `float - object` and passed only because the values happened to be floats. The
    rest of the codebase reports metrics as frozen dataclasses for exactly this
    reason; the scripts had not followed suit.
    """

    label: str
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    unparsed: int
    accuracy: float
    majority_accuracy: float
    precision: float
    recall: float
    f1: float
    #: The validity assessment, carried into the artifact rather than only printed.
    #: It was computed here and discarded, so a published adapter number could not be
    #: checked against the conditions that make a score misleading without rerunning a
    #: GPU job. `pharos.validity` exists to make that checkable; dropping its output
    #: made it a console warning instead.
    validity: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "n": self.n,
            "confusion": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn},
            "unparsed": self.unparsed,
            "accuracy": self.accuracy,
            "majority_accuracy": self.majority_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "validity": self.validity,
        }


def _scratch_dir() -> Path:
    """Where the trainer writes checkpoints, scoped so two jobs cannot collide.

    This was a bare `adapter-out/`, which is safe for exactly one run at a time. It is
    not safe for the way the cluster is actually used: `review-adapter.sbatch` trains
    four adapters in a loop and `adapter.sbatch` trains a fifth, and submitting the
    second while the first is running would have had both writing checkpoints and
    optimizer state through the same path. Nothing would have failed -- one job would
    simply have finished with the other's weights, and the artifact would have carried
    a provenance stamp saying otherwise.

    Slurm hands out a unique job id, so use it where there is one and fall back to the
    process id locally, where concurrent training is unlikely but not prevented.
    """
    job = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    return Path("adapter-out") / job


def verdict_text(significant: bool) -> str:
    """The completion the model must produce. Deliberately the shortest thing that
    `parse_verdict` accepts, so the measurement is of the decision and not of prose."""
    return f"VERDICT: {'SIGNIFICANT' if significant else 'ROUTINE'}"


def prompt_for(task: TriageTask) -> str:
    """Byte-identical to the prompt finding 5 scored, so the two are comparable."""
    return f"{reports_block(task)}\n\n{INSTRUCTION_NO_RULE}"


def world_targets(tasks: list[TriageTask]) -> dict[str, bool]:
    """The generator's own ground truth. What finding 6 trained on."""
    return {t.task_id: t.significant for t in tasks}


def review_targets(tasks: list[TriageTask], policy: AnalystPolicy, *, seed: int) -> dict[str, bool]:
    """The targets a reviewer's decisions would supply.

    Taken from the reviewer's own verdict rather than routed through a proposal.
    Finding 7 established that the target stream is proposal-independent -- an
    accepted verdict is one the reviewer agreed with and a corrected one is their own
    call, and target accuracy came out identical across all six models because of it.
    Constructing proposals here would add a moving part that provably changes nothing.

    The rng is keyed exactly as `AnalystPolicy.review` keys it, so a reviewer slips on
    the same tasks whether they are reviewing or teaching.
    """
    return {
        t.task_id: policy.verdict_for(t, random.Random(f"{seed}:{policy.name}:{t.task_id}"))
        for t in tasks
    }


#: A reviewer specified as a point on the standard-by-carefulness grid rather than by
#: name: `t2s0.15` is a two-of-three reviewer slipping 15% of the time. The named
#: ensemble has eight members and only four of them move target accuracy at all -- the
#: rest vary disclosure or grounds -- so a teacher fleet larger than four cannot be
#: expressed by name. This is the same grid `measure_review_sweep.py` sweeps, so a
#: teacher's target accuracy is already known before an adapter is trained on it.
_GRID_SPEC = re.compile(r"^t(?P<threshold>[123])s(?P<slip>0(?:\.\d+)?)$")


def resolve_reviewer(name: str) -> AnalystPolicy:
    """A reviewer from the default grid by name, or a grid point like `t2s0.15`.

    Named lookup is tried first and is unchanged, so every committed artifact keeps
    resolving to exactly the policy that produced it.
    """
    by_name = {p.name: p for p in DEFAULT_ENSEMBLE}
    if name in by_name:
        return by_name[name]
    spec = _GRID_SPEC.match(name)
    if spec is None:
        raise SystemExit(
            f"unknown reviewer {name!r}; have: {', '.join(sorted(by_name))}, "
            f"or a grid point like t2s0.15 (threshold 1-3, slip rate)"
        )
    return AnalystPolicy(
        name,
        escalation_threshold=int(spec["threshold"]),
        slip_rate=float(spec["slip"]),
    )


def build_examples(
    tasks: list[TriageTask], targets: dict[str, bool] | None = None
) -> list[dict[str, str]]:
    """Prompt/completion pairs, labelled by `targets` or by the world when omitted."""
    return [
        {
            "prompt": prompt_for(t),
            "completion": verdict_text(t.significant if targets is None else targets[t.task_id]),
        }
        for t in tasks
    ]


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


def evaluate(
    model,
    tokenizer,
    tasks: list[TriageTask],
    *,
    label: str,
    max_new_tokens: int = 320,
    truth: dict[str, bool] | None = None,
) -> EvalResult:
    """Greedy-decode a verdict per task and score it. No sampling, so this is stable.

    `truth` overrides what counts as correct. Defaults to the world's own answer;
    pass a teacher's targets to ask whether the model reproduced its teacher rather
    than the rule. Same decode either way, so the two scorings differ only in the
    answer key and are directly comparable.

    `max_new_tokens` defaults high, and that is a correctness requirement rather than
    a tuning choice. The prompt asks the model to reason briefly and *then* emit a
    verdict line. A trained adapter learns to emit the verdict alone and fits in a
    dozen tokens; an untrained checkpoint starts reasoning and gets truncated before
    it ever reaches the verdict.

    The first run of this experiment used 12, and the baseline came back with 57 of
    60 answers unparsable and an F1 of 1.000 computed on the three that survived.
    That is not a baseline, it is a token limit, and comparing a fine-tuned model
    against it would have manufactured the entire result. 320 matches what
    measure_rule_learnability gives its own baseline, so the two are comparable.
    """
    import torch

    model.eval()
    answers = {t.task_id: t.significant for t in tasks} if truth is None else truth
    tp = fp = tn = fn = unparsed = 0
    for index, task in enumerate(tasks):
        expected = answers[task.task_id]
        text = prompt_for(task)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3072).to(
            model.device
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        verdict = parse_verdict(answer)

        if verdict is None:
            unparsed += 1
        elif verdict and expected:
            tp += 1
        elif verdict and not expected:
            fp += 1
        elif not verdict and expected:
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
    validity = check_classification(
        tp=tp, fp=fp, tn=tn, fn=fn, unparsed=unparsed, label=f"adapter:{label}"
    )
    if not validity.quotable:
        print(f"\n!! {label} measurement has validity concerns:")
        for concern in validity.concerns:
            print(f"   - {concern}")

    return EvalResult(
        label=label,
        n=total,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        unparsed=unparsed,
        accuracy=round((tp + tn) / total, 4) if total else 0.0,
        majority_accuracy=round(max(prevalence, 1 - prevalence), 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        validity=validity.as_dict(),
    )


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
    parser.add_argument(
        "--attn",
        default="sdpa",
        choices=("sdpa", "eager"),
        help="attention path; never the library default, which pulls in Triton",
    )
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "draw the evaluation set from a DIFFERENT corpus instantiation. Omitted, "
            "training and evaluation split one corpus by event, which shows the rule "
            "generalises across events. Supplied, they share no corpus at all, which "
            "is the stronger claim and the one contamination-resistance guidance asks "
            "for: evaluate on a seed with no instance overlap."
        ),
    )
    parser.add_argument(
        "--reviewer",
        default=None,
        help=(
            "train on this reviewer's targets instead of the generator's ground truth. "
            "Names come from pharos.analyst.DEFAULT_ENSEMBLE. Omit for finding 6's "
            "clean-label run, which is what the committed artifact reproduces."
        ),
    )
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

    if args.eval_seed is None:
        evaluation = all_tasks[: args.eval_tasks]
        training = all_tasks[args.eval_tasks :]
        split = f"one corpus at seed {args.seed}, split by event"
    else:
        # A whole separate instantiation. Same generator, same config, different
        # seed: different vessels, different events, different renderings of the
        # same fact vocabulary. Nothing is shared but the rule, which is the point.
        training = all_tasks
        evaluation = build_triage_tasks(
            generate(GeneratorConfig(seed=args.eval_seed, n_events=args.events))
        )[: args.eval_tasks]
        split = f"train seed {args.seed}, eval seed {args.eval_seed}, no shared corpus"

    leaked = {t.event_id for t in evaluation} & {t.event_id for t in training}
    if leaked and args.eval_seed is None:
        raise SystemExit(
            f"{len(leaked)} events leaked between train and eval: {sorted(leaked)[:5]}"
        )
    # Across seeds the event ids are drawn from the same namespace and collide by
    # construction, so identity is checked on the text instead. Two corpora sharing
    # a rendered report would mean the seeds are not independent.
    if args.eval_seed is not None:
        shared_text = {t.prompt for t in evaluation} & {t.prompt for t in training}
        if shared_text:
            raise SystemExit(
                f"{len(shared_text)} prompts appear in both corpora; seeds "
                f"{args.seed} and {args.eval_seed} are not independent"
            )

    print(f"train {len(training)} tasks, eval {len(evaluation)} tasks -- {split}")
    print(f"train balance {_class_balance(training)}")
    print(f"eval  balance {_class_balance(evaluation)}")
    progress(
        "adapter.data",
        n_train=len(training),
        n_eval=len(evaluation),
        train_balance=_class_balance(training),
        eval_balance=_class_balance(evaluation),
    )

    # from_pretrained can return None for a name it cannot resolve to a tokenizer.
    # Unguarded, the next line raises AttributeError on NoneType several frames from
    # the cause; on a GPU allocation that is a confusing way to lose the allocation.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer is None:
        raise SystemExit(f"no tokenizer for {args.model!r}: check the model name")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # attn_implementation is explicit, not incidental. Left to the default,
    # transformers selects a fused attention path that JIT-compiles a CUDA shim
    # through Triton, and on this cluster that compile cannot succeed: the system
    # gcc is deliberately hidden behind /tools/bin/blindfold/gcc, which reports
    # "hidden from view" and exits non-zero. A diagnostic run confirmed that both
    # `sdpa` and `eager` generate without touching Triton at all.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn,
    )

    # ---- baseline: the same checkpoint, untrained ---------------------------
    base_result = None
    if not args.skip_base_eval:
        print("\n>>> baseline: this checkpoint, rule withheld, no training")
        base_result = evaluate(model, tokenizer, evaluation, label="base")
        print(json.dumps(base_result.as_dict(), indent=2))
        record("adapter.base_f1", base_result.f1, model=args.model)

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
    # Recorded in the artifact and not only printed. These two numbers are the
    # federation payload: a fleet ships adapters rather than models, so what crosses
    # the wire per round is `trainable` times the dtype width. Leaving them in a log
    # meant the one edge-cost figure the design implies could not be read off any
    # committed artifact, only recovered from a cluster job's stdout.
    lora_size = {"trainable_params": trainable, "total_params": total}
    if not 0 < trainable < total:
        raise SystemExit(
            f"peft did not freeze the base: {trainable} trainable of {total}. "
            "Training every parameter is not the experiment."
        )

    # Whose labels the adapter learns from. With no reviewer this is the world's
    # own truth, which is finding 6. With one, it is that reviewer's opinion, and
    # the two evaluations below are what separate learning the rule from learning
    # the teacher.
    teacher = None if args.reviewer is None else resolve_reviewer(args.reviewer)
    if teacher is None:
        train_targets = world_targets(training)
        eval_teacher_targets = None
    else:
        train_targets = review_targets(training, teacher, seed=args.seed)
        eval_teacher_targets = review_targets(evaluation, teacher, seed=args.seed)
        agree = sum(1 for t in training if train_targets[t.task_id] == t.significant)
        print(
            f"\nteacher {teacher.name}: needs {teacher.escalation_threshold} of 3, "
            f"slip {teacher.slip_rate:.0%}"
        )
        print(f"training targets agreeing with the world: {agree}/{len(training)}")
        progress(
            "adapter.teacher",
            reviewer=teacher.name,
            threshold=teacher.escalation_threshold,
            slip_rate=teacher.slip_rate,
            target_agreement=round(agree / max(len(training), 1), 4),
        )

    rows = [
        tokenize_masked(e, tokenizer, args.max_len) for e in build_examples(training, train_targets)
    ]

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

    out_dir = _scratch_dir()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            # A float below 1 means the same thing `warmup_ratio` did: a fraction of
            # total steps. The two arguments were merged in transformers 5, which is
            # why the extra now floors there -- on 4.x this field is an int and 0.1
            # truncates to zero, silently training with no warmup at all.
            warmup_steps=0.1,
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
    print(json.dumps(tuned_result.as_dict(), indent=2))

    # Scored a second time against the teacher's own answers rather than the
    # world's. This is the measurement the experiment exists for: a model that
    # matches its teacher and not the world has learned the analyst's standard,
    # which is what the design calls personalization and what finding 8 calls the
    # risk. Reporting only the first number would make the two indistinguishable.
    against_teacher = None
    if eval_teacher_targets is not None:
        against_teacher = evaluate(
            model,
            tokenizer,
            evaluation,
            label="adapter-vs-teacher",
            truth=eval_teacher_targets,
        )
        print("\n>>> adapter scored against the TEACHER's answers, not the world's")
        print(json.dumps(against_teacher.as_dict(), indent=2))
    record("adapter.f1", tuned_result.f1, model=args.model)

    # ---- verdict against the two anchors -----------------------------------
    print("\n" + "=" * 72)
    if base_result:
        print(f"base     F1 {base_result.f1:.3f}  acc {base_result.accuracy:.3f}")
    print(f"adapter  F1 {tuned_result.f1:.3f}  acc {tuned_result.accuracy:.3f}")
    print(f"ceiling  F1 {CEILING_F1:.3f}  (rule stated, checklist prompt)")
    if base_result:
        gap = CEILING_F1 - base_result.f1
        closed = (tuned_result.f1 - base_result.f1) / gap if gap > 0 else 0.0
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
                    "lora": lora_size,
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
                        "eval_seed": args.eval_seed,
                        "cross_corpus": args.eval_seed is not None,
                    },
                    "training_loss": final_loss,
                    "base": base_result.as_dict() if base_result else None,
                    "adapter": tuned_result.as_dict(),
                    # Present only for a review-taught run. Absent means the targets
                    # were the world's own, so "against the teacher" is the same
                    # question as "against the world" and a second row would imply a
                    # comparison that was not made.
                    "teacher": (
                        None
                        if teacher is None
                        else {
                            "reviewer": teacher.name,
                            "escalation_threshold": teacher.escalation_threshold,
                            "slip_rate": teacher.slip_rate,
                            "train_target_agreement": round(
                                sum(
                                    1 for t in training if train_targets[t.task_id] == t.significant
                                )
                                / max(len(training), 1),
                                4,
                            ),
                        }
                    ),
                    "adapter_vs_teacher": (
                        None if against_teacher is None else against_teacher.as_dict()
                    ),
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
