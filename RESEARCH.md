# Research

External evidence and the citations for it. Three files carry different things
and it is worth keeping them apart:

| File | Holds |
| --- | --- |
| `docs/` | How the pieces work and how to run them |
| `docs/findings.md` | What has been measured here, and the script that reproduces it |
| **`RESEARCH.md`** | **What is true outside this repo, and where that was checked** |
| the manuscript | The argument all three feed |

Every entry below was checked against arXiv, a DOI, or the publisher's own page
on the date given. Nothing is cited from memory. If an entry cannot be tied to a
specific claim in this repo, it does not belong here.

## What Pharos is, before what it is for

Pharos is a **labeled corpus generator with an acceptance gate**. The disclosure
boundary is the application that motivated building it and the one claim no public
corpus can substitute for, but it is not the extent of what the artifact does, and
scoping this file to that one claim once caused a real misjudgement about which
external data was worth looking at.

Three capabilities are general, and none of them is about disclosure:

- **The gate is a benchmark-validity instrument.** "Content-defined ground truth
  cannot have a chance-level surface baseline" is a statement about planted ground
  truth in any domain, not about compartments. It is why the paper sits against
  Geirhos, Gururangan, and SWAG rather than only against the federated-learning
  literature, and it is testable *outside* this generator.
- **The task is a rule-acquisition benchmark.** Conjunctive triage with the rule
  withheld has a measured floor and a demonstrated ceiling of F1 1.000. The
  six-model sweep result -- recall exactly 1.000 everywhere, 3B outscoring 14B --
  is a finding about that task, and would stand if the labels carried no lattice.
- **The reproducibility machinery is reusable.** Provenance stamps, the staleness
  guard, cross-platform bit-identical gating, and a retraction the artifacts caught
  are contributions to how generated corpora are released, independent of subject.

So the survey below answers a narrow question -- *which corpora could stand in for
Pharos on the boundary claim* -- and the answer is none. It does **not** establish
that Pharos has one use, and it should not be read that way.

## The claim the survey defends

The README and the docs site both open on the same claim: *Pharos supplies what
public corpora do not.* That is the load-bearing sentence for the boundary work, so
it is worth being exact about what it rests on. What follows is the survey.

It read "the one thing no public corpus does" until this file existed. The
quantifier went because an absolute claim asserts an exhaustive search that was
never run, and the discriminator (incomparability, not merely labels) went in
because it is the specific, checkable thing public corpora lack.

### The closest public corpora, and where each stops

| Corpus | What it supplies | Label structure | Why it cannot measure a disclosure boundary |
| --- | --- | --- | --- |
| **TAB** | 1,268 English ECHR court cases, human-annotated spans of personal information | Per-span: mask or do not mask, plus a semantic category, identifier type, and a confidential-attribute flag | Flat. There is no level ladder to compare and no compartment set to be incomparable under. It answers *does this span identify someone*, not *is this holder entitled to this fact* |
| **MIND** | ~160k English news articles, click logs over 1M users | None | Real per-user behavior and nothing about disclosure |
| **Enron** | ~500k real messages across ~150 mailboxes | None | Real organizational structure, no policy labels on any object |
| **TREC RAGTIME** | Multilingual news with report requests, graded on whether claims cite supporting evidence | None on documents | The evaluation grades provenance of *claims*, which is adjacent and useful, but no document carries a releasability label |
| **LEAF** | Federated benchmark datasets with natural client partitions | None | Solves the partition problem, not the labeling one |

They fail in the same way, and it is not a coincidence: **each supplies either the
task or the partition, and none supplies the lattice.** A disclosure boundary is
interesting exactly where two holders at the same sensitivity level, with
different need-to-know compartments, dominate each other in neither direction.
That requires a product lattice over the objects. Public corpora carry at most a
single ladder, where every label is comparable to every other and a join is a
maximum, which is the case the design is *least* worried about.

TAB deserves the precision because it is the closest and it is genuinely a
privacy corpus, not a repurposed one. Its annotations are richer than a binary
mask: semantic category, identifier type, confidential attributes, co-reference.
What it is not is *ordered*. Nothing in TAB says one holder may read a span and
another may not, so nothing in TAB can be wrong in the direction Pharos measures.
TAB measures whether an identity leaks. Pharos measures whether a policy holds
when the labels involved are incomparable.

### What this survey does not license

Five corpora, each chosen because it is the closest along one axis. That is an
argument from the nearest candidates, **not** a proof, and "no public corpus" should
be written as what it is: no corpus among the closest ones surveyed here. If a
corpus with a genuine compartment lattice turns up, it changes the framing of the
whole testbed and belongs in this file the day it is found.

## What Pharos should be used *with*

Pharos buys internal validity for the boundary and pays for it in realism: a
generated maritime world, a fixed fact vocabulary, officer voices from templates,
a plant defined by a three-fact conjunction. The corpora above buy back most of
what that costs, which makes them complements rather than alternatives.

| Claim under test | Use | Why |
| --- | --- | --- |
| The governed join enforces policy when labels are incomparable | Pharos only | Nothing public carries the lattice |
| An analyst-shaped output can be graded on the sources it is entitled to claim | RAGTIME | Its citation-based evaluation is the closest public analogue to the provenance join, and it is multilingual and adversarially curated |
| Personalization helps under a realistic non-IID split | MIND (per user) or Enron (per mailbox) | Natural partitions, not a random shard. A random shard is IID by construction and flatters personalization |
| Leakage detection against human ground truth | TAB | Human-annotated spans mean a false negative is a real one, not a generator artifact |
| Federated baselines and harness sanity | LEAF | Established partitions and reference implementations |

The asymmetry is the useful part. Pharos is the only place the boundary claim can
be measured, and the *worst* place to make a claim about realism. Any result that
depends on prose looking real, on vocabulary breadth, or on genuine user
behavior should be replicated on one of the others before it is written down.

### An external test the gate result deserves -- DONE 2026-07-31

**Result: the claim generalises.** Three public corpora, same probe, each compared
against its own permutation null: `ag_news` z=+8.09, `imdb` z=+7.01, and
adversarially filtered `hellaswag_endings` z=+3.65. All above null. Leakage falls
monotonically with construction care but does not reach chance even under adversarial
filtering, which is the standard remedy. Artifact:
`results/external_gate_validation.json`; method in `scripts/validate_gate_externally.py`.

The original framing of this item follows, kept because the prediction was made
before the measurement.

### The original open item

The calibration-instrument finding is stated as a general property of
content-defined ground truth, but it has only ever been measured on our own
generator. That is thin support for a claim that broad. Running the same
surface-only probe against a public benchmark whose positive class is *also* defined
by content would test it where we did not build the data: if such a corpus also
fails to reach a chance-level surface baseline, the claim generalises; if it reaches
chance, the finding is about our vocabulary rather than about content-defined labels,
and the paper must say so.

This is worth doing **before** step 3, unlike everything else in the table above,
because it validates a claim the resource paper already makes.

## Citations

Verified 2026-07-31 unless noted.

### The label lattice

- Denning, D. E. (1976). A Lattice Model of Secure Information Flow.
  *Communications of the ACM* 19(5), 236-243.
  [doi:10.1145/360051.360056](https://doi.org/10.1145/360051.360056)
  **Grounds** `pharos.labels`: security classes form a lattice, and the label of a
  derived object is the join over its inputs. The construction in
  `docs/reference/label-lattice.md` is this model with compartments as the
  non-total component.

### Attribution, and finding 1

- Koh, P. W., & Liang, P. (2017). Understanding Black-box Predictions via
  Influence Functions. *ICML 2017*. [arXiv:1703.04730](https://arxiv.org/abs/1703.04730)
  **Grounds** the README's claim that leave-one-out is "the ceiling that cheaper
  estimators approximate". Influence functions are an efficient approximation to
  leave-one-out retraining, so a failure measured at exact leave-one-out bounds
  the cheaper family rather than merely differing from it. This is the citation
  that makes finding 1 a statement about attribution and not about one script.

### The shortcut gate

- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M.,
  & Wichmann, F. A. (2020). Shortcut Learning in Deep Neural Networks.
  *Nature Machine Intelligence*. [arXiv:2004.07780](https://arxiv.org/abs/2004.07780)
  **Grounds** why the gate exists at all: a model that can predict the label from
  surface form has learned the insertion artifact, and a shared adapter would
  propagate that shortcut to every deployment in the fleet.

### Federated personalization

- McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017).
  Communication-Efficient Learning of Deep Networks from Decentralized Data.
  *AISTATS 2017*, PMLR 54. [arXiv:1602.05629](https://arxiv.org/abs/1602.05629)
  **Grounds** the aggregation baseline the design departs from.

- Arivazhagan, M. G., Aggarwal, V., Singh, A. K., & Choudhary, S. (2019).
  Federated Learning with Personalization Layers.
  [arXiv:1912.00818](https://arxiv.org/abs/1912.00818) (preprint, no venue)
  **Grounds** the two-adapter split's ancestry: a shared base plus client-retained
  layers. The difference worth stating in the manuscript is that FedPer splits by
  *architecture* while Pharos splits by *label*, decided before any gradient.

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., &
  Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
  **Grounds** the adapter mechanism the critical-path experiment will train.

- Sun, Y., Li, Z., Li, Y., & Ding, B. (2024). Improving LoRA in
  Privacy-preserving Federated Learning. *ICLR 2024*.
  [arXiv:2403.12313](https://arxiv.org/abs/2403.12313)
  **Grounds** the sharpest technical point available to the design. Clients jointly
  optimize both low-rank factors while the server aggregates them separately, and
  that discordance both destabilizes training under heterogeneity and *amplifies*
  the noise added for differential privacy. FFA-LoRA's answer is to freeze one
  factor. This is the paper behind any phrase like "aggregation corrected for
  averaging error", and it is why naive FedAvg over LoRA factors is not a
  defensible baseline to ship.

### Learning from analyst decisions

Verified 2026-08-01. These ground build-order step 3, which is what finding 6 leaves
open: the rule is gradient-learnable from clean labels, and what remains unmeasured
is whether it is learnable from the decisions an analyst actually makes.

- Gao, G., Taymanov, A., Salinas, E., Mineiro, P., & Misra, D. (2024). Aligning LLM
  Agents by Learning Latent Preference from User Edits.
  [arXiv:2404.15269](https://arxiv.org/abs/2404.15269) (v3 2024-11-23; the record
  carries no journal reference)
  **Grounds** that edit feedback is a studied supervision signal rather than an
  invention of this design. Their PRELUDE setting is the same interaction shape --
  the agent answers, the user optionally edits, and the edit carries preference
  beyond correctness -- and their cost metric is accumulated edit distance.

- Misra, D., Pacchiano, A., Chi, T.-C., & Gao, G. (2026). Principled Fine-tuning of
  LLMs from User-Edits: A Medley of Preference, Supervision, and Reward.
  [arXiv:2601.19055](https://arxiv.org/abs/2601.19055) (accepted at NeurIPS 2025 per
  the arXiv comment field)
  **Grounds** the supervision question directly, and sharpens how step 3 should be
  framed. Their result is that user-edit data unifies three feedback types usually
  studied apart -- preference, supervised label, and cost -- that learners built on
  each carry *different* bounds depending on the user and data distribution, and
  that ensembling across them beats any one alone. So "can the rule be learned from
  analyst decisions" is not one question, and a negative result on one feedback type
  would not settle it.

- Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., & Kiela, D. (2024). KTO:
  Model Alignment as Prospect Theoretic Optimization. *ICML 2024*.
  [arXiv:2402.01306](https://arxiv.org/abs/2402.01306) (v4 2024-11-19)
  **Grounds** the feedback shape a bare accept or reject actually has. KTO learns
  from unpaired binary judgements on single responses rather than from pairs, which
  is what a review stream supplies: a reviewer says whether *this* proposal will do,
  never which of two is better. That is why finding 7 counts accepted and rejected
  decisions separately from revised ones rather than treating the stream as
  preference data.

- Xu, Y., & Jurgens, D. (2026). Beyond Consensus: Perspectivist Modeling and
  Evaluation of Annotator Disagreement in NLP.
  [arXiv:2601.09065](https://arxiv.org/abs/2601.09065) (v2 2026-01-17; survey, no
  journal reference on the record)
  **Grounds the decision not to simulate reviewers with a language model.** On
  persona prompting and demographic conditioning the survey states that "persona
  variables explain only a small fraction of variance", that such approaches are
  "highly sensitive to prompts and model choice, and risk amplifying stereotypes",
  and that "LLM judgments compress human disagreement and rely on opaque priors",
  raising "concerns about reliability and epistemic validity of replacing human
  annotation labor". Putting that in the position of the feedback-generating process
  would make the instrument's own error the largest unmeasured term in the
  experiment. `pharos.analyst` is therefore a parameterised decision procedure, which
  buys ground truth by construction and buys no claim about human behaviour.

**State the delta narrowly.** Both edit-feedback papers learn from free-text edits to
a prose response. A Pharos analyst decision is accept, revise, or reject over a
*verdict and its governed label*, which is a lower-bandwidth signal on a discrete
output. The published bounds therefore bear on the shape of step 3 and do not
transfer to it numerically. The same caution applies in reverse to finding 7: its
reviewers are parameters, so its numbers bound a mechanism and estimate no
population.

### Privacy and adversaries

- Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., &
  Zhang, L. (2016). Deep Learning with Differential Privacy. *ACM CCS 2016*,
  308-318. [arXiv:1607.00133](https://arxiv.org/abs/1607.00133)
  **Grounds** the per-node privacy accounting.

- McMahan, H. B., Ramage, D., Talwar, K., & Zhang, L. (2018). Learning
  Differentially Private Recurrent Language Models. *ICLR 2018*.
  [arXiv:1710.06963](https://arxiv.org/abs/1710.06963)
  **Grounds** user-level rather than example-level DP, which is the unit that
  matters when a client *is* an analyst.

- Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). Machine
  Learning with Adversaries: Byzantine Tolerant Gradient Descent. *NIPS 2017*,
  119-129. [dblp](https://dblp.org/rec/conf/nips/BlanchardMGS17.html)
  **Grounds** Krum and the distance-based robust-aggregation family whose
  assumptions low-rank factorization breaks.

- Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., & Shmatikov, V. (2020). How To
  Backdoor Federated Learning. *AISTATS 2020*, PMLR 108, 2938-2948.
  [arXiv:1807.00459](https://arxiv.org/abs/1807.00459)
  **Grounds** the threat model, including that the attack was demonstrated
  *together with* evasion of anomaly detection. A defense evaluated only against
  naive poisoning is not evaluated against this literature.

### Releasing a corpus

- Akhtar, M., et al. (2024). Croissant: A Metadata Format for ML-Ready Datasets.
  *NeurIPS 2024 Datasets and Benchmarks Track*.
  [arXiv:2403.19546](https://arxiv.org/abs/2403.19546)
- Jain, N., et al. (2024). A Standardized Machine-readable Dataset Documentation
  Format for Responsible AI. [arXiv:2407.16883](https://arxiv.org/abs/2407.16883)
- Croissant RAI specification:
  <https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html>
  **Ground** `pharos.croissant` and `docs/releasing.md`. RAI properties carry the
  `rai:` prefix and cover the data life cycle, labeling, and known limitations,
  which is where a generated corpus is obliged to say so.

### Corpora

- Pilán, I., Lison, P., Øvrelid, L., Papadopoulou, A., Sánchez, D., & Batet, M.
  (2022). The Text Anonymization Benchmark (TAB): A Dedicated Corpus and
  Evaluation Framework for Text Anonymization. *Computational Linguistics* 48(4),
  1053-1101. [arXiv:2202.00443](https://arxiv.org/abs/2202.00443) ·
  [code](https://github.com/NorskRegnesentral/text-anonymization-benchmark)

- Wu, F., Qiao, Y., Chen, J.-H., Wu, C., Qi, T., Lian, J., Liu, D., Xie, X.,
  Gao, J., Wu, W., & Zhou, M. (2020). MIND: A Large-scale Dataset for News
  Recommendation. *ACL 2020*, 3597-3606.
  [doi:10.18653/v1/2020.acl-main.331](https://doi.org/10.18653/v1/2020.acl-main.331) ·
  [site](https://msnews.github.io/)
  The paper's abstract states 1M users and 160k+ articles. The frequently quoted
  "15M impression logs" is from the dataset site, not the paper, and should be
  attributed there.

- Klimt, B., & Yang, Y. (2004). The Enron Corpus: A New Dataset for Email
  Classification Research. *ECML 2004*, LNCS 3201, 217-226.
  [doi:10.1007/978-3-540-30115-8_22](https://doi.org/10.1007/978-3-540-30115-8_22) ·
  [CMU distribution](https://www.cs.cmu.edu/~enron/)

- Lawrie, D., MacAvaney, S., Mayfield, J., Soldaini, L., Yang, E., & Yates, A.
  (2026). Overview of the TREC 2025 RAGTIME Track.
  [arXiv:2602.10024](https://arxiv.org/abs/2602.10024) ·
  [track site](https://trec-ragtime.github.io/)

- Caldas, S., Duddu, S. M. K., Wu, P., Li, T., Konečný, J., McMahan, H. B.,
  Smith, V., & Talwalkar, A. (2018). LEAF: A Benchmark for Federated Settings.
  [arXiv:1812.01097](https://arxiv.org/abs/1812.01097) (preprint)

## The LAS-cleared corpus list (received 2026-07-31)

The corpus table above was assembled from a literature survey. This is the list LAS
actually cleared, which is a different question, and it changes what is worth
pursuing. Licences are as stated on the sheet; verify before publishing anything
derived from one.

**Directly loadable today.** `trec-ragtime/ragtime2` on the Hub is ungated,
CC-BY-SA-4.0, and its files are `eng/rus/spa/zho`, which confirms the 2026 language
set. Note what it is: a *document collection*, four languages of news, with no
relevance judgements in the public repo. Those come with TREC participation, so
RAGTIME cannot serve as an external gate corpus today -- there is no content-defined
label to probe. It is still the right source for the triage-to-draft task once the
qrels are in hand. `talkbank/callhome` is also public (CC-BY-NC-SA-4.0);
`Salesforce/DiverseSumm` returned 401 and needs auth.

**The three entries that matter most to Pharos are not public.** They have to be
requested from Bo Light (jjlight@ncsu.edu):

- **Synthetic Org Chart v2 and v3.** v2 adds noise, lateral edges, and email headers;
  v3 adds email bodies and calendar invites. That is an organisational structure with
  per-person message traffic, which is a far better fit for the non-IID fleet
  partition than a random shard, and closer to the compartment story than anything on
  the public list. The sheet notes v3 is expensive to generate and must be asked for.
- **Synthetic Suspicious Activity Reports** (provided by SAS). Structurally the
  nearest public-ish analogue to what Pharos generates: short analyst-written reports
  about entities, with the significance judgement that matters embedded in the text.
  If Pharos's realism is ever challenged, this is the corpus to be challenged against.
- **LAS Zendia** (SCADS 2025). Synthetic scenario data, also via Bo.

**What the list does not contain.** MIND and the Text Anonymization Benchmark are
absent. Both are public and usable regardless, but neither is LAS-cleared, and the
org-chart corpora are a better source of per-user behaviour for this application than
MIND is, because they carry the organisational structure the compartment model needs
and MIND does not.

**Multilingual and adjacent, if breadth is ever needed.** TelegramDB (CC-BY-4.0;
Russian, English, Farsi) is multi-channel messaging, which is the closest public
analogue to Pharos's several-channels-report-one-event structure. TREC AutoJudge
pilot data is directly relevant to calibrating a model-as-a-judge. The LDC packs are
government-use and several are not yet released.

## Facts here that will rot

Recorded because each one already caused a wrong statement somewhere.

**RAGTIME changes year to year and the year must be named.** The 2025 collection is
Arabic, Chinese, English, and Russian; the 2026 track site states English,
Spanish, Chinese, and Russian. Auto-nuggetization is new in 2026. The evaluation
is described on the track site as citation-based, meaning claims are graded on
supporting evidence, so calling RAGTIME "nugget-scored" is wrong: nuggetization is
a *participant task*, not the scoring method. Quoting the 2025 overview paper for
a 2027 plan produced exactly this error once already.

**Enron is quoted at many sizes, and the figures come from the distribution, not
the paper.** The CMU page for the 2015-05-07 version states "about 150 users" and
"a total of about 0.5M messages". Those are the numbers used in the table above.
The ECML paper describes an earlier state of the corpus, so any count more precise
than these needs the release version named alongside it.

## Open: claims not yet grounded

Kept explicit so silence does not read as coverage.

- **Bell-LaPadula** is the conventional citation for a levels-plus-compartments
  model and is *not verified here*, so it is deliberately absent from the citation
  list above rather than cited from memory.
- ~~**The capacity dimension of the lattice has no external grounding at all.**~~
  **Closed 2026-07-31: it is not original.** Sabelfeld and Sands classify
  declassification along four dimensions -- *what* is released, *who* releases it,
  *where*, and *when* -- and constraining release by the form of the output is an
  instance of the **what** dimension. Verified by DOI content negotiation, *Journal
  of Computer Security* 17(5):517-548, 2009,
  [doi:10.3233/JCS-2009-0352](https://doi.org/10.3233/JCS-2009-0352). Cited as
  `sabelfeld2009declassification`. The manuscript now says so explicitly rather
  than leaving a reviewer to notice. What remains ours is the *consequence* --
  that without it a join over eight sources collapses federation -- which is
  measured, not asserted.
- **Finding 3b's "scale does not help"** rests on six models at n=40 here and no
  external corroboration.
- ~~**Accept / revise / reject as a supervision signal** has no citation yet.~~
  **Closed 2026-08-01.** The published framing is learning from *user edits*, and it
  is now cited above: Gao et al. 2024 for the interaction shape, Misra et al. 2026
  for the result that edit data unifies preference, supervision, and cost under
  different bounds. What is still open is narrower and worth keeping separate: no
  published work measures this signal over a **discrete governed label**, which is
  the form a Pharos analyst decision takes.

## Adding an entry

1. Verify against arXiv, a DOI, or the publisher's page. Not from memory, and not
   from a search-result summary: open the record.
2. Record the verification date.
3. Name the claim, module, or finding in this repo that the entry grounds. An
   entry that grounds nothing is a reading list, and belongs somewhere else.
4. If verification *fails* or is not done, the entry goes under "Open" above
   rather than into the citation list.
