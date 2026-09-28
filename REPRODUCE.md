# Reproducing every result

CPU-only, ~15 GB RAM is enough; no GPU is required. Total wall time on a
12-core CPU is roughly 8 hours, dominated by activation extraction.

```bash
pip install -r requirements.txt
```

## 1. Model selection (Phase 1)

```bash
python scripts/01_vocab_and_benchmark.py            # 7 candidates, ~25 min
```
Writes `docs/model_benchmark.json|md`. `meta-llama/Llama-3.2-1B` and
`google/gemma-2-2b` are gated on the Hub and will be recorded as unevaluated
unless you set `HF_TOKEN`.

The vocabulary check runs three minimal pairs per term. Terms scoring below
~0.6 for the model you pick should be dropped from the lexicon before
generating data. `pythia-410m` fails `matte` (0.40), which is why `gpt2` is
the default despite a marginally lower overall score.

## 2. Datasets

The pilot narratives are committed, so a replication runs on byte identical data.
Fingerprints: `natural.jsonl` sha256 `2cc9a1c85019ad94`, `shuffled.jsonl` sha256
`180476f26a17be60`, 1000 narratives each. Regenerating them reproduces those
bytes; before correction 21 it did not, because the generator seeded from
Python's per process randomised string hash.

```bash
python scripts/02_generate_dataset.py --name pilot --entities lips eyes \
    --attributes color finish --n 1000
python scripts/02_generate_dataset.py --name full  --n 1800
# length control: 2 entities held at the 4-entity narrative length
python scripts/02_generate_dataset.py --name long2 --entities lips eyes \
    --attributes color finish coverage --n 1800 \
    --min-sentences 10 --max-sentences 16
```

## 3. Activations (the slow step)

```bash
python scripts/03_extract.py --dataset full --model gpt2      # ~2 h
```
Stores both read-out conventions: `X.npy` (last token) and `Xmean.npy`
(mean-pooled over the prefix), plus `E.npy` (non-contextual embedding
baseline). ~1 GB per condition for the full set.

## 4. Everything else

The three orchestration scripts run the phases in order and are the exact ones
used for the reported numbers:

```bash
bash scripts/run_full_all.sh            # full 4-entity: all phases
bash scripts/run_length_control.sh      # entity count vs narrative length
bash scripts/run_pythia_replication.sh  # second model
```

Each waits on the previous one's completion marker, so they can be launched
together.

## Individual analyses

```bash
python scripts/04_probe.py    --dataset full --model gpt2 --layer-stride 2
python scripts/15_binding_by_distance.py --dataset full --model gpt2
python scripts/16_intervene_v2.py --dataset full --model gpt2 \
    --entity lips --other-entity eyes --attribute finish
python scripts/17_significance.py --dataset full --model gpt2
python scripts/09_report.py   --dataset full --model gpt2
```

## Conventions used throughout

- p-values in the reports are **two-sided**; one-sided values are retained in
  the JSON as `p_one_sided_greater`.
- Confidence intervals over narrative positions use a **cluster bootstrap
  resampling whole narratives**, because positions inside a narrative share its
  wordings and its state and are not independent.
- Causal claims use a steering gain of **g = 2**, in units of the full
  direction's norm. At g = 8 a random direction of matched norm also reaches
  significance, which puts the edit off distribution. The gain is written g to
  keep it apart from the significance level.
- MDL is reported over a **sweep of the probe's regularisation strength**; a
  single arbitrary `C` produces compression ratios anywhere from 0.75x to
  1.44x on identical data.

## Result files

`results/*.json` are machine-readable; `docs/RESULTS_<dataset>_<model>.md` are
the generated reports. `docs/CORRECTIONS.md` records every claim that was
revised, with cause and fix.

## Running the scale arm on a GPU

`colab_scale_arm.ipynb` runs the fourth model, Qwen3-1.7B-Base, on a free Colab
T4 in about twenty minutes. The same arm takes roughly four hours on a 15 GB CPU
machine and pages heavily, since the model needs 6.8 GB in float32. The notebook
clones this repository, checks the dataset fingerprints, runs extraction,
probing, both read out checks and the decomposition, then prints the four model
comparison and zips the result files.

## The write-up

`docs/index.html` is built from three sources:

    results/_paper_head.html        markup and styles
    results/_paper_script.html      chart code
    results/_paperdata_inline.json  every number the charts draw

Rebuild it after editing any of them:

    python scripts/25_build_paper.py

`.github/workflows/pages.yml` runs that build on every push to main and
publishes `docs/` to GitHub Pages. Enable it once under Settings, Pages, Source,
GitHub Actions.
