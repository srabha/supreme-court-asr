# Supreme Court hearing ASR

Builds an ASR training set from Supreme Court of India hearing recordings and their published transcripts, then fine-tunes Whisper (tiny or small, LoRA) on it and measures WER/CER against the zero-shot model.

The hearings are long (11 min to almost 6 h) and the transcripts are lightly edited rather than verbatim, so most of the work is in `sc-asr data-prep`: it lines up each transcript with a first-pass ASR run, cuts 2–20 s chunks at silences where both agree, and drops chunks where they don't. See [REPORT.md](REPORT.md) for the approach and results.

## Setup

Tested with Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[data,dev]"
```

`data` adds the data-prep dependencies (PyAV, pypdf, faster-whisper, RapidFuzz, …). No system ffmpeg is needed. Data prep itself doesn't use torch; training and evaluation do.

## Data

`scripts/download_data.py` downloads the MP3s and PDF transcripts listed in the assignment CSV into `sc_hearings/mp3/` and `sc_hearings/transcripts/`. Audio and transcripts are matched by name (`001_161132009_2023-12-12.mp3` ↔ `161132009_2023-12-12.pdf`).

## Usage

```bash
sc-asr data-prep --config configs/data_prep_subset.yaml       # 6 hearings, ~4 h of audio
sc-asr evaluate --manifest data/subset/manifests/hf/test.jsonl --output results/whisper-tiny-subset-zeroshot.json
sc-asr train --train data/subset/manifests/hf/train.jsonl --validation data/subset/manifests/hf/validation.jsonl \
    --output models/whisper-tiny-subset-e5 --epochs 5 --batch-size 1 --gradient-accumulation-steps 1 \
    --warmup-steps 20 --learning-rate 5e-4
sc-asr evaluate --manifest data/subset/manifests/hf/test.jsonl --model models/whisper-tiny-subset-e5 \
    --output results/whisper-tiny-subset-e5.json
```

These are the settings of the latest reported run (5 epochs, see [REPORT.md](REPORT.md#subset-run-6-hearings-41-h-of-audio)). `--base-model` defaults to `openai/whisper-tiny`; pass `openai/whisper-small` for the larger model. `evaluate` without `--model` scores the base model as-is, which is the baseline.

On Windows, `scripts/run_subset_cpu.ps1` runs all four steps with these settings as defaults (about 4 h on a laptop CPU the first time, mostly data prep; retraining alone with `-SkipDataPrep` takes about 15 min).

### data-prep stages

| Stage | | Output |
| --- | --- | --- |
| 0 | pair audio with PDFs, decode to 16 kHz mono WAV | `wav16k/` |
| 1 | extract the spoken text from each PDF (drops cover page, line numbers, timestamps, speaker labels) | `text/` |
| 2 | Silero VAD speech map, then faster-whisper word timestamps in ~20 min windows | `vad/`, `asr/` |
| 3 | align transcript to ASR words, cut chunks at silences, filter | `segments/` |
| 4 | split by case, write chunks and manifests | `chunks/`, `manifests/`, `reports/` |

Each stage skips work that already exists, so an interrupted run can just be restarted. `--stage` / `--stop-stage` run a range, e.g. `--stage 3` to re-cut after changing thresholds. All settings are in `configs/data_prep.yaml`; `hearings:` limits a run to some hearings and `case_splits:` pins cases to a split.

Outputs (under `output_dir`):

- `manifests/hf/{train,validation,test}.jsonl` — `audio`, `text`, … ; read by `sc-asr train` / `evaluate`
- `manifests/nemo/{train,validation,test}.json` — NeMo format (`audio_filepath`, `duration`, `text`, …), for Parakeet or other NeMo models
- `chunks/<split>/*.wav` — 16 kHz mono; manifest paths are relative, so the folder can be moved
- `reports/` — pairing, alignment stats and reject reasons per hearing, split summary, and `review_sample.csv` with 50 random chunks to listen to

`text` is the transcript as written (case, punctuation, digits) and is the training label. `text_norm` (lowercase, numbers spelled out) is only used for matching and scoring.

## Layout

```
src/sc_asr/
  cli.py            sc-asr entry point
  train.py          Whisper + LoRA fine-tuning
  evaluate.py       WER/CER on a manifest, with or without an adapter
  metrics.py        scoring normalization; WER, CER, substitution/deletion/insertion rates
  data/             data-prep pipeline (pairing, transcripts, vad, anchors, align, export, ...)
configs/            data-prep settings (full set, CPU subset)
scripts/            data download, CPU subset run
tests/              pytest
```

## Tests

```bash
pytest
```
