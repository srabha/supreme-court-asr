# Supreme Court hearing ASR — report

## Summary

The goal was to turn Supreme Court of India hearing recordings and their published transcripts into aligned training data, fine-tune an ASR model on it, and measure the result.

The data pipeline (`sc-asr data-prep`) is complete and tested end to end. It aligns each multi-hour hearing to its transcript using a first-pass ASR run as anchors, cuts 2–20 s chunks at silences, and drops chunks where the transcript and audio disagree. The model is Whisper (tiny by default, small optionally) fine-tuned with LoRA, chosen to fit the available compute: a laptop CPU and a free Colab T4. Parakeet would be the next model to try with a proper GPU.

On a 6-hearing subset (4.1 h of audio, 0.73 h kept), fine-tuning Whisper tiny with LoRA on CPU has not yet beaten the zero-shot model:
- **1 epoch:** test WER moved from 0.348 to 0.339, which is within noise. That run was also held back by a learning-rate warmup longer than the run itself.
- **5 epochs with that fixed:** the model overfit and falls into repetition loops on a few test chunks. With an output-length cap, WER is 0.448 against 0.348 zero-shot, mostly from inserted words (insertion rate 0.130 against 0.053). The bootstrap interval for the difference (−0.021 to +0.258) includes zero, and per chunk 11 improved and 17 got worse.

The data is the main constraint: 0.6 h of training audio is too little. The next steps are a decoding guard that prevents loops rather than only capping them, and more kept data (see [Future work](#future-work)).

The whole pipeline needs no manual annotation, and everything reported here ran on a laptop CPU at no cost.

## Data

- 24 hearing recordings (MP3, 94.6 h in total, 11 min to 5.8 h each) from 15 cases, with transcripts as text PDFs ("Transcribed by TERES"). One PDF was an exact duplicate.
- The transcripts are not verbatim. They are lightly edited, leave out some interjections and cross-talk, and write numbers and citations in legal style ("Section 2(21)", "11th of March, 2024").
- Each PDF starts with a cover page and a document-control block. The proceedings begin at a clock time ("10:45 AM IST"), every line ends with a line number, speakers are labelled ("CHIEF JUSTICE DY CHANDRACHUD:"), and the text ends with "END OF THIS PROCEEDING" or "END OF DAY'S PROCEEDINGS".

## Alignment and processing

Forced-aligning a whole hearing at once doesn't work well here. The files are hours long, and the aligner has no way to handle the places where the transcript departs from what was said. Instead, each hearing goes through five stages:

1. **Audio.** Pair each MP3 with its PDF by name and decode it to 16 kHz mono WAV.
2. **Transcript.** Extract the PDF text and keep only the proceedings. Everything before the first clock time or speaker label goes, as do line numbers, clock times, page headers, speaker labels, transcriber notes and the end marker.
3. **Anchors.** Build a speech/silence map with Silero VAD, then run faster-whisper (`small.en` on CPU, `medium.en` on GPU) to get a timestamp for every word. Long hearings are transcribed in roughly 20-minute windows cut at silences, which keeps memory flat.
4. **Alignment.** Both texts are normalised the same way (lowercase, numbers spelled out, no punctuation) and aligned word by word with an edit distance.
   - Cuts are only allowed inside runs of at least 3 exactly matching words, at a pause in the ASR output, and each cut is moved into the nearest silence from the VAD map. So every chunk starts and ends in silence with matching text on both sides.
   - A chunk is dropped if 4 or more words appear on one side only (skipped or added text, usually cross-talk), if its CER against the ASR is above 0.35, if the speaking rate is implausible, or if a VAD check finds speech at its edges.
5. **Export.** Split by case and write chunk WAVs and manifests.

The training label is always the transcript text, never the ASR output. The ASR is only used for timing and as a sanity check. Where the two disagree on a single word it's usually the ASR that's wrong, for example "learned solution" where the transcript has "learned Solicitor General".

Each chunk keeps two versions of its text:
- `text`: the transcript as written, with case, punctuation and digits. Whisper is trained on this.
- `text_norm`: the normalised form, used only for matching and scoring.

### Splits

Splits are made by case rather than by hearing. Several cases have more than one hearing, and splitting by hearing would put the same names, statutes and arguments in both train and test.

The full-dataset split is pinned in `configs/data_prep.yaml`:

| Split | Cases | Hearings | Audio |
| --- | --- | --- | --- |
| train | 90121999, 297962019, 92872006, 101772014, 202342022, 400992022, 2024042484-3, 279352017, 479282023, 103822024 | 18 | 77.6 h (82 %) |
| validation | 129962005, 118052024 | 2 | 5.8 h (6 %) |
| test | 296862017, 161132009, 255362010 | 4 | 11.2 h (12 %) |

How this split was chosen:
- The test set spans three cases rather than one large one, so the score covers several benches and topics.
- Validation is kept small (5.8 h); that's enough to pick a checkpoint, and the rest goes to training.
- The subset's held-out cases stay held out here (118052024 in validation, 296862017 in test), and none of the subset's training cases are in test. A model trained on the subset can therefore be scored on the full test set without leakage, and compared directly with a full-data model.
- The hours above are raw audio. The kept-chunk hours shift slightly because the share kept differs by hearing; `reports/splits.json` has the actual figures after a run.

The smaller runs use the same rule at a smaller scale:

| | Smoke | Subset | Full |
| --- | --- | --- | --- |
| Audio | 52 min, 3 hearings | 4.1 h, 6 hearings | 94.6 h, 24 hearings |
| Cases (train / val / test) | 1 / 1 / 1 | 3 / 1 / 1 | 10 / 2 / 3 |
| How split | automatic (seed 13) | pinned | pinned |
| Test chunks | 7 | 31 | roughly 1,000+ (estimate) |
| Purpose | check the code end to end | first real result on CPU | the real model (needs a GPU) |

Splitting by case doesn't separate speakers. The Chief Justice and several senior counsel speak in almost every hearing. So the test measures performance on new cases, not new voices, which matches how the model would be used.

### Output

The output is a Hugging Face-style JSONL manifest for Whisper, plus a NeMo manifest so the same data can be used for Parakeet later. Per-hearing reports record how many chunks were kept and why the others were rejected. A random 50-chunk sample is kept for listening checks.

### No manual annotation

No audio was labelled or timed by hand:
- **Labels** are the published transcripts, cleaned by rules (cover page, line numbers, timestamps, speaker labels and end markers removed).
- **Timing** comes from the anchor ASR's word timestamps and the VAD silence map.
- **Quality control** is automatic: the mismatch, CER, speaking-rate and edge-speech checks decide which chunks are kept, and every reject is logged with its reason.
- **Splits** follow case numbers, so no one has to judge which recordings belong together.

The only manual work was during development: reading rejected chunks to set the filter thresholds (see [Challenges](#challenges)) and pinning cases to splits in the config. Both are one-time settings in `configs/`, not per-hearing work, so adding hearings needs no human effort. The 50-chunk listening sample is an optional spot check, not a step the pipeline depends on.

## Challenges

**Broken downloads.** The first download attempt saved Dropbox web pages with an `.mp3` extension, and some files were really MP4/AAC. The download script now checks for this and re-encodes, and the pipeline checks each file's header before using it.

**PDF layout.** Early extractions mixed the cover page into the text, kept the line numbers (`uploaded 3 for us on the screen. 4`) and the closing marker. The cleaner now starts at the first clock time or speaker label and strips trailing line numbers. Two different end-marker wordings turned up across the 24 PDFs. Every transcript now reports whether both markers were found, and all 24 do.

**A filter that rejected good data.** The first version kept a chunk only if its CER against the ASR was under 0.12, which kept 4 chunks out of about 105 on the smoke set. Looking at the rejects, the transcript was right in most of them and `small.en` was wrong, typically on legal terms. That's exactly the material fine-tuning should learn from. The strict CER check was replaced with the mismatch check described above, with CER kept only as a loose limit.

**Clipped words.** Most of the remaining rejects had speech in the last 80 ms. Whisper's word end times tend to be early, so the cuts were landing inside the last word. Moving each cut into a real silence from the VAD map fixed most of these. With the two changes together the smoke set went from 4 kept chunks to 42.

**Limited hardware.** The development machine has 15 GB of RAM, mostly in use, and no GPU. One long run was killed for low memory. Data prep was rewritten to stream audio and process long hearings in windows, so memory doesn't grow with recording length, and to avoid torch entirely (Windows Smart App Control was also blocking torch's DLLs at the time). Full-set alignment and anything bigger than Whisper small need a GPU, such as a free Colab T4.

**Too few cases for a clean split.** With the 6-hearing subset, the automatic 80/10/10 split by case left train with less than an hour. Cases can now be pinned to a split in the config. The subset uses three cases for train (~3 h), one for validation and one for test.

**Overfitting and repetition loops.** With only 0.6 h of training audio, longer training made the model memorise. Over 5 epochs, train loss fell to about 0.2 while validation loss rose after epoch 2. The model also started falling back on frequent phrases ("my Lord", "the Supreme Court") and repeating them until the token limit, so a single chunk could dominate the test WER. Three things were done about it: the checkpoint with the lowest validation loss is kept, `sc-asr evaluate` caps output length per second of audio for every model, and each hypothesis's compression ratio is recorded so loops show up in the results. The cap limits the damage but does not stop loops. Temperature fallback and more data are the remaining fixes (see [Evaluation](#evaluation)).

## Model choice

Whisper with LoRA, tiny by default and small as the step up:

- **It fits the compute.** Whisper tiny (39M parameters) trains on a laptop CPU at about 2 s per step in under 2 GB of RAM, and in minutes on a T4. LoRA trains about 0.8 % of the weights, and the resulting adapter is around 1 MB.
- **It's a reasonable starting point.** Whisper is pretrained on a lot of varied, accented English and outputs cased, punctuated text, which matches the labels. Hugging Face Transformers and PEFT support it directly.
- **Its weak spots are what fine-tuning targets.** Small Whisper models get legal names and terms wrong, and Whisper can hallucinate on silence. The chunking keeps silence to a minimum.

Other options considered:

- **NVIDIA Parakeet-TDT-0.6B** is the recommended next step, skipped for now. It's likely the most accurate option here: it's near the top of the Open ASR leaderboard for English, handles accents well, decodes fast and hallucinates little. It needs a 16–24 GB GPU to fine-tune comfortably. The NeMo manifests are already generated, so no data changes would be needed.
- **icefall Zipformer** (`pruned_transducer_stateless7`). The available checkpoint is trained on LibriSpeech, read US English audiobooks, which is far from this audio. It needs a Linux/CUDA k2 build and outputs uppercase text without punctuation. It would make sense mainly for streaming or on-device use.
- **NeMo Conformer-CTC**: older and weaker than Parakeet, and no punctuation.

### Compute and cost

Everything reported here ran on a laptop CPU (no GPU, 15 GB RAM shared with other apps). Nothing was paid for.

| Step | Run | Time on laptop CPU | Notes |
| --- | --- | --- | --- |
| Data prep | smoke, 52 min of audio | about 39 min | nearly all in the `small.en` anchor pass |
| Data prep | subset, 4.1 h of audio | about 4 h | same; memory stays flat because long hearings are processed in windows |
| Fine-tuning | subset, 1 epoch (44 steps) | about 3 min | Whisper tiny + LoRA, under 2 GB RAM |
| Fine-tuning | subset, 5 epochs (865 steps) | about 13 min | |
| Evaluation | subset test, 31 chunks | about 1 min per model | |

The LoRA adapter is about 1 MB, so each experiment costs almost no storage.

Data prep scales with the amount of audio. At the CPU rate above (about 0.75–1 h of compute per hour of audio), the full 94.6 h set would take roughly 3–4 days on this laptop. That's why the full run is planned for a GPU, where a larger and more accurate anchor model (`medium.en`) is also affordable. Fine-tuning Whisper tiny or small on the full set fits a free Colab T4. Parakeet needs a 16–24 GB GPU, which is the main reason it is deferred.

## Evaluation

- **Metrics.** Corpus WER and CER on the held-out test case, after the same normalisation of reference and hypothesis (lowercase, punctuation removed). WER is also split into substitution, deletion and insertion rates, which add up to the WER and show what kind of error changed.
- **Per chunk.** `sc-asr evaluate` saves each chunk's reference, hypothesis, scores and compression ratio. These are used to count repetition loops (`repetition_loops`: chunks with a compression ratio above 2.4), to count chunks that got better or worse, and for a bootstrap confidence interval on the WER difference.
- **Baseline.** The same command without `--model` scores the base Whisper model with no adapter, so baseline and fine-tuned numbers come from identical code.
- **What to look at.** The qualitative review focuses on names, case titles, statutes and citations, cross-talk and room acoustics.
- **No tuning on test.** The test case isn't used for model or threshold selection.

### Smoke run (3 hearings, 52 min of audio, `small.en` anchors on CPU)

- 79–84 % of transcript words matched the ASR exactly.
- 42 chunks kept, 8.5 min in total: 2.3–19.9 s each (median 12.2 s), median CER against the ASR 0.055.
- Split: 20 train, 15 validation and 7 test chunks.
- Rejects: 21 for mismatch, 16 for CER, 9 for speech at an edge and 2 for low speech ratio.
- About 39 min for all stages on CPU, nearly all of it in the ASR pass.
- Whisper tiny + LoRA, 1 epoch at batch size 1 (20 steps): train loss 2.55, validation loss 2.76. On the 7 test chunks:

  | | WER | CER |
  | --- | --- | --- |
  | zero-shot | 0.395 | 0.269 |
  | fine-tuned | 0.407 | 0.266 |

  No real change, which is expected from 20 steps on 4 minutes of audio. This run only checks that training and evaluation work end to end.

### Subset run (6 hearings, ~4.1 h of audio)

Run on a laptop CPU with `scripts/run_subset_cpu.ps1`: `small.en` anchors, then Whisper tiny + LoRA. Two training runs were made on the same data: 1 epoch first, then 5 epochs, which are now the script's defaults (5 epochs, learning rate 5e-4, 20 warmup steps, batch size 1).

**Data.** 211 chunks, 0.73 h, were kept from 4.1 h of audio (18 %):

| Split | Hearings | Chunks | Hours |
| --- | --- | --- | --- |
| train | 3 | 173 | 0.60 |
| validation | 1 | 7 | 0.03 |
| test | 2 | 31 | 0.10 |

The anchor matched 79–89 % of transcript words per hearing. The most common reject reasons were low speech ratio (57, mostly in 103822024, which has long silent stretches) and mismatch runs (42).

**First run: 1 epoch.** 44 optimizer steps: batch size 1 with the default 4-step gradient accumulation. It took about 3 minutes; validation loss was 2.76.

**Results on the test case (31 chunks, 1,106 words):**

| | WER | CER |
| --- | --- | --- |
| Whisper tiny, zero-shot | 0.348 | 0.194 |
| Whisper tiny + LoRA, 1 epoch | 0.339 | 0.191 |

The 0.009 WER improvement is within noise. A bootstrap over test chunks gives a 95 % interval of −0.008 to +0.028. Per chunk, 5 got better, 6 got worse and 20 were unchanged.

**What changed.** The fine-tuned model picked up some of the domain's phrasing:
- "emergence of new infrastructure … interpretation", where the baseline dropped words: WER 0.45 → 0.23;
- "three provisos" instead of "a pro-whizor";
- "examining the possible repercussions";
- contractions closer to the transcript's style ("they've", "it's").

Legal references are still often wrong in both models: "Section 2(21)" came out as "section 2221", and "Motor Vehicles Act" as "Montauveckel's act". A few chunks got slightly worse, for example "The Government say" became "It compensates".

**Why the effect is small**, roughly in order of impact:
1. **Learning-rate warmup.** The default warmup is 100 steps, but the run only had 44, so the learning rate never reached its target. It peaked at about 4e-5 instead of 1e-4.
2. **Very little training data.** 0.6 h, a consequence of the 18 % yield.
3. **Too little training.** One epoch, LoRA on the attention query/value projections only, and the smallest Whisper.
4. **A small test set.** 31 chunks from one case, so differences of about a point are not measurable.

**Second run: 5 epochs.** Same data and model, with the warmup and batch issues fixed: batch size 1 without accumulation, 20 warmup steps, learning rate 5e-4, 865 steps in 12 minutes on CPU. The checkpoint with the lowest validation loss (epoch 2) was kept.

| | WER | CER | WER without chunk 13 |
| --- | --- | --- | --- |
| zero-shot | 0.348 | 0.194 | 0.349 |
| 1 epoch | 0.339 | 0.191 | 0.340 |
| 5 epochs (best = epoch 2) | 0.606 | 0.373 | 0.352 |

The jump in WER comes almost entirely from one test chunk where the fine-tuned model fell into a repetition loop. It produced "my Lord, my Lord, …" until the token limit, giving a WER of 12.65 on that chunk. Insertions went from 5 % to 29 % of reference words. Without that chunk the run is level with the baseline, 0.352 against 0.349. These figures come from an evaluation without an output cap. The rerun below overwrote `results/whisper-tiny-subset-e5.json`, so they are kept here only as history.

The run also overfits. Train loss fell to about 0.2, while validation loss was lowest after epoch 2 (1.088) and drifted up to 1.156 by epoch 5. "My Lord" is the most frequent phrase in the training transcripts, so a small model trained hard on 0.6 h of audio learns to fall back on it.

**Rerun with an output-length cap.** `sc-asr evaluate` now limits generation to 8 tokens per second of audio plus 10 (at most 440), for every model, and records each hypothesis's compression ratio. The 5-epoch run was retrained with the same settings (13 min on CPU, best checkpoint again epoch 2) and scored again:

| | WER | CER | WER without chunks 13, 17, 27 |
| --- | --- | --- | --- |
| zero-shot | 0.348 | 0.194 | 0.339 |
| 5 epochs (best = epoch 2), length cap | 0.448 | 0.277 | 0.328 |

The cap shortens loops but does not prevent them. Three chunks still loop until the cap: "…the Supreme Court, the Supreme Court, …" (chunk 27, 110 words of output for a 54-word reference, WER 1.76), "That's my turn. My turn. My turn. …" (chunk 17, WER 1.48), and chunk 13 (WER 1.48). The `repetition_loops` count in the results file reports 2, not 3, because chunk 17's compression ratio (2.32) is just under the 2.4 threshold.

Leaving out those three chunks, the pooled WER of the other 28 is slightly lower for the fine-tuned model (0.328 against 0.339), and some chunks improve a lot (chunk 1: 0.45 → 0.14, chunk 24: 0.25 → 0.07). Counted chunk by chunk, though, more get worse than better: of the 31, 11 improved, 17 got worse and 3 were unchanged. The fine-tuned model is not better overall.

**Error types.** Splitting WER into substitutions, deletions and insertions (each as a share of reference words) shows where the regression comes from:

| | Substitutions | Deletions | Insertions | WER |
| --- | --- | --- | --- | --- |
| zero-shot | 0.238 | 0.057 | 0.053 | 0.348 |
| 1 epoch | 0.231 | 0.062 | 0.045 | 0.339 |
| 5 epochs (best = epoch 2) | 0.260 | 0.057 | **0.130** | 0.448 |

Insertions more than double. That is the repetition loops: the model adds words that were never said. Substitutions also rise slightly (0.238 → 0.260), so outside the loops the 5-epoch model is not recognising words better either. The 1-epoch model's small gain is spread across substitutions and insertions.

**Uncertainty.** A bootstrap over test chunks (2,000 resamples) puts the 5-epoch model's WER change against zero-shot at +0.100, with a 95 % interval of −0.021 to +0.258 (positive means worse). The interval includes zero: with 31 test chunks, this run is not significantly worse than the baseline either, but there is no sign of improvement.

Two things follow:
- **Decoding needs a guard that prevents loops, not just a cap.** Whisper's own decoding does this with temperature fallback: an output whose compression ratio is above 2.4 or whose average log-probability is below −1.0 is decoded again at a higher temperature. Blocking repeated n-grams is a poor fit here, because courtroom speech really does repeat ("my Lord, my Lord"). Baseline and fine-tuned models should use the same decoding settings.
- **More data matters more than more epochs.** With 173 training chunks and 7 validation chunks, longer training mainly memorises, and checkpoint selection is unreliable. Until there is more data, a lower learning rate (about 1e-4) and about 2 epochs are a better setting than 5e-4 for 5 epochs.

The concrete next steps are listed under [Future work](#future-work).

### Results files

Each file in `results/` holds the corpus scores (WER, CER, and substitution, deletion and insertion rates), the `repetition_loops` count, and per-chunk references, hypotheses, scores and compression ratios. All five were scored with the current `sc-asr evaluate`, which includes the output-length cap.

| File | Run | Test chunks | WER | CER | Sub. | Del. | Ins. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `whisper-tiny-smoke-zeroshot.json` | smoke, zero-shot | 7 | 0.395 | 0.269 | 0.169 | 0.099 | 0.128 |
| `whisper-tiny-smoke.json` | smoke, 1 epoch | 7 | 0.407 | 0.266 | 0.186 | 0.105 | 0.116 |
| `whisper-tiny-subset-zeroshot.json` | subset, zero-shot | 31 | 0.348 | 0.194 | 0.238 | 0.057 | 0.053 |
| `whisper-tiny-subset.json` | subset, 1 epoch | 31 | 0.339 | 0.191 | 0.231 | 0.062 | 0.045 |
| `whisper-tiny-subset-e5.json` | subset, 5 epochs (best = epoch 2), **latest** | 31 | 0.448 | 0.277 | 0.260 | 0.057 | 0.130 |

## Limitations

- **Low yield.** Only 16–18 % of the audio ends up in chunks with `small.en` anchors (smoke and subset runs). Hearings with a lot of cross-talk or long silences lose the most. A larger anchor model (`medium.en` or `large-v3`) and a looser speech-ratio check for courtroom pauses are the first things to try.
- **Labels follow the transcript.** Legal number formats like "2(21)" may not match how they were said.
- **Overlapping speech** is dropped rather than modelled.
- **Shared speakers.** Train and test share speakers (the bench and frequent counsel), so results don't show how the model would do on unseen voices.
- **Anchor bias.** Audio the anchor model handles badly is under-represented in the kept chunks, which may understate how much fine-tuning helps on exactly that audio.
- **TERES-specific cleaning.** The transcript cleaning is written for the TERES layout. Other transcript formats would need the line-number and marker rules checked.

## Future work

In priority order. The first three address what limits the current results; the rest need more compute.

1. **Stop repetition loops in decoding.** Add Whisper's temperature fallback to `sc-asr evaluate` (re-decode when the compression ratio is above 2.4 or the average log-probability is below −1.0), with the same settings for every model. Make the loop count catch near-threshold cases like chunk 17, for example by counting repeated n-grams as well.
2. **Keep more of the audio.** Only 16–18 % of the audio becomes training chunks. Use a larger anchor model (`medium.en` or `large-v3`) on a GPU, and loosen the speech-ratio check, which currently rejects normal courtroom pauses.
3. **A larger validation set.** 7 chunks can't pick a checkpoint reliably. With the full split (2 cases, 5.8 h), validation loss becomes a usable signal, and WER on validation can replace it.
4. **Gentler training until there's more data.** A learning rate of about 1e-4 and about 2 epochs instead of 5e-4 for 5, and LoRA on more layers (the attention output and feed-forward projections, not only query and value).
5. **The full 94.6 h run on a GPU.** Data prep for all 24 hearings, then Whisper tiny and small, scored on the full 3-case test set (about 11 h of raw audio).
6. **Parakeet-TDT-0.6B.** The likely best model for this audio once a 16–24 GB GPU is available. The NeMo manifests are already produced by `sc-asr data-prep`.
7. **Score legal references separately.** WER treats "Section 2(21)" written as "section 2221" as one error among many. A small entity-level check (case numbers, sections, names of parties and judges) would show whether fine-tuning helps where it matters most. Scoring could also normalise number formats so that "2(21)" and "two twenty-one" are not counted as different.
8. **A test set with unseen speakers.** Train and test currently share the bench and frequent counsel. Hearings before a different bench would show how the model handles new voices.

## Reproducing

See the [README](README.md). The latest subset run (5 epochs) is `scripts/run_subset_cpu.ps1` with its defaults, which is the same as:

```bash
sc-asr data-prep --config configs/data_prep_subset.yaml
sc-asr evaluate --manifest data/subset/manifests/hf/test.jsonl --output results/whisper-tiny-subset-zeroshot.json
sc-asr train --train data/subset/manifests/hf/train.jsonl --validation data/subset/manifests/hf/validation.jsonl \
    --output models/whisper-tiny-subset-e5 --epochs 5 --batch-size 1 --gradient-accumulation-steps 1 \
    --warmup-steps 20 --learning-rate 5e-4
sc-asr evaluate --manifest data/subset/manifests/hf/test.jsonl --model models/whisper-tiny-subset-e5 \
    --output results/whisper-tiny-subset-e5.json
```

The earlier 1-epoch run used `--epochs 1 --batch-size 1` with the other training options at their defaults. Its adapter is in `models/whisper-tiny-subset`.

## Tools used

PyAV, pypdf, num2words, faster-whisper (including its bundled Silero VAD model), RapidFuzz, Hugging Face Transformers and PEFT, and jiwer.
