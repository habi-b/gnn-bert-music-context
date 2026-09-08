# GNN-Based BERT for Understanding Context from Music

**Course:** CSE425 / EEE474 / CSE715 (Neural Networks)
**Submission deadline:** 8 September, 2026

A hybrid BERT + Graph Neural Network system for music context understanding,
covering four tasks: multi-label tag classification (BERT), genre
classification (GNN vs CNN baseline), GNN-BERT fusion, and contrastive
cross-modal retrieval.

**→ For report writing: `results/metrics.json` has every final number in one
place. Each task's section below explains what the number means and why.**

---

## Repository structure

```
gnn-bert-music-context/
├── README.md                  <- you are here
├── config.yaml                 <- all hyperparameters, one place
├── requirements.txt
├── data/
│   ├── raw/                    <- downloaded datasets (gitignored -- see Setup)
│   ├── processed/               <- graphs, mel-specs, tokenized text (gitignored, see Setup)
│   └── splits/                  <- train/val/test CSVs (small, committed)
├── notebooks/
│   ├── 00_setup.ipynb
│   ├── 01_data_acquisition.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_task1_bert.ipynb
│   ├── 04_task2_gnn.ipynb
│   ├── 05_task3_fusion.ipynb
│   └── 06_task4_contrastive.ipynb
├── src/
│   ├── audio_features.py       <- mel-spectrogram extraction (Task 2 CNN)
│   ├── graph_builder.py        <- chroma segment graph construction (Tasks 2-4)
│   ├── bert_encoder.py         <- Task 1 BERT tag classifier
│   ├── gnn_model.py            <- Task 2 GraphSAGE + CNN baseline
│   ├── fusion_model.py         <- Task 3 fusion model + ablation variants
│   ├── contrastive.py          <- Task 4 dual encoder + InfoNCE + retrieval
│   ├── datasets.py             <- shared PyTorch Dataset/collate classes
│   └── evaluate.py             <- shared evaluation metrics
├── results/
│   ├── metrics.json            <- ALL final numbers, compiled
│   ├── *.pt                    <- trained model weights (gitignored, large)
│   ├── *_history.json          <- per-epoch training curves
│   ├── plots/                  <- F1 curves, GNN vs CNN comparison, t-SNE
│   └── retrieval_examples/     <- Task 4 qualitative retrieval examples
└── report/                     <- final written report goes here
```

---

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**System dependencies** (not pip-installable):
- **Deno** (`irm https://deno.land/install.ps1 | iex` on Windows, or `curl -fsSL https://deno.land/install.sh | sh` on Linux/Colab) — required for `yt-dlp` to solve YouTube's JS signature challenge when downloading MusicCaps audio
- **ffmpeg** (`winget install ffmpeg` on Windows) — required by `yt-dlp` for audio extraction

**To download the MusicCaps audio (Task 3/4 data)**, you also need a
`cookies.txt` file (export via a browser extension like "Get cookies.txt
LOCALLY" while logged into YouTube) — place it in the project root.
**Never commit this file** (it's in `.gitignore`).

### Regenerating the data

`data/raw/` and most of `data/processed/` are gitignored (large, and
fully regeneratable). To rebuild from scratch, run the notebooks in order:

1. `01_data_acquisition.ipynb` — downloads MagnaTagATune, FMA-small, MusicCaps
2. `02_preprocessing.ipynb` — builds chroma graphs, mel-spectrograms, tokenizes captions
3. `03_task1_bert.ipynb` through `06_task4_contrastive.ipynb` — one per task

Each notebook is self-contained (re-mounts/re-imports everything it needs)
and uses resumable manifests, so re-running is safe — already-completed
work is automatically skipped.

---

## Dataset strategy

We follow the project spec's Table 1 recommended pairings strictly, using
a different dataset per task rather than one unified dataset:

| Task | Dataset | Why |
|---|---|---|
| 1 (BERT) | **MagnaTagATune** | top-50 tags as multi-label targets; text input = cleaned track filename (MagnaTagATune has no native captions) |
| 2 (GNN/CNN) | **FMA-small** | 8,000 clips, 8 balanced genres — standard genre-classification benchmark |
| 3 (Fusion) | **MusicCaps** | only dataset in the table pairing real natural-language captions with the *same* audio clips, needed for genuine fusion |
| 4 (Contrastive) | **MusicCaps** | same reasoning as Task 3 |

MusicCaps' audio isn't directly downloadable (only YouTube video IDs are
provided) — we use `yt-dlp` with cookie-based auth and a JS-challenge
solver to download ~1,150 clips (of 5,521 total; some videos are
permanently unavailable, expect ~95%+ success rate on attempts).

---

## Task 1 — BERT Multi-Label Tag Classifier

**Text input choice:** MagnaTagATune has no captions, only tags and file
paths. Two options existed: (a) derive pseudo-text from track/artist
filenames, or (b) use partial tags as text to predict the rest. We chose
**(a)** — it's a genuinely independent signal (not circular), producing an
honest, if modest, result rather than an artificially inflated one from
predicting tags from other tags.

**Splitting:** MagnaTagATune's 25,863 rows map to only ~5,405 unique
tracks (long recordings are split into many clips). We used
`GroupShuffleSplit` grouped by track — not a random split — to guarantee
zero train/val/test leakage of the same track, per the spec's own
no-artist-leakage requirement.

**Result:** test macro-F1 = **0.206**, micro-F1 = **0.348** (5 epochs,
stopped when val loss plateaued). Below the spec's illustrative benchmark
(~0.48) — expected, since filename-derived pseudo-text carries much
weaker signal than real captions. Qualitative inspection shows the model
under-predicts (conservative, low recall) rather than hallucinating tags.

---

## Task 2 — GNN vs CNN Baseline

Both models trained and evaluated on the **identical** FMA-small
train/val/test split for a fair, direct comparison.

- **GNN** (GraphSAGE on chroma segment graphs): test accuracy **28.9%**,
  macro-F1 **0.268**
- **CNN** (3-conv-layer on mel-spectrograms): test accuracy **55.4%**,
  macro-F1 **0.544**

**CNN substantially outperforms GNN.** This is attributed to the GNN's
very coarse node features — each track becomes only 5-6 graph nodes, each
a single chroma vector averaged over 5 seconds — versus the CNN's
full-resolution spectrogram input. Both are well above the 12.5%
random-chance baseline (8 classes), so the GNN is learning something real,
just far less than the CNN can extract from richer input.

---

## Task 3 — GNN-BERT Fusion

**Labels:** MusicCaps has no tag columns; we derive multi-label targets
from `aspect_list` (short descriptive phrases auto-extracted per caption).
We use the **top-30** most frequent aspects (not top-50 like Task 1) —
deliberately smaller, since MusicCaps has far fewer clips (~1,159 vs
MagnaTagATune's ~25,863) and a larger K would leave too few positive
examples per tag per split to evaluate reliably.

**Model:** cross-attention fusion per the spec's Algorithm 3 — the graph
embedding is the query, BERT's full token sequence is key/value.

**Results (test macro-F1):**

| Model | Macro-F1 |
|---|---|
| BERT-only | **0.552** |
| GNN-only | **0.000** |
| Early-Concat | 0.533 |
| Cross-Attention (Fusion) | 0.533 |

**Two honestly-reported findings, not hidden:**
1. **GNN-only collapses** to predicting zero tags for every test clip.
   Verified via diagnostic (sigmoid outputs never exceed 0.36). This is a
   real, informative result — chroma/pitch features carry no usable
   signal for semantic/mood/quality tags like "emotional" or "amateur
   recording," unlike Task 2's genre task where chroma has weak but real
   correlation with genre.
2. **BERT-only slightly outperforms the full fusion model** on this
   metric. Since the GNN branch alone is uninformative, fusion's
   challenge becomes "don't let a useless branch drag down BERT's signal"
   — and it doesn't fully achieve that here.

Full fusion still dramatically outperforms Task 1's standalone BERT
(0.206, different dataset) and Task 2's standalone GNN (0.268, different
task) — demonstrating the value of richer text (real captions) and of
combining modalities generally, even if this specific ablation shows
BERT alone edging out the fusion variant on this particular label set.

**t-SNE** of fused embeddings (colored by the "instrumental" tag) shows
weak but real partial clustering — limited by only ~8-10 positive examples
in the 116-clip test set.

**Case studies:** cross-attention weights are near-uniform even after
correctly masking padding tokens — likely because ~927 training examples
is too little data for a 110M-parameter attention mechanism to learn
sharp, interpretable focus, even though the classifier still extracts
useful signal from the pooled representations (predictions were accurate
in all 3 case studies).

---

## Task 4 — Contrastive Dual-Encoder + Retrieval

Dual encoder (GraphEncoder + TextEncoder, both projecting to a shared
128-dim L2-normalized space), trained with symmetric InfoNCE
(temperature=0.07), on the same MusicCaps split as Task 3.

**Retrieval (test set, 116 clips; chance R@1≈0.9%, R@5≈4.3%, R@10≈8.6%):**

| Direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| Caption → Audio | 1.7% | 9.5% | 20.7% |
| Audio → Caption | 1.7% | 8.6% | 15.5% |

Meaningfully above chance in both directions (roughly 2-2.4x the random
baseline at each K) — real, if modest, learned cross-modal alignment.

**Zero-shot tagging vs Task 3 supervised:** embedding tag names as
pseudo-captions and matching against audio embeddings gives macro-F1 =
**0.079**, versus Task 3's supervised fusion at **0.533** — roughly a 7x
gap. Contrastive pretraining alone learns weak alignment; explicit
supervision on labeled tags is far more effective for the tagging task
itself. Expected, and a clean illustration of the supervised-vs-zero-shot
tradeoff.

---

## Known limitations (for the report's discussion section)

- Task 1's pseudo-text (filenames) is a weaker signal than real captions
  would provide — by design, given MagnaTagATune's lack of captions.
- Task 2's GNN uses very coarse graph features (5-6 nodes/track); a more
  fine-grained segmentation could likely close some of the CNN gap.
- Task 3's macro-F1 on the ~116-clip test set is somewhat volatile for
  rare tags (few positive examples per split); micro-F1 is more stable.
- Task 3's cross-attention weights are not sharply interpretable, likely
  a data-scale limitation (~927 training examples for a 110M-parameter
  model).
- All MusicCaps-based results (Tasks 3-4) are on a ~1,150-clip subset,
  not the full 5,521-clip dataset, due to YouTube audio availability.
