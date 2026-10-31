# GapSet-R

Relative support correction for incomplete CSI gesture recognition.

This repository accompanies **GapSet-R: Relative Support Correction for Incomplete CSI Gesture Recognition**. The model combines visible anchors, span calibration, and support-weighted spatial aggregation. A relative readout of feature states H, actual support R, and reference support P preserves the reference accumulation scale. The reference branch neither reconstructs hidden CSI nor introduces independently learned parameters.

## Repository contents

| Path | Purpose |
|---|---|
| `src/gapset/models/relative_model.py` | Full GapSet-R with 73,100 parameters |
| `src/gapset/models/` | Paper baselines, ablations matched to the final architecture, and structural controls on the absolute-normalization path |
| `src/gapset/data/` | Widar amplitude preprocessing, group splits, common8 encoding, and the fixed missingness protocol |
| `src/gapset/training/` | Training, source-domain checkpoint selection, evaluation, numerical gates, and resource guards |
| `src/gapset/analysis/`, `tools/` | Group-level analysis, support diagnostics, content interventions, cue classifiers, FLOPs estimation, and result plotting |
| `checkpoints/` | Only the three full GapSet-R paper checkpoints, with seeds 42/43/44 |
| `results/` | CSV exports of final manuscript Tables 1–6 (including Tables 4a/b/c) and existing aggregate values for Figures 3/4 |
| `tests/` | Protocol tests that neither read real data nor run model forward passes |
| `environment.yml`, `requirements-cuda.txt` | Conda environment and CUDA extensions pinned to the archived versions |

The repository excludes other models' weights, raw CSI, per-window predictions, experiment-machine configurations, experiment logs, manuscript Word/PDF files, obsolete figures, and writing-process documents. Baseline implementations support comparisons under the common protocol; they do not reproduce every original paper's complete system. See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md) for the MIT license and required third-party notices.

## Environment

The archived formal experiments record **Python 3.11, PyTorch 2.5.1+cu124, NumPy 2.4.6, Mamba 2.2.6.post3, causal-conv1d 1.4.0, Triton 3.1.0, and transformers 4.44.2**. Version numbers such as 2.7.1 in the previous release notes have been corrected against the actual records. Computation uses FP32 with TF32 disabled.

Create a Linux/WSL + NVIDIA environment from the repository root:

```bash
conda env create -f environment.yml
conda activate gapset-r
python -m pip install --no-build-isolation -r requirements-cuda.txt
python -m pip check
python -m gapset --help
python tools/check_weights.py
python -m unittest discover -s tests -v
```

`environment.yml` first installs PyTorch and the project's raw-data, baseline, and plotting dependencies. `requirements-cuda.txt` then installs the CUDA extensions so that their builds can use the installed PyTorch. These extensions require compatible prebuilt wheels or a CUDA development toolchain, including `nvcc`, that matches PyTorch. The CUDA runtime bundled with a PyTorch wheel is not a complete compiler toolchain. GPU training and formal evaluation use Linux/WSL; file locking uses `fcntl`.

The YAML pins key scientific dependencies recorded in the archives; it is not a byte-for-byte lockfile for the entire operating system and all transitive dependencies. No environment was created or installed during release preparation. See the [upstream Mamba instructions](https://github.com/state-spaces/mamba/tree/v2.2.6.post3) for CUDA extension requirements and the [official previous-version instructions](https://pytorch.org/get-started/previous-versions/#v251) for the PyTorch build. The CPU reference recurrence and weight checks do not require Mamba CUDA extensions.

Some baseline authors' source snapshots lack an explicit redistribution license at the pinned revision. Those sources are excluded; only revision locks and checksum-verified retrieval are provided. See `THIRD_PARTY.md`:

```bash
python tools/fetch_baselines.py --component all
```

## Using the paper checkpoints

`checkpoints/manifest.json` records each seed, selected epoch, original checkpoint SHA-256, and release-file SHA-256. The three checkpoints were selected at epochs 176, 163, and 172, respectively, using the first strict maximum of clean source-validation macro-F1. Seeds were not selected using challenge performance. The release files preserve exactly the same model tensors and minimal identity metadata, while removing optimizer state, random state, history, and private paths. The manifest records checkpoint availability; the loader reports missing files explicitly.

```bash
python tools/check_weights.py
```

```python
from gapset.checkpoints import load_paper_model

model, metadata = load_paper_model(seed=42, directory="checkpoints")
# model(image, timestamp): image[B,8,30,256], timestamp[B,256]
# Encode image through common8 using fit-only statistics; do not pass a raw CSI heatmap.
```

Across the three seeds, clean macro-F1 is **73.19±0.15%**, and mean macro-F1 over the 45 missingness challenges is **58.96±0.49%** (sample standard deviation). These are results under the Widar six-class cross-domain development protocol, not an untouched independent final test. Results across missingness families, participants, and classes are retained. See the paper for mask/duration cues and the scope of development-set feedback during design. FLOPs are algorithmic estimates under the stated counting convention, not measured latency.

## Data preparation

Obtain the raw `.dat` files under the [Widar3.0 dataset authors' terms](https://tns.thss.tsinghua.edu.cn/widar3.0/) and place the date directories under `data/raw/CSI`. Data are not distributed with the code.

```bash
python -m gapset prepare --raw data/raw/CSI --output data/cache --workers 2
python -m gapset audit --data data/cache --raw-root data/raw/CSI --output outputs/input_identity.json
python -m gapset freeze-masks --data data/cache --output outputs/expected_masks.json
```

Inputs contain 3 antenna streams, 30 native subcarriers, and 256 time bins of 10 ms each. Apply `log1p` to visible amplitudes and standardize using training-set statistics. Masks, temporal positions, and subcarrier positions form the common8 input. All receivers and windows belonging to an original gesture group remain in the same partition. The fixed 45 challenges cover RF, Span, Step, Burst, Link, and Joint. Span retains a contiguous subcarrier interval; plot x-axes report the deleted proportion.

## Evaluating the paper checkpoints

The following entry point is for new evaluations run by the user; evaluation was not executed during repository preparation. By default, it covers all 46 conditions in each of source-domain validation and cross-domain development, without retraining or model selection:

```bash
python tools/evaluate_checkpoint.py --seed 42 --data data/cache --input-identity outputs/input_identity.json --expected-masks outputs/expected_masks.json --output outputs/paper_s42 --max-hours 12
```

Use seeds 42/43/44 separately to reproduce the three-seed results. The input manifest hashes in the checkpoint manifest identify the original experiment cache. A locally preprocessed manifest may have a different hash because raw-file paths differ. The evaluation entry point checks that the local input audit and frozen masks refer to the same cache.

The fixed-weight interventions in Table 6 use seed 42 and clean inputs:

```bash
python tools/evaluate_checkpoint.py --seed 42 --data data/cache --input-identity outputs/input_identity.json --expected-masks outputs/expected_masks.json --split cross_domain_dev --conditions clean --intervention fit_mean_content --output outputs/content_mean
python tools/evaluate_checkpoint.py --seed 42 --data data/cache --input-identity outputs/input_identity.json --expected-masks outputs/expected_masks.json --split cross_domain_dev --conditions clean --intervention mask_pattern_time_shuffle42 --output outputs/content_shuffle
python tools/cue_controls.py --data data/cache --output outputs/cue_controls
```

## Training from scratch and ablations

Training uses clean inputs for 200 epochs, with a microbatch size of 128, 4 gradient-accumulation steps, and an effective batch size of 512. The recipe uses AdamW, a learning rate of 1e-4, weight decay of 1e-3, a 200-epoch cosine schedule, cross-entropy loss, and gradient clipping at 1, without synthetic missingness augmentation, external pretraining, or early stopping. See `configs/paper_protocol.json`. Run the CPU and CUDA gates for a case before training and evaluation:

```bash
python -m gapset cpu-gate --model ours_gapset_relative_mass_control --seed 42 --data data/cache --input-identity outputs/input_identity.json --output outputs/relative_s42/cpu
python -m gapset gate --model ours_gapset_relative_mass_control --seed 42 --data data/cache --input-identity outputs/input_identity.json --cpu-gate outputs/relative_s42/cpu/cpu_gate.json --output outputs/relative_s42/gate --max-hours 2
python -m gapset train --model ours_gapset_relative_mass_control --seed 42 --data data/cache --input-identity outputs/input_identity.json --cpu-gate outputs/relative_s42/cpu/cpu_gate.json --output outputs/relative_s42/train --max-hours 24
python -m gapset eval --model ours_gapset_relative_mass_control --seed 42 --data data/cache --input-identity outputs/input_identity.json --cpu-gate outputs/relative_s42/cpu/cpu_gate.json --training outputs/relative_s42/train --expected-masks outputs/expected_masks.json --output outputs/relative_s42/eval --max-hours 12
```

| Paper model or control | `--model` |
|---|---|
| Full GapSet-R | `ours_gapset_relative_mass_control` |
| No mass correction | `ours_gapset_state_no_mass_norm_control` |
| Fixed-lag difference in the final architecture | `ours_gapset_relative_fixed_lag` |
| Absolute-normalized variant | `absolute_norm` |
| Four supplemental structural controls in Table 4c | `early_fixed_lag`, `ordered_antennas`, `ordered_normalized`, `grid_state` |
| GRU-D common8 adaptation | `grud_common8` |

The last four rows of Table 4c use the absolute-normalization path as their reference; they are not independent component ablations of the final model. Prepare GRU-D training means with:

```bash
python -m gapset fit-means --data data/cache --output third_party/fit_means.json
```

Use a new output directory for every training case. Resource guards retain a single-GPU lock, a 3 GiB memory reserve, an 82°C exit threshold, and an 80 GiB disk-space floor. The process exits if another compute job is detected; an OOM stops the run without changing the batch size. The local entry points do not include remote scheduling, automatic search, or resumption of historical jobs.

## Replotting results and running analyses

```bash
python tools/plot_paper.py --output generated_figures
python tools/complexity.py --model gapset_relative --output outputs/flops.json
```

The first command reads only the included aggregate values and exports the six-model, three-seed heatmaps in Figure 3 and the four missingness curves in Figure 4 as 600 dpi PNG and SVG files. SD is calculated across the three training seeds. Burst curves average the three positions within each seed; the heatmaps retain separate positions. DejaVu Serif is used if Times New Roman is unavailable. The raw data for the paper's real-CSI examples and the author's editable architecture-diagram PPT are not distributed in this repository.

`tools/build_group_counts.py` aggregates complete evaluation outputs by original gesture group. `tools/analyze_groups.py` performs 501 paired group-resampling iterations within fixed participants. The former takes `--manifest`, `--evaluations`, and `--output`; the latter takes `--input` and `--output`. See each entry point's `--help`. `tools/support_diagnostics.py` accepts a new plan following `configs/support_diagnostics.example.json` and supports only the originally prespecified 256-window support diagnostic for seed 42. Supply your own data paths, weight SHA, and new execution deadline.

## Release verification

Release checks covered source/license inventories, AST equivalence of core computations, manuscript tables against existing values, protocol tests, CLI entry points, result replotting, and checkpoint hashes against formal evaluation records. Complete verification records are kept outside the upload directory. No new 200-epoch training, real-data evaluation, or hardware-efficiency measurements were performed after packaging.
