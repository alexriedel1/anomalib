# anomalib — qbox fork (`anomalib-qbox2`)

This is a fork of [anomalib](https://github.com/open-edge-platform/anomalib), the anomaly detection
library by Intel / Open Edge Platform. For documentation, installation and the list of models, see
the [upstream repository](https://github.com/open-edge-platform/anomalib) and
[its docs](https://anomalib.readthedocs.io/en/latest/). The code is licensed under the upstream
[Apache-2.0 license](LICENSE).

The `anomalib-qbox2` branch tracks upstream `main` and adds the changes qbox needs. This README
lists them. **Every change below has to survive merges from upstream `main`.**

## How qbox uses this branch

- qboxv2 installs anomalib from git, pinned to a **commit** of this branch in `qboxv2/uv.lock`. A
  change here does not reach qbox until you push it and bump that pin.
- Each qbox model spec (`qbox/training/specs.py`) builds its anomalib model. qbox passes
  `precision="float16"` and expects it to run in **bfloat16** (see below).
- qbox trains on folders of normal images only. Validation and test sets come from synthetic
  anomalies (`TestSplitMode.SYNTHETIC`), with the blend factor set by the expected-severity option.

Upstream baseline at the last merge: `7d2a8ead` (`chore(lib): bump to dev (#3791)`). To list every
change against upstream:

```bash
git fetch https://github.com/open-edge-platform/anomalib.git main
git diff FETCH_HEAD anomalib-qbox2 -- src tests
```

## Changes not in upstream `main`

### 1. Configurable synthetic anomalies

Upstream hardcodes the anomaly texture source (`./datasets/dtd`) and the blend factor
(`(0.01, 0.2)`) inside `make_synthetic_dataset`. qbox sets both per training.

| File                                            | Change                                                                                                                                                                                                                     |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/anomalib/data/datamodules/base/image.py`   | `AnomalibDataModule` accepts `anomaly_source_path` and `anomaly_blend_factor` and passes them to `SyntheticAnomalyDataset.from_dataset` for synthetic val and test splits.                                                 |
| `src/anomalib/data/datamodules/image/folder.py` | `Folder` exposes the same two arguments (default blend `(0.03, 0.07)`).                                                                                                                                                    |
| `src/anomalib/data/utils/synthetic.py`          | `SyntheticAnomalyDataset` builds the `PerlinAnomalyGenerator` from those arguments and hands it to `make_synthetic_dataset`. Generation uses `tqdm` `progress_apply`, which qbox hooks to drive its training progress bar. |
| `src/anomalib/data/datasets/image/folder.py`    | `make_folder_dataset` always sets the task to `segmentation`, so pixel metrics run on folders that have no masks but get synthetic ones.                                                                                   |

### 2. `precision="float16"` runs in bfloat16

float16 overflows on these backbones: un-normalized DINOv3 features reach about 1e4, beyond float16's
maximum of 65504. That puts NaNs in the memory bank and makes every threshold NaN, so F1 is 0 while
AUROC still looks normal. bfloat16 has float32's range. Upstream uses `.half()`; this fork maps
`PrecisionType.FLOAT16` to bfloat16.

| File                                                        | Change                                                                                                                                                      |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/anomalib/models/image/patchcore/lightning_model.py`    | `FLOAT16` → `.bfloat16()`. Also contains an unused `configure_precision` helper.                                                                            |
| `src/anomalib/models/image/anomaly_dino/lightning_model.py` | `FLOAT16` → `.bfloat16()`.                                                                                                                                  |
| `src/anomalib/models/image/super_add/lightning_model.py`    | `FLOAT16` → `.bfloat16()`.                                                                                                                                  |
| `src/anomalib/models/image/efficient_ad/lightning_model.py` | New `precision` argument (`configure_precision`, bfloat16 for `FLOAT16`), quantiles computed in float32, default input size `768x768` instead of `256x256`. |
| `src/anomalib/models/image/efficient_ad/torch_model.py`     | Inputs cast to the network dtype; ImageNet normalization tensors follow the input dtype.                                                                    |
| `src/anomalib/models/components/filters/blur.py`            | `GaussianBlur2d` convolves in float32 so it works with half-precision anomaly maps.                                                                         |
| `src/anomalib/models/image/patchcore/anomaly_map.py`        | Anomaly-map smoothing uses `kornia.filters.gaussian_blur2d` instead of `GaussianBlur2d`.                                                                    |

### 3. SuperADD: RADIO backbone, capped threshold, exportable models

| File                                                               | Change                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/anomalib/models/image/super_add/components/radio.py` (new)    | `RadioBackbone` loads NVIDIA C-RADIOv4 (`c-radio_v4-h`, `c-radio_v4-so400m`) through `torch.hub` and presents the same interface as `DinoV3Backbone`. `make_bound_methods_picklable` rebinds the methods RADIO attaches to the model instance, so a torch-exported model can be loaded again by `TorchInferencer`. Without it, loading fails with `'VisionTransformer' object has no attribute '_forward_cpe'`. |
| `src/anomalib/models/image/super_add/components/__init__.py` (new) | Exports `RadioBackbone` and `is_radio_backbone`.                                                                                                                                                                                                                                                                                                                                                                |
| `src/anomalib/models/image/super_add/torch_model.py`               | `SuperADDModel` selects `RadioBackbone` for RADIO names, `DinoV3Backbone` otherwise.                                                                                                                                                                                                                                                                                                                            |
| `src/anomalib/models/image/super_add/lightning_model.py`           | For RADIO, the default pre-processor skips ImageNet normalization (RADIO expects `[0, 1]`) and the default post-processor uses the `capped` threshold. `configure_pre_processor` has a `normalize` argument.                                                                                                                                                                                                    |
| `src/anomalib/models/image/super_add/post_processor.py`            | `pixel_threshold_method="capped"`: `min(p95 * 1.421, p50 + k * (max - p50))` with `pixel_threshold_cap_k=0.98`. The original `p95 * 1.421` rule sets RADIO's threshold above every normal pixel. The default stays `scaled_percentile`.                                                                                                                                                                         |
| `tests/unit/models/image/super_add/test_super_add.py`              | Tests for patch stitching, percentile and capped thresholds, RADIO geometry and pre-processing, and pickling of bound methods.                                                                                                                                                                                                                                                                                  |

Known limitation: loading an exported RADIO model still requires RADIO's hub code to be imported
(`torch.hub.load("NVlabs/RADIO", ...)`), because its classes live in the hub cache. qbox's startup
model check does that; a bare Python process gets `No module named 'radio'`.

## Keeping the changes when merging upstream

1. `git fetch https://github.com/open-edge-platform/anomalib.git main && git merge FETCH_HEAD`.
2. Check conflicts in the files listed above in particular. Upstream edits the same precision
   blocks (`.half()`) and SuperADD files, and a clean merge can still silently revert
   `bfloat16()` to `.half()`. That is how the SuperADD NaN regression reached qbox.
3. Run `uv run --extra cu130 --extra test python -m pytest tests/unit/models/image/super_add`.
4. Push, then bump the anomalib commit in `qboxv2/uv.lock` and retrain the affected models. Old
   exported `.pt` files are not rewritten.
