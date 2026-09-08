# Manuscript-to-code consistency audit

Audit date: 2026-09-08

## Scope

The supplied manuscript archive and three-file code snapshot were checked against each other for architecture, text guidance, contrastive learning, preprocessing, training/validation, inference, and evaluation. No dataset, checkpoint, raw prediction, or experiment log was supplied, so numerical results could not be independently reproduced.

## Findings

| Area | Supplied code vs. manuscript | Severity | Release action |
|---|---|---:|---|
| VGG16 encoder | The slices `[:5]`, `[5:10]`, and `[10:17]` produce 64/128/256-channel maps at 112/56/28 pixels for 224-pixel input, matching the manuscript. | Match | Preserved. |
| Decoder output | The manuscript specifies a 1x1 output convolution. The supplied code uses a 3x3 transposed convolution. It also applies the same `upconv1_1` layer twice, sharing weights unintentionally. | High | Replaced with distinct convolutional blocks and a 1x1 output head. Retraining is required. |
| Sigmoid | The manuscript describes sigmoid after the output layer. The supplied model returns logits and applies sigmoid in loss/metrics, which is the numerically correct BCE-with-logits implementation. | Benign wording difference | Kept logits internally; documented thresholding. |
| CLIP text pipeline | CLIP 512D -> Linear(384) -> LayerNorm -> GELU -> Linear(256) and learned 28x28 outer-product expansion match the manuscript. | Match | Preserved, with configurable online/local model loading. |
| CLIP path | `/mnt/clip-vit-base-patch32` is hard-coded and `local_files_only=True`, preventing use on a new machine. | High | Replaced by a configurable model ID or local path. |
| CLIP optimization | The supplied optimizer includes the complete CLIP model, so it is fine-tuned, but the manuscript does not state whether CLIP is frozen. | Medium | Exposed `freeze_clip`; default `false` preserves the supplied behavior. The manuscript should state this choice. |
| Horizontal flip text | The sequential placeholder replacement can turn `left` into `LEFT_TEMP` and similarly corrupt Chinese laterality. | High | Replaced with atomic, case-aware swapping. |
| Cross-position attention | The manuscript defines `A @ H_V`; the supplied code applies `A^T @ H_V`. | High | Release follows the manuscript equation (`A @ H_V`). Retraining is required. |
| Contrastive pair construction | Foreground image anchors, corresponding mask-encoder positives, within-image background negatives, cosine similarity, mean negative similarity, temperature 0.07, and three-scale averaging are represented in the supplied loss. | Partial match | Preserved and made deterministic under seeded execution. |
| Projection heads | Three GAP/MLP heads are instantiated and evaluated, but their outputs are discarded. Their parameters are also absent from the optimizer. Therefore the projection-head subsection and Figure 2 are not implemented by the supplied training path. | Critical | Release applies the shared per-level GAP/MLP to every sampled 1x1 feature patch and includes its parameters in the optimizer. Retraining is required. |
| Boundary awareness | Sampling uses all foreground and all background pixels; it does not explicitly identify a contour or prioritize pixels near the boundary. The name describes the intended effect, not an explicit boundary-band algorithm. | Medium scientific limitation | Kept consistent with the manuscript equations. Avoid claiming explicit contour sampling unless the method is changed and re-evaluated. |
| Small/empty masks | The supplied loss returns a Python float and silently skips a sample whenever fewer than five foreground or background points exist. The manuscript only defines `K=min(...)` and does not state this rule. | Medium | Release accepts any non-empty pair set and returns a differentiable zero only when a class is absent. |
| Random sampling device | `torch.randperm` is created on CPU while features may be on CUDA, which is version/device dependent. | Medium | Sampling now occurs on the feature device. |
| Mask preprocessing | Dividing masks by 255 fails for masks already encoded as 0/1. | High | Any nonzero mask pixel is treated as foreground. |
| Augmentation | Resize 224, normalization 0.5/0.5, rotation up to 20 degrees, and horizontal flip probability 0.5 match the manuscript and supplied code. | Match | Preserved. |
| MosMed training | Adam, learning rate 0.001, batch size 8, 120 epochs, temperature 0.07, scheduler factor 0.5/patience 15, and checkpoint thresholds match the stated MosMed setup. | Mostly match | Core values are in `configs/mosmed.yaml`; best validation Dice is saved without threshold gating. |
| Contrastive schedule | The supplied code holds 0.05 for 20 epochs and then ramps to 0.5 over the next 80 epochs. The manuscript says a 20-epoch warm-up "from 0.05 to 0.5." | High | Release linearly ramps 0.05 -> 0.5 during the first 20 epochs. Use the legacy schedule only if the manuscript is revised and experiments are rerun accordingly. |
| QaTa training | No QaTa loader, split, training path, or exact contrastive weight was included in the code snapshot. | Critical completeness gap | Added a generic QaTa configuration. Its fixed 0.5 weight and metadata filenames are marked as unverified defaults that must be checked against the actual experiment. |
| Validation | The supplied code recomputes randomized contrastive loss on validation masks although selection is based on validation Dice. | Medium | Validation now uses segmentation loss and per-case metrics only; ground-truth masks never enter the model feature path at inference. |
| Metrics | Dice and IoU are flattened over the entire dataset, not averaged per case. This cannot produce the manuscript's per-case confidence intervals or paired tests. Empty-empty IoU is reported as zero. | Critical | Evaluation now writes per-case Dice/IoU, means, and normal-approximation 95% CIs; empty-empty cases score 1. |
| Statistical tests | The supplied code contains no paired t-test, KS analysis, or CI implementation despite manuscript claims. | Critical completeness gap | Added `scripts/compare.py` and CI output. The exact manuscript statistics still require original per-case predictions. |
| Checkpoints | A shallow `state_dict().copy()` is retained in memory; saved files omit optimizer, scheduler, loss-head state, config, and epoch. The printed final filename does not match the saved filename. | High | Release stores complete resumable `best.pt` and `last.pt` checkpoints. |
| Entrypoint | The training function returns nothing but the main block unpacks three values, causing a `TypeError` after training. Claims of checkpoint recovery and saved plots are not implemented. | High | Replaced by a clean CLI with JSONL history and resume support. |
| Inference/evaluation | Neither exists in the supplied package. | Critical completeness gap | Added single-image inference, split evaluation, and paired comparison scripts. |
| Reproducibility controls | No random seeds or deterministic option are provided. | Medium | Added seed control and an optional deterministic mode. |

## Interpretation

The supplied snapshot captures the paper's broad idea, text-projection block, encoder scales, and core positive/negative similarity formula. It is not a complete or exact implementation of the manuscript as written. The most consequential issue is that the claimed contrastive projection heads do not affect training at all. The attention transpose and decoder head also differ from the equations/text.

This release repairs those discrepancies, so it is a paper-aligned reference implementation rather than a byte-for-byte reproduction of the code that produced the reported table. Existing weights from the supplied snapshot are not architecture-compatible. Reported Dice/IoU values must not be presented as reproduced by this release until training and evaluation are rerun on the stated fixed splits.

## Manuscript items to confirm before submission

1. State explicitly whether the CLIP text encoder is fine-tuned or frozen.
2. Confirm the exact QaTa contrastive-loss weight schedule and scheduler patience from the original experiment records.
3. Decide whether "boundary-aware" means foreground/background discrimination (current equations) or explicit contour-band sampling. Do not imply the latter without implementing and evaluating it.
4. Recalculate the performance table, confidence intervals, paired tests, and ablations from preserved per-case predictions and seeds.
5. Resolve the internal tension between global average pooling and pointwise spatial sampling. This release interprets each selected location as a 1x1 patch, for which GAP is the identity before the MLP.
