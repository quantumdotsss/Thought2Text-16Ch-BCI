# Experimental Results

The following values were recovered from the local experiment logs and
aggregated prediction files. No raw EEG, image data, captions, or per-sample
predictions are included in this repository.

| Pipeline | Input channels | Best encoder classification accuracy | Object accuracy after EEG-to-text generation |
|---|---:|---:|---:|
| Upstream-style baseline | 128 | 52.19% | 44.49% (884 / 1,987) |
| Low-density adaptation | 16 | 38.94% | 30.10% (598 / 1,987) |

## Interpretation

The 16-channel experiment reduces sensor dimensionality by 8x and retains a
measurable semantic signal, but it does not match the 128-channel baseline.
This is an early feasibility result rather than a state-of-the-art claim.

The runs also used different encoder training schedules (including different
epoch counts, batch sizes, and learning rates), so the table is not a strict
controlled ablation. A stronger follow-up would repeat both conditions under
the same optimization schedule and report confidence intervals across seeds
and subjects.

## Deployment benchmark policy

`scripts/benchmark_edge.py` uses a synthetic all-zero tensor. It measures only
the inference graph and never reads EEG data. Desktop CPU results are reported
as an edge-readiness proxy and must not be presented as measurements from an
Android or iOS device.
