# Edge-Readiness Benchmark Record

## Validated environment

- Date: 2026-07-20
- CPU: AMD Ryzen 9 9900X 12-Core Processor
- OS: Linux x86_64
- Python: 3.12.7
- PyTorch: 2.3.0+cu121
- Inference threads: 1
- Warm-up iterations: 5
- Measured iterations: 30
- Input: synthetic zeros, `[1, 1, 16, 440]`

## Results

| Metric | Value |
|---|---:|
| Mean | 203.683 ms |
| Median | 203.545 ms |
| P95 | 206.684 ms |
| Adapter parameters | 9,216 |
| Total EEG encoder parameters | 5,236,186 |

The benchmark uses no EEG data. It measures the FP32 adapter and ChannelNet
front end only; multimodal projection and LLM decoding are out of scope.

## Export validation

TorchScript tracing completed successfully and produced a 21,154,273-byte
fixed-shape graph. The generated artifact is excluded from Git because it can
be reproduced from the released weights.

The local Python 3.12 + PyTorch 2.3 combination could not run `torch.export`
because that PyTorch release does not support Dynamo on Python 3.12. The
repository retains the `torch.export` path for newer compatible environments,
but it is not claimed as locally validated here.

## Interpretation

This is a desktop CPU proxy for deployment planning, not a benchmark from an
Android or iOS device. A true mobile evaluation should lower the exported
graph to the selected runtime, run on target hardware, and report warm/cold
latency, peak resident memory, energy use, and numerical agreement.
