# Model Card: 16-Channel EEG Encoder Bundle

## Contents

- `model.safetensors`: FP32 ChannelNet backbone trained in the 16-channel adaptation experiment.
- `adapter.pth`: FP32 16→64→128 channel adapter.
- `model_fp16.safetensors`: FP16 storage-compressed backbone.
- `adapter_fp16.pth`: FP16 storage-compressed adapter.
- `config.json`: ChannelNet architecture configuration.

The bundle contains model parameters only. It does not contain raw EEG,
images, captions, subject identifiers, optimizer state, or LLM weights.

## Intended use

Research and portfolio demonstration of low-density EEG representation
learning, offline inference, model export, and edge-readiness evaluation.

## Limitations

- The model was evaluated in a controlled visual-stimulus dataset setting.
- It does not decode arbitrary private thoughts.
- It is not a medical device and must not be used for diagnosis or treatment.
- The FP16 checkpoint is a storage optimization; latency depends on the target runtime and hardware.
- No claim of completed Android or iOS deployment is made.

## Training data

The experiment followed the public-data workflow documented by upstream
Thought2Text. Training data is intentionally not redistributed here. Consult
the upstream project for dataset provenance and access instructions.
