# ComfyUI RH RVC

- [RunningHub China](https://www.runninghub.cn/?inviteCode=rh-v1367)
- [RunningHub International](https://www.runninghub.ai/?inviteCode=rh-v1367)

![License](https://img.shields.io/badge/License-Apache%202.0-green)

ComfyUI custom nodes for Retrieval-based Voice Conversion (RVC). This plugin wraps inference and training code from [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) and exposes model loading, ZIP model upload, voice conversion, and one-click training nodes in ComfyUI.

## Features

- Load existing RVC `.pth` models from `ComfyUI/models/RVC`.
- Upload an RVC model ZIP from the ComfyUI node UI and extract `.pth` plus optional `.index` files into `ComfyUI/models/RVC/_uploaded`.
- Convert ComfyUI `AUDIO` inputs and return ComfyUI `AUDIO` outputs for downstream save, mix, or video nodes.
- Run RVC preprocessing, F0/feature extraction, model training, and index training from a local dataset folder.
- Save training outputs as `<save_name>.pth`, `<save_name>.index`, and `<save_name>.zip` under `ComfyUI/output/RVC/<save_name>/`.
- Use HuBERT from `ComfyUI/models/RVC/_assets/hubert`.
- Support optional RMVPE pitch extraction with `ComfyUI/models/RVC/_assets/rmvpe/rmvpe.pt`.
- Keep model binaries outside the plugin repository.

## Installation

Clone this repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone <repository-url> ComfyUI_RH_RVC
cd ComfyUI_RH_RVC
pip install -r requirements.txt
```

Restart ComfyUI after installation.

The plugin includes the RVC inference source under:

```text
ComfyUI_RH_RVC/rvc_source
```

Set `RVC_PROJECT_ROOT=/absolute/path/to/rvc_source` only if you intentionally want to use a different RVC source tree.

## Model Download and Installation

The plugin repository does not include model binaries. Put all RVC assets under `ComfyUI/models/RVC`.

### Model Directory Structure

```text
ComfyUI/
└── models/
    └── RVC/
        ├── _assets/
        │   ├── hubert/
        │   │   └── hubert_base.pt
        │   └── rmvpe/
        │       └── rmvpe.pt
        ├── your_voice_model/
        │   ├── your_voice_model.pth
        │   └── your_voice_model.index
        └── _uploaded/
            └── uploaded_zip_models_are_extracted_here/
```

Required files:

| File | Required | Destination | Description |
| --- | --- | --- | --- |
| RVC `.pth` voice model | Yes | `ComfyUI/models/RVC/<model-folder>/` | Main voice conversion checkpoint. |
| HuBERT | Yes | `ComfyUI/models/RVC/_assets/hubert/hubert_base.pt` | Content feature model. |
| RVC `.index` | Optional | Same folder as `.pth` or any subfolder under `models/RVC` | Retrieval index for better timbre matching. |
| RMVPE | Optional | `ComfyUI/models/RVC/_assets/rmvpe/rmvpe.pt` | Required only when `f0_method=rmvpe`. |
| Pretrained G/D | Optional | `ComfyUI/models/RVC/_assets/pretrained` or `_assets/pretrained_v2` | Training auto-detects these paths when the pretrained inputs are empty; missing files fall back to training from scratch. |

### Download Methods

#### Method 1: Download core assets with aria2

Run these commands from `ComfyUI/models/RVC`:

```bash
mkdir -p _assets/hubert _assets/rmvpe
aria2c -x 16 -s 16 -k 1M -c -o _assets/hubert/hubert_base.pt \
  https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt
aria2c -x 16 -s 16 -k 1M -c -o _assets/rmvpe/rmvpe.pt \
  https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
```

#### Method 2: Download core assets with hf

```bash
cd ComfyUI/models/RVC
mkdir -p _assets/hubert _assets/rmvpe
hf download lj1995/VoiceConversionWebUI hubert_base.pt --local-dir _assets/hubert
hf download lj1995/VoiceConversionWebUI rmvpe.pt --local-dir _assets/rmvpe
```

#### Method 3: Manual download

| Model | Link | Destination |
| --- | --- | --- |
| HuBERT | https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt | `ComfyUI/models/RVC/_assets/hubert/hubert_base.pt` |
| RMVPE | https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt | `ComfyUI/models/RVC/_assets/rmvpe/rmvpe.pt` |

Bring your own RVC `.pth` and optional `.index` voice model files. The `.pth` and `.index` file names do not have to be identical, but using the same stem is recommended because it makes automatic matching more reliable.

### Model Selection Guide

| Use case | Recommended setup | Notes |
| --- | --- | --- |
| Basic voice conversion | `.pth` + HuBERT | Works without an index, but quality may be lower. |
| Better timbre matching | `.pth` + matching `.index` + HuBERT | Put the `.index` near the `.pth` or use the ZIP loader. |
| RMVPE pitch extraction | `.pth` + HuBERT + RMVPE | Required only when `f0_method=rmvpe`. |

### F0 Method Notes

| Method | Extra requirement |
| --- | --- |
| `harvest` | Uses `pyworld`; installed from `requirements.txt`. |
| `dio` | Uses `pyworld`; installed from `requirements.txt`. |
| `pm` | Uses `praat-parselmouth`; installed from `requirements.txt`. |
| `crepe` | Uses `torchcrepe`; installed from `requirements.txt`. |
| `rmvpe` | Requires `models/RVC/_assets/rmvpe/rmvpe.pt`. |

## Usage

### Example Workflow

Import this workflow into ComfyUI:

- [`examples/rvc_voice_conversion_basic_api.json`](examples/rvc_voice_conversion_basic_api.json)
- [`examples/rvc_training_basic_api.json`](examples/rvc_training_basic_api.json)

The workflow demonstrates:

1. Load or upload an RVC model ZIP with `RunningHub RVC ZIP Model Loader`.
2. Load audio, trim it to a default end time of 20 seconds, and separate stems with `AudioSeparation`.
3. Convert the vocal stem with RVC.
4. Mix the converted vocal back with the other stems.
5. Save the final audio with a standard audio save node.

The training example uses `RunningHub RVC One-Click Train`. Training data can be provided in two ways:

1. `trainset_dir`: dataset audio folder, empty by default; if filled, it takes priority.
2. `audio`: optional audio input, used when `trainset_dir` is empty. It accepts ComfyUI `AUDIO`, lists of `AUDIO`, audio file paths, or path lists from upstream nodes.

The training node enables `audio_auto_clean` by default. It converts training audio to mono, trims silence, skips clips that are too short or too quiet, and peak-normalizes the kept clips before RVC preprocessing. If the source has backing music, enable `auto_extract_vocals` to run Demucs/HDEMUCS vocal extraction first; keep it disabled for dry vocal datasets to avoid extra separation artifacts. Multi-speaker, heavy-reverb, or noisy sources should still be cleaned upstream so training sees clean single-speaker vocals.

`trainset_dir` examples:

- Absolute path: `/workspace/ComfyUI/input/my_rvc_trainset`
- Relative to ComfyUI input: `my_rvc_trainset`

The `save_name` input controls the final output file names. A completed run writes:

```text
ComfyUI/output/RVC/<save_name>/<save_name>.pth
ComfyUI/output/RVC/<save_name>/<save_name>.index
ComfyUI/output/RVC/<save_name>/<save_name>.zip
```

## Node Reference

### RunningHub RVC Model Loader

Loads an existing `.pth` model from `ComfyUI/models/RVC` and returns an `RVC_MODEL` handle.

### RunningHub RVC ZIP Model Loader

Uploads or selects a `.zip` file from ComfyUI input and extracts `.pth` plus optional `.index` files into `ComfyUI/models/RVC/_uploaded`.

Upload limit: 150 MB per ZIP.

### RunningHub RVC Voice Conversion

Converts a ComfyUI `AUDIO` input with an `RVC_MODEL` and returns converted `AUDIO` plus runtime information. Use ComfyUI audio save nodes for file output.

### RunningHub RVC One-Click Train

Runs the complete RVC training pipeline from a dataset folder or optional audio input. The node returns `info`, which includes output `.pth`, `.index`, `.zip` paths plus a log summary. It also reports the final `.zip` as a ComfyUI output file so RunningHub post-processing can upload it to COS. `save_name` controls the final output files; `experiment_name` controls the training log and intermediate folder.

Notes:

- `f0_method=rmvpe` requires `ComfyUI/models/RVC/_assets/rmvpe/rmvpe.pt`; use `harvest` if RMVPE is not prepared.
- Training requires HuBERT at `ComfyUI/models/RVC/_assets/hubert/hubert_base.pt`.
- Runtime and VRAM use depend on dataset length, `batch_size`, `total_epoch`, and GPU.
- The output ZIP can be used directly with the ZIP Model Loader and contains same-name `.pth` plus optional `.index`.

## License

This project is licensed under the [Apache License 2.0](LICENSE). Redistribution and modified versions must retain the license and attribution notices, including the notices in [`NOTICE`](NOTICE).

The bundled RVC inference source is based on [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), which is distributed by its upstream project under the MIT License.

## Links

- [RunningHub China](https://www.runninghub.cn/?inviteCode=rh-v1367)
- [RunningHub International](https://www.runninghub.ai/?inviteCode=rh-v1367)
- [Original RVC WebUI Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [RVC core assets on Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI)

## Acknowledgements

This project is based on [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) by the RVC Project contributors.
