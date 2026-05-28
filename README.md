# ComfyUI RH RVC

- [RunningHub China](https://www.runninghub.cn/?inviteCode=rh-v1367)
- [RunningHub International](https://www.runninghub.ai/?inviteCode=rh-v1367)

![License](https://img.shields.io/badge/License-Apache%202.0-green)

ComfyUI custom nodes for Retrieval-based Voice Conversion (RVC). This plugin wraps inference code from [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) and exposes model loading, ZIP model upload, and voice conversion nodes in ComfyUI.

## Features

- Load existing RVC `.pth` models from `ComfyUI/models/RVC`.
- Upload an RVC model ZIP from the ComfyUI node UI and extract `.pth` plus optional `.index` files into `ComfyUI/models/RVC/_uploaded`.
- Convert ComfyUI `AUDIO` inputs and return ComfyUI `AUDIO` outputs for downstream save, mix, or video nodes.
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

The workflow demonstrates:

1. Load or upload an RVC model ZIP with `RunningHub RVC ZIP Model Loader`.
2. Load audio, trim it to a default end time of 20 seconds, and separate stems with `AudioSeparation`.
3. Convert the vocal stem with RVC.
4. Mix the converted vocal back with the other stems.
5. Save the final audio with a standard audio save node.

## Node Reference

### RunningHub RVC Model Loader

Loads an existing `.pth` model from `ComfyUI/models/RVC` and returns an `RVC_MODEL` handle.

### RunningHub RVC ZIP Model Loader

Uploads or selects a `.zip` file from ComfyUI input and extracts `.pth` plus optional `.index` files into `ComfyUI/models/RVC/_uploaded`.

Upload limit: 150 MB per ZIP.

### RunningHub RVC Voice Conversion

Converts a ComfyUI `AUDIO` input with an `RVC_MODEL` and returns converted `AUDIO` plus runtime information. Use ComfyUI audio save nodes for file output.

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
