# ComfyUI RH RVC

This plugin adds two ComfyUI nodes under `RunningHub/RVC`:

- `RunningHub RVC Model Loader`: loads and caches an RVC `.pth` model from `models/RVC`.
- `RunningHub RVC ZIP Model Loader`: uploads or selects an RVC model zip, extracts `.pth` and optional `.index` files into `models/RVC/_uploaded`, then returns `RVC_MODEL`.
- `RunningHub RVC Voice Conversion`: converts a ComfyUI `AUDIO` input and returns `AUDIO` plus runtime info. Use ComfyUI audio save nodes for file output.

This wrapper includes the required RVC inference source under:

`ComfyUI_RH_RVC/rvc_source`

Set `RVC_PROJECT_ROOT=/absolute/path/to/rvc_source` only if you intentionally want to override the bundled source.
The plugin repository does not include model binaries. Put all `.pth`, `.pt`, and `.index` files under `<ComfyUI>/models/RVC`.

Required model files:

- RVC voice model: `<ComfyUI>/models/RVC/<model>.pth`
- HuBERT model: `<ComfyUI>/models/RVC/_assets/hubert/hubert_base.pt`
- Optional index files: `<ComfyUI>/models/RVC/**/*.index`, `rvc_source/logs/**/*.index`, or `rvc_source/assets/indices/**/*.index`
- Optional RMVPE model for `f0_method=rmvpe`: `<ComfyUI>/models/RVC/_assets/rmvpe/rmvpe.pt`

`f0_method=pm` requires `praat-parselmouth`; `f0_method=crepe` requires `torchcrepe`. The default example uses `harvest`, and `rmvpe` uses the RMVPE model from `models/RVC/_assets`.

Example workflow:

- `examples/rvc_voice_conversion_basic_api.json`
