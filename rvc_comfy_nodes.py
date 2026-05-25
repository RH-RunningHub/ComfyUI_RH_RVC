import contextlib
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    import folder_paths
except Exception:  # pragma: no cover - only used outside ComfyUI for import checks.
    folder_paths = None


PLUGIN_ROOT = Path(__file__).resolve().parent
DEFAULT_RVC_PROJECT_ROOT = PLUGIN_ROOT / "rvc_source"
RVC_PROJECT_ROOT = Path(
    os.environ.get("RVC_PROJECT_ROOT", str(DEFAULT_RVC_PROJECT_ROOT))
).expanduser().resolve()
ASSETS_DIR = RVC_PROJECT_ROOT / "assets"
COMFYUI_MODELS_DIR = Path(
    getattr(folder_paths, "models_dir", PLUGIN_ROOT.parent / "ComfyUI" / "models")
).expanduser()
WEIGHTS_DIR = COMFYUI_MODELS_DIR / "RVC"
RVC_ASSETS_DIR = WEIGHTS_DIR / "_assets"
HUBERT_PATH = RVC_ASSETS_DIR / "hubert" / "hubert_base.pt"
RMVPE_DIR = RVC_ASSETS_DIR / "rmvpe"
INDEX_DIRS = (WEIGHTS_DIR, RVC_PROJECT_ROOT / "logs", ASSETS_DIR / "indices")
CATEGORY = "RunningHub/RVC"
NO_MODEL_LABEL = "<no .pth model found in models/RVC>"

_RVC_LOCK = threading.RLock()
_MODEL_CACHE = {}
_RVC_MODULE_PREFIXES = ("configs", "infer", "i18n")


@dataclass
class RVCModelHandle:
    vc: object
    model_name: str
    device: str
    is_half: bool
    auto_index_path: str


def _list_weight_models():
    if not RVC_PROJECT_ROOT.exists():
        return [NO_MODEL_LABEL]
    if not WEIGHTS_DIR.exists():
        return [NO_MODEL_LABEL]
    names = []
    for path in WEIGHTS_DIR.rglob("*.pth"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(WEIGHTS_DIR)
        if relative_path.parts and relative_path.parts[0] == "_assets":
            continue
        names.append(str(relative_path))
    names = sorted(names)
    return names or [NO_MODEL_LABEL]


def _list_devices():
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{idx}" for idx in range(torch.cuda.device_count()))
    return devices


def _setup_rvc_environment():
    os.environ["weight_root"] = str(WEIGHTS_DIR)
    os.environ["weight_uvr5_root"] = str(RVC_ASSETS_DIR / "uvr5_weights")
    os.environ["index_root"] = str(WEIGHTS_DIR)
    os.environ["outside_index_root"] = str(ASSETS_DIR / "indices")
    os.environ["rmvpe_root"] = str(RMVPE_DIR)
    os.environ["rvc_hubert_path"] = str(HUBERT_PATH)
    os.environ.setdefault("TEMP", tempfile.gettempdir())
    os.environ.setdefault("TMP", tempfile.gettempdir())


@contextlib.contextmanager
def _rvc_runtime_context():
    old_cwd = os.getcwd()
    old_argv = sys.argv[:]
    old_sys_path = sys.path[:]
    saved_modules = {}
    try:
        _setup_rvc_environment()
        os.chdir(RVC_PROJECT_ROOT)
        root_str = str(RVC_PROJECT_ROOT)
        sys.path = [path for path in sys.path if path != root_str]
        sys.path.insert(0, root_str)
        for name in list(sys.modules):
            if name in _RVC_MODULE_PREFIXES or name.startswith(
                tuple(f"{prefix}." for prefix in _RVC_MODULE_PREFIXES)
            ):
                saved_modules[name] = sys.modules.pop(name)
        sys.argv = sys.argv[:1]
        yield
    finally:
        for name in list(sys.modules):
            if name in _RVC_MODULE_PREFIXES or name.startswith(
                tuple(f"{prefix}." for prefix in _RVC_MODULE_PREFIXES)
            ):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.argv = old_argv
        sys.path = old_sys_path
        os.chdir(old_cwd)


def _resolve_device(device):
    if device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_half(device, is_half):
    return bool(is_half and device != "cpu")


def _find_auto_index(model_name):
    stem = Path(model_name).stem
    candidates = []
    for index_dir in INDEX_DIRS:
        if not index_dir.exists():
            continue
        for path in index_dir.rglob("*.index"):
            if "trained" in path.name:
                continue
            if stem in str(path):
                candidates.append(path)
    return str(sorted(candidates)[0]) if candidates else ""


def _load_rvc_model(model_name, device, is_half):
    if not RVC_PROJECT_ROOT.exists():
        raise FileNotFoundError(
            "找不到内置 RVC 推理源码目录。默认路径为 "
            f"{DEFAULT_RVC_PROJECT_ROOT}；如需自定义，可设置 RVC_PROJECT_ROOT。"
        )

    if not model_name or model_name == NO_MODEL_LABEL:
        raise RuntimeError(
            "没有找到 RVC 模型权重。请把 .pth 模型文件放到 "
            f"{WEIGHTS_DIR} 后刷新节点。"
        )
    if Path(model_name).parts and Path(model_name).parts[0] == "_assets":
        raise ValueError("models/RVC/_assets 只用于基础模型，不可作为 RVC 声线模型加载。")

    model_path = (WEIGHTS_DIR / model_name).resolve()
    try:
        model_path.relative_to(WEIGHTS_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"RVC 模型路径必须位于 models/RVC 内: {model_name}") from exc
    if not model_path.exists():
        raise FileNotFoundError(f"RVC 模型不存在: {model_path}")

    if not HUBERT_PATH.exists():
        raise FileNotFoundError(
            "缺少 HuBERT 特征模型，RVC 推理必须使用该文件: "
            f"{HUBERT_PATH}"
        )

    resolved_device = _resolve_device(device)
    resolved_half = _resolve_half(resolved_device, is_half)
    cache_key = (model_name, resolved_device, resolved_half)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with _RVC_LOCK, _rvc_runtime_context():
        from configs.config import Config
        from infer.modules.vc.modules import VC

        config = Config()
        config.device = resolved_device
        config.is_half = resolved_half
        vc = VC(config)
        vc.get_vc(model_name)

    handle = RVCModelHandle(
        vc=vc,
        model_name=model_name,
        device=resolved_device,
        is_half=resolved_half,
        auto_index_path=_find_auto_index(model_name),
    )
    _MODEL_CACHE[cache_key] = handle
    return handle


def _audio_to_temp_wav(audio):
    if audio is None:
        return ""
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise TypeError("audio 输入必须是 ComfyUI AUDIO 格式。")

    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 44100) or 44100)
    if not isinstance(waveform, torch.Tensor) or waveform.numel() == 0:
        raise ValueError("audio 输入为空，无法执行 RVC 转换。")

    waveform = waveform.detach().cpu().float()
    if waveform.dim() == 3:
        waveform = waveform[0]
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() != 2:
        waveform = waveform.reshape(1, -1)

    samples = waveform.clamp(-1.0, 1.0).numpy().T
    suffix = ".wav"
    fd, temp_path = tempfile.mkstemp(prefix="rvc_input_", suffix=suffix)
    os.close(fd)

    try:
        import soundfile as sf

        sf.write(temp_path, samples, sample_rate)
    except Exception:
        import wave

        samples_i16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(temp_path, "wb") as wav:
            wav.setnchannels(1 if samples_i16.ndim == 1 else samples_i16.shape[1])
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(samples_i16.reshape(-1).tobytes())
    return temp_path


def _resolve_input_audio_path(input_audio_path):
    path_text = str(input_audio_path or "").strip()
    if not path_text:
        return ""

    path = Path(path_text).expanduser()
    if path.is_absolute() and path.exists():
        return str(path)

    if folder_paths is not None:
        try:
            annotated = Path(folder_paths.get_annotated_filepath(path_text))
            if annotated.exists():
                return str(annotated)
        except Exception:
            pass

        try:
            input_dir_path = Path(folder_paths.get_input_directory()) / path_text
            if input_dir_path.exists():
                return str(input_dir_path)
        except Exception:
            pass

    rvc_project_path = RVC_PROJECT_ROOT / path_text
    if rvc_project_path.exists():
        return str(rvc_project_path)

    plugin_path = PLUGIN_ROOT / path_text
    if plugin_path.exists():
        return str(plugin_path)

    raise FileNotFoundError(f"输入音频文件不存在: {input_audio_path}")


def _resolve_index_path(index_path, model_handle):
    path_text = str(index_path or "").strip()
    if not path_text:
        return model_handle.auto_index_path

    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = RVC_PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"RVC index 文件不存在: {index_path}")
    return str(path)


def _make_output_path(filename_prefix):
    prefix = str(filename_prefix or "rvc").strip() or "rvc"
    prefix = prefix.replace("\\", "_").replace("/", "_")
    if folder_paths is not None:
        out_dir = Path(folder_paths.get_output_directory())
    else:
        out_dir = PLUGIN_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    counter = 1
    while True:
        candidate = out_dir / f"{prefix}_{counter:05d}.wav"
        if not candidate.exists():
            return candidate
        counter += 1


def _wav_tuple_to_audio(wav_opt):
    sample_rate, audio_np = wav_opt
    if sample_rate is None or audio_np is None:
        raise RuntimeError("RVC 推理没有返回有效音频。")

    arr = np.asarray(audio_np)
    if arr.size == 0:
        raise RuntimeError("RVC 推理返回了空音频。")

    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_value
    else:
        arr = arr.astype(np.float32)

    arr = np.clip(arr, -1.0, 1.0)
    if arr.ndim == 1:
        channels_first = arr[None, :]
    elif arr.ndim == 2:
        channels_first = arr if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1] else arr.T
    else:
        channels_first = arr.reshape(1, -1)

    waveform = torch.from_numpy(channels_first.copy()).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _save_audio(audio, output_path):
    waveform = audio["waveform"].detach().cpu().float()
    if waveform.dim() == 3:
        waveform = waveform[0]
    samples = waveform.clamp(-1.0, 1.0).numpy().T

    try:
        import soundfile as sf

        sf.write(str(output_path), samples, int(audio["sample_rate"]))
    except Exception:
        import wave

        samples_i16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1 if samples_i16.ndim == 1 else samples_i16.shape[1])
            wav.setsampwidth(2)
            wav.setframerate(int(audio["sample_rate"]))
            wav.writeframes(samples_i16.reshape(-1).tobytes())


class RunningHubRVCModelLoader:
    DESCRIPTION = (
        "Loads a Retrieval-based Voice Conversion model from ComfyUI models/RVC "
        "and keeps it cached for later voice conversion nodes. The HuBERT model "
        "must exist at models/RVC/_assets/hubert/hubert_base.pt."
    )
    RETURN_TYPES = ("RVC_MODEL",)
    RETURN_NAMES = ("rvc_model",)
    OUTPUT_TOOLTIPS = ("已加载并缓存的 RVC 模型句柄，连接到 Voice Conversion 节点使用。",)
    FUNCTION = "load_model"
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    _list_weight_models(),
                    {
                        "tooltip": "RVC .pth 模型文件名。请将模型放在 ComfyUI 的 models/RVC 目录后刷新节点列表。"
                    },
                ),
                "device": (
                    _list_devices(),
                    {
                        "default": "auto",
                        "tooltip": "推理设备。auto 会优先使用 cuda:0，没有 CUDA 时使用 CPU。",
                    },
                ),
                "is_half": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "是否使用半精度加载模型。CPU 会自动禁用；老显卡或精度异常时可关闭。",
                    },
                ),
            }
        }

    def load_model(self, model_name, device, is_half):
        return (_load_rvc_model(model_name, device, is_half),)


class RunningHubRVCVoiceConversion:
    DESCRIPTION = (
        "Converts an input voice with a loaded RVC model. Provide either a ComfyUI "
        "AUDIO input or an audio file path. The node returns ComfyUI AUDIO, the "
        "saved WAV path, and the RVC runtime log."
    )
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "output_path", "info")
    OUTPUT_TOOLTIPS = (
        "转换后的 ComfyUI AUDIO，可继续连接到音频保存、视频合成或后处理节点。",
        "转换后 WAV 文件在 ComfyUI output 目录中的保存路径。",
        "RVC 推理日志，包含模型、设备、index 使用情况和耗时信息。",
    )
    FUNCTION = "convert"
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rvc_model": (
                    "RVC_MODEL",
                    {"tooltip": "由 RunningHub RVC Model Loader 输出的已加载模型。"},
                ),
                "input_audio_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选输入音频路径。连接 audio 输入时可留空；支持绝对路径、ComfyUI input 文件名或相对本插件目录路径。",
                    },
                ),
                "speaker_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999,
                        "step": 1,
                        "tooltip": "目标说话人 ID。单说话人模型通常为 0，多说话人模型按训练时的 ID 选择。",
                    },
                ),
                "f0_up_key": (
                    "INT",
                    {
                        "default": 0,
                        "min": -24,
                        "max": 24,
                        "step": 1,
                        "tooltip": "升降调半音数。正数升调，负数降调，0 保持原调。",
                    },
                ),
                "f0_method": (
                    ["harvest", "pm", "dio", "crepe", "rmvpe"],
                    {
                        "default": "harvest",
                        "tooltip": "F0 提取算法。harvest 稳定但较慢；pm 较快；rmvpe 通常质量更好但需要 models/RVC/_assets/rmvpe/rmvpe.pt。",
                    },
                ),
                "index_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选 .index 文件路径。留空时会按模型名在 logs 和 assets/indices 下自动查找；没有 index 时仍可转换。",
                    },
                ),
                "index_rate": (
                    "FLOAT",
                    {
                        "default": 0.66,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "检索特征混合比例。值越高越贴近目标音色，但可能带来咬字或噪声问题。",
                    },
                ),
                "filter_radius": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 7,
                        "step": 1,
                        "tooltip": "F0 中值滤波半径。大于 0 可减少毛刺，通常 3 即可。",
                    },
                ),
                "resample_sr": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 48000,
                        "step": 1000,
                        "tooltip": "输出重采样率。0 表示使用模型原始采样率；设置为 16000 以上会重采样输出。",
                    },
                ),
                "rms_mix_rate": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "响度包络混合比例。1 更保留输入响度，0 更接近目标模型响度。",
                    },
                ),
                "protect": (
                    "FLOAT",
                    {
                        "default": 0.33,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "辅音和呼吸声保护强度。数值越大越保护原音，音色转换程度会降低。",
                    },
                ),
                "output_prefix": (
                    "STRING",
                    {
                        "default": "rvc",
                        "multiline": False,
                        "tooltip": "保存到 ComfyUI output 目录的 WAV 文件名前缀。",
                    },
                ),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {"tooltip": "可选 ComfyUI AUDIO 输入。连接后优先使用该音频，input_audio_path 可留空。"},
                )
            },
        }

    def convert(
        self,
        rvc_model,
        input_audio_path,
        speaker_id,
        f0_up_key,
        f0_method,
        index_path,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        output_prefix,
        audio=None,
    ):
        temp_audio_path = ""
        try:
            if audio is not None:
                temp_audio_path = _audio_to_temp_wav(audio)
                source_path = temp_audio_path
            else:
                source_path = _resolve_input_audio_path(input_audio_path)

            if not source_path:
                raise ValueError("请连接 audio 输入，或填写 input_audio_path。")

            resolved_index_path = _resolve_index_path(index_path, rvc_model)

            with _RVC_LOCK, _rvc_runtime_context():
                info, wav_opt = rvc_model.vc.vc_single(
                    int(speaker_id),
                    source_path,
                    int(f0_up_key),
                    None,
                    f0_method,
                    resolved_index_path,
                    None,
                    float(index_rate),
                    int(filter_radius),
                    int(resample_sr),
                    float(rms_mix_rate),
                    float(protect),
                )

            if wav_opt is None or wav_opt[0] is None or wav_opt[1] is None:
                raise RuntimeError(str(info or "RVC 推理失败。"))

            output_audio = _wav_tuple_to_audio(wav_opt)
            output_path = _make_output_path(output_prefix)
            _save_audio(output_audio, output_path)

            extra_info = (
                f"{info}\n"
                f"Model: {rvc_model.model_name}\n"
                f"Device: {rvc_model.device}, half: {rvc_model.is_half}\n"
                f"Index: {resolved_index_path or 'not used'}\n"
                f"Output: {output_path}"
            )
            return (output_audio, str(output_path), extra_info)
        finally:
            if temp_audio_path:
                try:
                    os.unlink(temp_audio_path)
                except OSError:
                    pass


NODE_CLASS_MAPPINGS = {
    "RunningHubRVCModelLoader": RunningHubRVCModelLoader,
    "RunningHubRVCVoiceConversion": RunningHubRVCVoiceConversion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHubRVCModelLoader": "RunningHub RVC Model Loader",
    "RunningHubRVCVoiceConversion": "RunningHub RVC Voice Conversion",
}
