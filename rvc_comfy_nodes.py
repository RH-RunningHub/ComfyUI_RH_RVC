import contextlib
import hashlib
import os
import shutil
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch

try:
    import folder_paths
except Exception:  # pragma: no cover - only used outside ComfyUI for import checks.
    folder_paths = None

try:
    import aiohttp
    from aiohttp import web
    from server import PromptServer
except Exception:  # pragma: no cover - only used outside ComfyUI.
    aiohttp = None
    PromptServer = None
    web = None


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
UPLOADED_MODELS_DIR = WEIGHTS_DIR / "_uploaded"
RVC_ASSETS_DIR = WEIGHTS_DIR / "_assets"
HUBERT_PATH = RVC_ASSETS_DIR / "hubert" / "hubert_base.pt"
RMVPE_DIR = RVC_ASSETS_DIR / "rmvpe"
INDEX_DIRS = (WEIGHTS_DIR, RVC_PROJECT_ROOT / "logs", ASSETS_DIR / "indices")
CATEGORY = "RunningHub/RVC"
NO_MODEL_LABEL = "<no .pth model found in models/RVC>"

_RVC_LOCK = threading.RLock()
_MODEL_CACHE = {}
_RVC_MODULE_PREFIXES = ("configs", "infer", "i18n")
_UPLOAD_ROUTE_REGISTERED = False
MAX_ZIP_UPLOAD_BYTES = 150 * 1024 * 1024
MAX_ZIP_EXTRACT_BYTES = 4 * 1024 * 1024 * 1024
MAX_ZIP_MODEL_FILES = 128


@dataclass
class RVCModelHandle:
    vc: object
    model_name: str
    device: str
    is_half: bool
    auto_index_path: str


def _safe_path_component(value, default="model"):
    text = str(value or "").strip()
    cleaned = []
    for char in text:
        if char.isalnum() or char in ("-", "_", ".", " "):
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned).strip(" ._")
    return result or default


def _input_directory():
    if folder_paths is not None:
        return Path(folder_paths.get_input_directory())
    return PLUGIN_ROOT / "input"


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_zip_in_input(zip_file):
    text = str(zip_file or "").strip()
    if not text:
        raise ValueError("请先上传 RVC 模型 zip 文件。")

    input_dir = _input_directory()
    input_path = input_dir.resolve()
    if folder_paths is not None:
        try:
            name, base_dir = folder_paths.annotated_filepath(text)
        except Exception:
            name, base_dir = text, None
        name = str(name or "").replace("\\", "/")
        if Path(name).is_absolute() or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"zip 路径不安全: {zip_file}")
        if Path(name).suffix.lower() != ".zip":
            raise ValueError(f"只支持 RVC 模型 zip 文件: {zip_file}")
        if base_dir is not None:
            try:
                Path(base_dir).resolve().relative_to(input_path)
            except ValueError as exc:
                raise ValueError("只支持从 ComfyUI input 目录加载 zip 文件。") from exc

        if not folder_paths.exists_annotated_filepath(text):
            raise FileNotFoundError(f"找不到上传的 RVC 模型 zip: {zip_file}")
        zip_path = Path(folder_paths.get_annotated_filepath(text, str(input_dir))).resolve()
    else:
        name = text.replace("\\", "/")
        if Path(name).is_absolute() or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"zip 路径不安全: {zip_file}")
        if Path(name).suffix.lower() != ".zip":
            raise ValueError(f"只支持 RVC 模型 zip 文件: {zip_file}")
        zip_path = (input_path / name).resolve()

    try:
        zip_path.relative_to(input_path)
    except ValueError as exc:
        raise ValueError("zip 文件必须位于 ComfyUI input 目录。") from exc
    if zip_path.suffix.lower() != ".zip":
        raise ValueError(f"只支持 RVC 模型 zip 文件: {zip_file}")
    if not zip_path.exists():
        raise FileNotFoundError(f"找不到上传的 RVC 模型 zip: {zip_file}")
    if zip_path.stat().st_size > MAX_ZIP_UPLOAD_BYTES:
        raise ValueError("RVC 模型 zip 超过 150MB 限制。")
    return zip_path


def _safe_zip_member_parts(name):
    normalized = str(name or "").replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"zip 内包含不安全路径: {name}")
    if Path(normalized).is_absolute():
        raise ValueError(f"zip 内包含绝对路径: {name}")
    return [_safe_path_component(part, "file") for part in parts]


def _collect_model_files(root):
    pth_files = sorted(path for path in root.rglob("*.pth") if path.is_file())
    index_files = sorted(path for path in root.rglob("*.index") if path.is_file())
    if not pth_files:
        raise FileNotFoundError("上传的 zip 中没有找到 RVC .pth 模型文件。")

    pth_path = pth_files[0]
    stem = pth_path.stem.lower()
    preferred_indexes = [
        path for path in index_files if stem in path.name.lower() or stem in str(path.parent).lower()
    ]
    index_path = (preferred_indexes or index_files or [""])[0]
    return pth_path, str(index_path) if index_path else ""


def _copy_model_folder(source_dir, target_dir):
    copied = 0
    total_size = 0
    for source in sorted(source_dir.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in (".pth", ".index"):
            continue
        copied += 1
        if copied > MAX_ZIP_MODEL_FILES:
            raise ValueError("上传包中的模型文件过多。")
        total_size += source.stat().st_size
        if total_size > MAX_ZIP_EXTRACT_BYTES:
            raise ValueError("上传包中的模型文件总大小超过限制。")
        relative_parts = [_safe_path_component(part, "file") for part in source.relative_to(source_dir).parts]
        destination = target_dir.joinpath(*relative_parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    if copied == 0:
        raise FileNotFoundError("上传目录中没有找到 .pth 或 .index 文件。")


def _extract_model_zip(zip_path, target_dir):
    copied = 0
    total_size = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                suffix = Path(member.filename).suffix.lower()
                if suffix not in (".pth", ".index"):
                    continue
                copied += 1
                if copied > MAX_ZIP_MODEL_FILES:
                    raise ValueError("上传 zip 中的模型文件过多。")
                total_size += int(member.file_size or 0)
                if total_size > MAX_ZIP_EXTRACT_BYTES:
                    raise ValueError("上传 zip 中的模型文件总大小超过限制。")
                destination = target_dir.joinpath(*_safe_zip_member_parts(member.filename))
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, open(destination, "wb") as output:
                    shutil.copyfileobj(source, output)
    except BadZipFile as exc:
        raise ValueError(f"上传文件不是有效 zip: {zip_path}") from exc

    if copied == 0:
        raise FileNotFoundError("上传的 zip 中没有找到 .pth 或 .index 文件。")


def _prepare_uploaded_model(zip_file):
    source = _resolve_zip_in_input(zip_file)
    source_hash = _sha256_file(source)[:12]
    package_name = f"{_safe_path_component(source.stem)}_{source_hash}"

    target_dir = (UPLOADED_MODELS_DIR / package_name).resolve()
    try:
        target_dir.relative_to(WEIGHTS_DIR.resolve())
    except ValueError as exc:
        raise ValueError("上传模型目录必须位于 models/RVC 内。") from exc

    if not target_dir.exists() or not any(target_dir.rglob("*.pth")):
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        _extract_model_zip(source, target_dir)

    return (*_collect_model_files(target_dir), target_dir)


def _load_uploaded_rvc_model(zip_file, device, is_half):
    pth_path, index_path, target_dir = _prepare_uploaded_model(zip_file)
    model_name = str(pth_path.relative_to(WEIGHTS_DIR))
    handle = _load_rvc_model(model_name, device, is_half)
    if index_path:
        handle.auto_index_path = index_path
    return handle, (
        f"Model: {model_name}\n"
        f"Index: {index_path or 'not found'}\n"
        f"Extracted to: {target_dir}"
    )


def _register_upload_route():
    global _UPLOAD_ROUTE_REGISTERED
    prompt_server = getattr(PromptServer, "instance", None) if PromptServer is not None else None
    if _UPLOAD_ROUTE_REGISTERED or prompt_server is None or web is None or aiohttp is None:
        return

    @prompt_server.routes.post("/extensions/ComfyUI_RH_RVC/upload_zip_model")
    async def upload_zip_model(request):
        tmp_path = None
        try:
            reader = await request.multipart()
            uploaded_file = None
            async for part in reader:
                if part.name == "file":
                    uploaded_file = part
                    break
            if uploaded_file is None:
                return web.json_response({"success": False, "error": "missing file"}, status=400)

            filename = Path(getattr(uploaded_file, "filename", "") or "").name
            if not filename.lower().endswith(".zip"):
                return web.json_response({"success": False, "error": "only .zip files are supported"}, status=400)

            stored_name = f"{_safe_path_component(Path(filename).stem, 'rvc_model')}.zip"
            zip_path = _input_directory() / stored_name
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = zip_path.with_name(f".{zip_path.name}.tmp")

            total_size = 0
            too_large = False
            with open(tmp_path, "wb") as output:
                while True:
                    chunk = await uploaded_file.read_chunk(size=1024 * 1024)
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > MAX_ZIP_UPLOAD_BYTES:
                        too_large = True
                        continue
                    if not too_large:
                        output.write(chunk)

            if too_large:
                return web.json_response(
                    {"success": False, "error": "zip file is too large, max 150MB"},
                    status=400,
                )

            try:
                with zipfile.ZipFile(tmp_path, "r") as archive:
                    archive.testzip()
            except BadZipFile:
                return web.json_response({"success": False, "error": "invalid zip file"}, status=400)

            os.replace(tmp_path, zip_path)
            tmp_path = None
            return web.json_response({"success": True, "name": stored_name, "subfolder": "", "type": "input"})
        except Exception as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=500)
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    Path(tmp_path).unlink()

    _UPLOAD_ROUTE_REGISTERED = True


_register_upload_route()


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


class RunningHubRVCZipModelLoader:
    DESCRIPTION = (
        "Uploads or selects an RVC model zip from ComfyUI input, extracts .pth/.index "
        "files into models/RVC/_uploaded, and returns an RVC_MODEL handle."
    )
    RETURN_TYPES = ("RVC_MODEL", "STRING")
    RETURN_NAMES = ("rvc_model", "info")
    OUTPUT_TOOLTIPS = (
        "从 zip 中加载出的 RVC 模型句柄，连接到 Voice Conversion 节点使用。",
        "解压目录、模型路径和 index 自动匹配信息。",
    )
    FUNCTION = "load_model"
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zip_file": (
                    [],
                    {
                        "zip_upload": True,
                        "tooltip": "上传或选择包含 RVC .pth 和可选 .index 的 zip 文件。",
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

    def load_model(self, zip_file, device, is_half):
        return _load_uploaded_rvc_model(zip_file, device, is_half)

    @classmethod
    def IS_CHANGED(cls, zip_file, **kwargs):
        try:
            zip_path = _resolve_zip_in_input(zip_file)
        except Exception:
            return ""
        try:
            stat = zip_path.stat()
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            return ""

    @classmethod
    def VALIDATE_INPUTS(cls, zip_file, **kwargs):
        if not zip_file:
            return "请先上传或选择 RVC 模型 zip 文件。"
        try:
            _resolve_zip_in_input(zip_file)
        except Exception as exc:
            return str(exc)
        return True


class RunningHubRVCVoiceConversion:
    DESCRIPTION = (
        "Converts a ComfyUI AUDIO input with a loaded RVC model. The node returns "
        "ComfyUI AUDIO and the RVC runtime log; use ComfyUI audio save nodes for files."
    )
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    OUTPUT_TOOLTIPS = (
        "转换后的 ComfyUI AUDIO，可继续连接到音频保存、视频合成或后处理节点。",
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
                "audio": (
                    "AUDIO",
                    {"tooltip": "ComfyUI AUDIO 输入。请先用 LoadAudio、音频裁剪或音频分离节点接入音频。"},
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
            },
        }

    def convert(
        self,
        rvc_model,
        audio,
        speaker_id,
        f0_up_key,
        f0_method,
        index_path,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        temp_audio_path = ""
        try:
            temp_audio_path = _audio_to_temp_wav(audio)
            source_path = temp_audio_path
            if not source_path:
                raise ValueError("请连接 audio 输入。")

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

            extra_info = (
                f"{info}\n"
                f"Model: {rvc_model.model_name}\n"
                f"Device: {rvc_model.device}, half: {rvc_model.is_half}\n"
                f"Index: {resolved_index_path or 'not used'}"
            )
            return (output_audio, extra_info)
        finally:
            if temp_audio_path:
                try:
                    os.unlink(temp_audio_path)
                except OSError:
                    pass


NODE_CLASS_MAPPINGS = {
    "RunningHubRVCModelLoader": RunningHubRVCModelLoader,
    "RunningHubRVCZipModelLoader": RunningHubRVCZipModelLoader,
    "RunningHubRVCVoiceConversion": RunningHubRVCVoiceConversion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHubRVCModelLoader": "RunningHub RVC Model Loader",
    "RunningHubRVCZipModelLoader": "RunningHub RVC ZIP Model Loader",
    "RunningHubRVCVoiceConversion": "RunningHub RVC Voice Conversion",
}
