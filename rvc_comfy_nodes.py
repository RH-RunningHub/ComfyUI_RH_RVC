import contextlib
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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
TRAINSET_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
}
SAMPLE_RATE_VALUES = {"32k": 32000, "40k": 40000, "48k": 48000}
TRAIN_FEATURE_DIM = {"v1": 256, "v2": 768}


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


def _output_directory():
    if folder_paths is not None:
        return Path(folder_paths.get_output_directory())
    return PLUGIN_ROOT / "output"


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


def _ensure_link_or_dir(path, target=None):
    path = Path(path)
    if path.exists() or path.is_symlink():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if target is not None:
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(target, target_is_directory=True)
            return
        except OSError:
            pass
    path.mkdir(parents=True, exist_ok=True)


def _ensure_local_dir(path):
    path = Path(path)
    if path.is_symlink():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _ensure_training_layout():
    if not RVC_PROJECT_ROOT.exists():
        raise FileNotFoundError(f"找不到内置 RVC 源码目录: {RVC_PROJECT_ROOT}")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_link_or_dir(ASSETS_DIR / "hubert", RVC_ASSETS_DIR / "hubert")
    _ensure_link_or_dir(ASSETS_DIR / "rmvpe", RMVPE_DIR)
    _ensure_link_or_dir(ASSETS_DIR / "pretrained", RVC_ASSETS_DIR / "pretrained")
    _ensure_link_or_dir(ASSETS_DIR / "pretrained_v2", RVC_ASSETS_DIR / "pretrained_v2")
    _ensure_link_or_dir(ASSETS_DIR / "indices", WEIGHTS_DIR / "_indices")
    _ensure_local_dir(ASSETS_DIR / "weights")
    (RVC_PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)


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


def _run_rvc_command(
    args,
    log_path,
    cwd=None,
    extra_env=None,
    allow_training_exit=False,
    output_callback=None,
):
    _setup_rvc_environment()
    env = os.environ.copy()
    env.update(
        {
            "weight_root": str(WEIGHTS_DIR),
            "weight_uvr5_root": str(RVC_ASSETS_DIR / "uvr5_weights"),
            "index_root": str(WEIGHTS_DIR),
            "outside_index_root": str(ASSETS_DIR / "indices"),
            "rmvpe_root": str(RMVPE_DIR),
            "rvc_hubert_path": str(HUBERT_PATH),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(RVC_PROJECT_ROOT)
            + os.pathsep
            + env.get("PYTHONPATH", ""),
        }
    )
    if extra_env:
        env.update(extra_env)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(log_path, "a", encoding="utf-8", errors="ignore") as log_file:
        log_file.write("$ " + " ".join(str(arg) for arg in args) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            [str(arg) for arg in args],
            cwd=str(cwd or RVC_PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE if output_callback is not None else log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if output_callback is not None and process.stdout is not None:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                output_callback(line)
            process.wait()
        else:
            process.wait()
        log_file.write(f"[exit={process.returncode} elapsed={time.time() - started:.1f}s]\n")

    if process.returncode != 0 and not allow_training_exit:
        tail = _read_text_tail(log_path, max_chars=4000)
        raise RuntimeError(f"RVC 命令执行失败，退出码 {process.returncode}:\n{tail}")
    return process.returncode


def _make_progress_bar(total):
    try:
        import comfy.utils

        return comfy.utils.ProgressBar(max(1, int(total)))
    except Exception:
        return None


def _progress_update(progress_bar, value, total):
    if progress_bar is None:
        return
    try:
        progress_bar.update_absolute(min(int(value), int(total)), int(total))
    except Exception:
        pass


def _read_text_tail(path, max_chars=8000):
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-max_chars:]


def _read_training_log_summary(path, max_errors=5):
    path = Path(path)
    if not path.exists():
        return ""

    preprocess_done = False
    f0_seen = False
    feature_done = False
    latest_feature = ""
    last_epoch = ""
    training_done = False
    final_ckpt = ""
    last_exit = ""
    errors = []

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "end preprocess" in stripped:
            preprocess_done = True
        if "todo-f0" in stripped:
            f0_seen = True
        if "all-feature" in stripped:
            latest_feature = stripped
            if "all-feature-done" in stripped:
                feature_done = True
        match = re.search(r"====>\s*Epoch:\s*(\d+)", stripped)
        if match:
            last_epoch = match.group(1)
        if "Training is done" in stripped:
            training_done = True
        if "saving final ckpt" in stripped:
            final_ckpt = "Success" if "Success" in stripped else stripped
        if stripped.startswith("[exit="):
            last_exit = stripped
        if any(keyword in stripped for keyword in ("Traceback", "RuntimeError", "Exception", "Error")):
            errors.append(stripped)

    items = []
    if preprocess_done:
        items.append("preprocess=done")
    if f0_seen:
        items.append("f0=done")
    if feature_done:
        items.append("feature=done")
    elif latest_feature:
        items.append(f"feature={latest_feature}")
    if last_epoch:
        items.append(f"train_epoch={last_epoch}")
    if training_done:
        items.append("train=done")
    if final_ckpt:
        items.append(f"final_ckpt={final_ckpt}")
    if last_exit:
        items.append(f"last_exit={last_exit}")
    if errors:
        items.append("errors=" + " | ".join(errors[-max_errors:]))

    if not items:
        return ""
    return "training_log_summary:\n" + "\n".join(items)


def _safe_experiment_name(exp_name):
    name = _safe_path_component(exp_name, "rvc_train")
    if name in ("logs", "mute"):
        name = f"{name}_exp"
    return name


def _resolve_training_dataset(trainset_dir):
    text = str(trainset_dir or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (_input_directory() / path).resolve()
    else:
        path = path.resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"训练集目录不存在: {path}")
    return path


def _copy_audio_path_to_dataset(path, target_dir, copied):
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = (_input_directory() / source).resolve()
    else:
        source = source.resolve()
    if not source.exists():
        return copied
    if source.is_dir():
        for child in sorted(source.rglob("*")):
            copied = _copy_audio_path_to_dataset(child, target_dir, copied)
        return copied
    if source.suffix.lower() not in TRAINSET_AUDIO_EXTENSIONS:
        return copied
    copied += 1
    destination = target_dir / f"path_audio_{copied:05d}{source.suffix.lower()}"
    shutil.copy2(source, destination)
    return copied


def _write_audio_dict_to_dataset(audio, target_dir, copied):
    if not isinstance(audio, dict) or "waveform" not in audio:
        return copied
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 44100) or 44100)
    if not isinstance(waveform, torch.Tensor) or waveform.numel() == 0:
        return copied

    waveform = waveform.detach().cpu().float()
    if waveform.dim() == 1:
        waveform = waveform.reshape(1, 1, -1)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() != 3:
        waveform = waveform.reshape(1, 1, -1)

    import soundfile as sf

    for batch_idx in range(waveform.shape[0]):
        item = waveform[batch_idx].clamp(-1.0, 1.0)
        samples = item.numpy().T
        copied += 1
        sf.write(target_dir / f"audio_input_{copied:05d}.wav", samples, sample_rate)
    return copied


def _materialize_audio_training_dataset(audio, target_dir):
    copied = 0

    def visit(value):
        nonlocal copied
        if value is None:
            return
        if isinstance(value, dict):
            if "waveform" in value:
                copied = _write_audio_dict_to_dataset(value, target_dir, copied)
                return
            for key in ("audio", "audios", "file", "files", "path", "paths", "filename", "filenames"):
                if key in value:
                    visit(value[key])
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
            return
        if isinstance(value, (str, os.PathLike)):
            copied = _copy_audio_path_to_dataset(value, target_dir, copied)

    visit(audio)
    if copied == 0:
        raise ValueError("可选 audio 输入没有解析到可训练的音频。请连接 AUDIO、音频路径、音频路径列表或填写 trainset_dir。")
    return copied


def _resolve_pretrained_path(path_text):
    text = str(path_text or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        candidates = [
            (RVC_ASSETS_DIR / path).resolve(),
            (ASSETS_DIR / path).resolve(),
            (RVC_PROJECT_ROOT / path).resolve(),
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(f"预训练模型不存在: {path}")
    return str(path)


def _default_pretrained_paths(version, sample_rate, use_f0):
    path_name = "pretrained" if version == "v1" else "pretrained_v2"
    prefix = "f0" if use_f0 else ""
    g_path = RVC_ASSETS_DIR / path_name / f"{prefix}G{sample_rate}.pth"
    d_path = RVC_ASSETS_DIR / path_name / f"{prefix}D{sample_rate}.pth"
    return (str(g_path) if g_path.exists() else "", str(d_path) if d_path.exists() else "")


def _write_train_config(exp_dir, version, sample_rate):
    config_key = f"{version}/{sample_rate}.json"
    if version == "v2" and sample_rate == "40k":
        config_key = "v1/40k.json"
    source = RVC_PROJECT_ROOT / "configs" / config_key
    if not source.exists():
        raise FileNotFoundError(f"找不到训练配置: {source}")
    config_save_path = exp_dir / "config.json"
    if not config_save_path.exists():
        data = json.loads(source.read_text(encoding="utf-8"))
        config_save_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return config_save_path


def _build_filelist(exp_dir, exp_name, sample_rate, use_f0, speaker_id, version):
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    if not gt_wavs_dir.exists() or not feature_dir.exists():
        raise FileNotFoundError("请先完成训练集预处理和特征提取。")
    names = {path.stem for path in gt_wavs_dir.glob("*.wav")} & {
        path.stem for path in feature_dir.glob("*.npy")
    }
    if use_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        names &= {path.name.replace(".wav.npy", "") for path in f0_dir.glob("*.wav.npy")}
        names &= {path.name.replace(".wav.npy", "") for path in f0nsf_dir.glob("*.wav.npy")}
    if not names:
        raise RuntimeError("没有可用于训练的切片，请检查训练集预处理、F0 和特征提取日志。")

    rows = []
    for name in sorted(names):
        if use_f0:
            rows.append(
                f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|"
                f"{exp_dir}/2a_f0/{name}.wav.npy|{exp_dir}/2b-f0nsf/{name}.wav.npy|{speaker_id}"
            )
        else:
            rows.append(f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{speaker_id}")

    mute_root = RVC_PROJECT_ROOT / "logs" / "mute"
    fea_dim = TRAIN_FEATURE_DIM[version]
    for _ in range(2):
        if use_f0:
            rows.append(
                f"{mute_root}/0_gt_wavs/mute{sample_rate}.wav|{mute_root}/3_feature{fea_dim}/mute.npy|"
                f"{mute_root}/2a_f0/mute.wav.npy|{mute_root}/2b-f0nsf/mute.wav.npy|{speaker_id}"
            )
        else:
            rows.append(
                f"{mute_root}/0_gt_wavs/mute{sample_rate}.wav|"
                f"{mute_root}/3_feature{fea_dim}/mute.npy|{speaker_id}"
            )
    random.shuffle(rows)
    filelist_path = exp_dir / "filelist.txt"
    filelist_path.write_text("\n".join(rows), encoding="utf-8")
    return filelist_path, len(names)


def _train_index_for_experiment(exp_name, version, output_index_path=None):
    try:
        import faiss
        from sklearn.cluster import MiniBatchKMeans
    except Exception as exc:
        raise RuntimeError("训练索引需要 faiss-cpu 和 scikit-learn。") from exc

    exp_dir = RVC_PROJECT_ROOT / "logs" / exp_name
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    if not feature_dir.exists():
        raise FileNotFoundError("请先进行特征提取。")
    feature_files = sorted(feature_dir.glob("*.npy"))
    if not feature_files:
        raise FileNotFoundError("特征目录为空，请先进行特征提取。")

    npys = [np.load(path) for path in feature_files]
    big_npy = np.concatenate(npys, 0)
    order = np.arange(big_npy.shape[0])
    np.random.shuffle(order)
    big_npy = big_npy[order]
    messages = [f"features={big_npy.shape}"]
    if big_npy.shape[0] > 2e5:
        messages.append(f"kmeans {big_npy.shape[0]} frames to 10000 centers")
        big_npy = MiniBatchKMeans(
            n_clusters=10000,
            verbose=False,
            batch_size=256 * max(os.cpu_count() or 1, 1),
            compute_labels=False,
            init="random",
        ).fit(big_npy).cluster_centers_

    np.save(exp_dir / "total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    n_ivf = max(n_ivf, 1)
    index = faiss.index_factory(256 if version == "v1" else 768, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    trained_path = exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"
    added_path = (
        Path(output_index_path).resolve()
        if output_index_path
        else exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_{version}.index"
    )
    added_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(trained_path))
    for start in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[start : start + 8192])
    faiss.write_index(index, str(added_path))

    messages.append(f"index={added_path}")
    return added_path, "\n".join(messages)


def _package_training_outputs(save_name, model_path, index_path=""):
    safe_name = _safe_path_component(save_name, "rvc_model")
    output_dir = (_output_directory() / "RVC" / safe_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_output = Path(model_path).resolve()
    expected_model_output = (output_dir / f"{safe_name}.pth").resolve()
    if model_output != expected_model_output:
        shutil.copy2(model_output, expected_model_output)
        model_output = expected_model_output

    index_output = ""
    if index_path:
        source_index = Path(index_path).resolve()
        if source_index.exists():
            index_output_path = (output_dir / f"{safe_name}.index").resolve()
            if source_index != index_output_path:
                shutil.copy2(source_index, index_output_path)
            index_output = str(index_output_path)

    zip_path = output_dir / f"{safe_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_output, arcname=f"{safe_name}.pth")
        if index_output:
            archive.write(index_output, arcname=f"{safe_name}.index")
    return str(model_output), index_output, str(zip_path)


def _output_file_info(path):
    path = Path(path).resolve()
    output_dir = _output_directory().resolve()
    relative_path = path.relative_to(output_dir)
    return {
        "filename": path.name,
        "subfolder": str(relative_path.parent) if str(relative_path.parent) != "." else "",
        "type": "output",
    }


def _first_list_value(value):
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _resolve_device(device):
    if device != "auto":
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _auto_gpu_parts():
    if not torch.cuda.is_available():
        return []
    return [str(idx) for idx in range(torch.cuda.device_count())]


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


class RunningHubRVCOneClickTrain:
    DESCRIPTION = (
        "Runs the RVC training pipeline from a local training audio folder: preprocess, "
        "extract F0/features, train the model, and optionally build a FAISS index. "
        "The exported model and index are saved under ComfyUI output/RVC/<save_name>."
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_TOOLTIPS = (
        "训练流程日志摘要和生成文件位置。输出文件写入 ComfyUI output/RVC/<save_name>/。",
    )
    OUTPUT_NODE = True
    INPUT_IS_LIST = True
    FUNCTION = "train"
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trainset_dir": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "训练音频目录。可填写绝对路径，或填写 ComfyUI input 目录下的相对目录；目录内放 wav/mp3/flac 等音频。",
                    },
                ),
                "experiment_name": (
                    "STRING",
                    {
                        "default": "rvc_exp",
                        "multiline": False,
                        "tooltip": "实验名。会用于 rvc_source/logs/<实验名> 存放训练日志和中间产物。",
                    },
                ),
                "save_name": (
                    "STRING",
                    {
                        "default": "rvc_model",
                        "multiline": False,
                        "tooltip": "保存文件名。最终会在 ComfyUI output/RVC/<save_name>/ 下生成 <save_name>.pth、<save_name>.index 和 <save_name>.zip。",
                    },
                ),
                "version": (
                    ["v2", "v1"],
                    {
                        "default": "v2",
                        "tooltip": "RVC 模型版本。v2 使用 768 维特征；v1 使用 256 维特征。",
                    },
                ),
                "sample_rate": (
                    ["40k", "48k", "32k"],
                    {
                        "default": "40k",
                        "tooltip": "训练采样率。40k 兼容性最好；48k 高频更完整但更吃资源；32k 仅 v2 常用。",
                    },
                ),
                "use_f0": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "是否训练带音高引导的模型。唱歌和保留旋律通常开启；纯说话可按需求关闭。",
                    },
                ),
                "f0_method": (
                    ["rmvpe", "harvest", "dio", "pm"],
                    {
                        "default": "rmvpe",
                        "tooltip": "训练集音高提取算法。rmvpe 质量通常更好但需要 models/RVC/_assets/rmvpe/rmvpe.pt；harvest 稳定但较慢。",
                    },
                ),
                "speaker_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999,
                        "step": 1,
                        "tooltip": "写入训练 filelist 的说话人 ID。单人模型通常使用 0。",
                    },
                ),
                "total_epoch": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "总训练轮数。小数据集常用 20-100；数据越多可适当增加。",
                    },
                ),
                "save_every_epoch": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "每多少 epoch 保存一次 G/D checkpoint。",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "训练 batch size。显存不足时调小；调到 1 仍 OOM 时需要更大显存。",
                    },
                ),
                "cpu_processes": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "tooltip": "训练集预处理和部分 F0 提取使用的 CPU 进程数。",
                    },
                ),
                "is_half": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "特征提取和训练是否启用半精度。CPU/MPS 环境或数值异常时关闭。",
                    },
                ),
                "train_index": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "训练结束后是否基于特征构建 FAISS 检索索引。推理时 index_rate 大于 0 通常需要它。",
                    },
                ),
            },
            "optional": {
                "audio": (
                    "*",
                    {
                        "tooltip": "可选音频输入。trainset_dir 为空时使用；支持 ComfyUI AUDIO、AUDIO 列表、音频文件路径或路径列表。",
                    },
                ),
                "pretrained_G": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选预训练 Generator 路径。留空会自动尝试 assets/pretrained(_v2) 下的默认文件；不存在则从头训练。",
                    },
                ),
                "pretrained_D": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "可选预训练 Discriminator 路径。留空会自动尝试 assets/pretrained(_v2) 下的默认文件；不存在则从头训练。",
                    },
                ),
                "save_latest_only": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "是否只保留 latest G/D checkpoint。开启可减少磁盘占用。",
                    },
                ),
                "cache_dataset_in_gpu": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "是否把训练集缓存进显存。10 分钟以下小数据可加速，大数据容易显存不足。",
                    },
                ),
                "save_every_weights": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "是否每次保存 checkpoint 时额外导出可推理的小模型。关闭时仍会在训练结束导出最终模型。",
                    },
                ),
            },
        }

    def train(
        self,
        trainset_dir,
        experiment_name,
        save_name,
        version,
        sample_rate,
        use_f0,
        f0_method,
        speaker_id,
        total_epoch,
        save_every_epoch,
        batch_size,
        cpu_processes,
        is_half,
        train_index,
        audio=None,
        pretrained_G="",
        pretrained_D="",
        save_latest_only=True,
        cache_dataset_in_gpu=False,
        save_every_weights=False,
    ):
        trainset_dir = _first_list_value(trainset_dir)
        experiment_name = _first_list_value(experiment_name)
        save_name = _first_list_value(save_name)
        version = _first_list_value(version)
        sample_rate = _first_list_value(sample_rate)
        use_f0 = _first_list_value(use_f0)
        f0_method = _first_list_value(f0_method)
        speaker_id = _first_list_value(speaker_id)
        total_epoch = _first_list_value(total_epoch)
        save_every_epoch = _first_list_value(save_every_epoch)
        batch_size = _first_list_value(batch_size)
        cpu_processes = _first_list_value(cpu_processes)
        is_half = _first_list_value(is_half)
        train_index = _first_list_value(train_index)
        pretrained_G = _first_list_value(pretrained_G)
        pretrained_D = _first_list_value(pretrained_D)
        save_latest_only = _first_list_value(save_latest_only)
        cache_dataset_in_gpu = _first_list_value(cache_dataset_in_gpu)
        save_every_weights = _first_list_value(save_every_weights)

        if sample_rate == "32k" and version == "v1":
            raise ValueError("v1 不支持 32k 训练，请选择 v2 或改用 40k/48k。")
        _ensure_training_layout()
        dataset_dir = _resolve_training_dataset(trainset_dir)
        trainset_tempdir = None
        if dataset_dir is None:
            trainset_tempdir = tempfile.TemporaryDirectory(prefix="rvc_trainset_")
            extracted = _materialize_audio_training_dataset(audio, Path(trainset_tempdir.name))
            dataset_dir = Path(trainset_tempdir.name)
        exp_name = _safe_experiment_name(experiment_name)
        output_name = _safe_path_component(save_name, exp_name)
        output_dir = (_output_directory() / "RVC" / output_name).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_model_path = output_dir / f"{output_name}.pth"
        output_index_path = output_dir / f"{output_name}.index"
        exp_dir = RVC_PROJECT_ROOT / "logs" / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        train_log = exp_dir / "comfy_train.log"
        train_log.write_text("", encoding="utf-8")

        sr_value = SAMPLE_RATE_VALUES[sample_rate]
        gpu_parts = _auto_gpu_parts()
        gpus = "-".join(gpu_parts)
        feature_parts = gpu_parts or ["-"]
        progress_total = (
            1
            + (1 if use_f0 else 0)
            + len(feature_parts)
            + max(1, int(total_epoch))
            + (1 if train_index else 0)
            + 1
        )
        progress_value = 0
        progress_bar = _make_progress_bar(progress_total)

        def set_progress(value):
            nonlocal progress_value
            progress_value = max(progress_value, min(int(value), progress_total))
            _progress_update(progress_bar, progress_value, progress_total)

        set_progress(0)

        info = [
            f"experiment={exp_name}",
            f"save_name={output_name}",
            f"dataset={dataset_dir}",
            f"version={version}, sample_rate={sample_rate}, use_f0={use_f0}",
            f"gpus={gpus or 'cpu'}",
        ]
        if trainset_tempdir is not None:
            info.append(f"audio_input_files={extracted}")

        _run_rvc_command(
            [
                sys.executable,
                "infer/modules/train/preprocess.py",
                str(dataset_dir),
                sr_value,
                int(cpu_processes),
                str(exp_dir),
                "False",
                "3.7",
            ],
            train_log,
        )
        info.append("preprocess=done")
        set_progress(progress_value + 1)

        extract_log = exp_dir / "extract_f0_feature.log"
        extract_log.write_text("", encoding="utf-8")
        if use_f0:
            if f0_method == "rmvpe":
                rmvpe_model = RMVPE_DIR / "rmvpe.pt"
                if not rmvpe_model.exists():
                    raise FileNotFoundError(
                        "使用 rmvpe 提取训练音高需要模型: "
                        f"{rmvpe_model}。可改用 harvest，或先准备 rmvpe.pt。"
                    )
                if gpu_parts and torch.cuda.is_available():
                    for idx, gpu_id in enumerate(gpu_parts):
                        _run_rvc_command(
                            [
                                sys.executable,
                                "infer/modules/train/extract/extract_f0_rmvpe.py",
                                len(gpu_parts),
                                idx,
                                gpu_id,
                                str(exp_dir),
                                str(bool(is_half)),
                            ],
                            train_log,
                        )
                else:
                    _run_rvc_command(
                        [
                            sys.executable,
                            "infer/modules/train/extract/extract_f0_print.py",
                            str(exp_dir),
                            int(cpu_processes),
                            "harvest",
                        ],
                        train_log,
                    )
                    info.append("f0_method=harvest fallback because CUDA GPU is unavailable")
            else:
                _run_rvc_command(
                    [
                        sys.executable,
                        "infer/modules/train/extract/extract_f0_print.py",
                        str(exp_dir),
                        int(cpu_processes),
                        f0_method,
                    ],
                    train_log,
                )
            info.append("f0=done")
            set_progress(progress_value + 1)

        for idx, gpu_id in enumerate(feature_parts):
            args = [
                sys.executable,
                "infer/modules/train/extract_feature_print.py",
                _resolve_device("auto"),
                len(feature_parts),
                idx,
            ]
            if gpu_id != "-":
                args.extend([gpu_id, str(exp_dir), version, str(bool(is_half))])
            else:
                args.extend([str(exp_dir), version, str(bool(is_half))])
            _run_rvc_command(args, train_log)
            set_progress(progress_value + 1)
        info.append("feature=done")

        _write_train_config(exp_dir, version, sample_rate)
        filelist_path, slice_count = _build_filelist(
            exp_dir, exp_name, sample_rate, bool(use_f0), int(speaker_id), version
        )
        info.append(f"filelist={filelist_path} slices={slice_count}")

        default_g, default_d = _default_pretrained_paths(version, sample_rate, bool(use_f0))
        pretrained_G = _resolve_pretrained_path(pretrained_G) if pretrained_G else default_g
        pretrained_D = _resolve_pretrained_path(pretrained_D) if pretrained_D else default_d
        train_args = [
            sys.executable,
            "infer/modules/train/train.py",
            "-e",
            exp_name,
            "-sr",
            sample_rate,
            "-f0",
            1 if use_f0 else 0,
            "-bs",
            int(batch_size),
            "-te",
            int(total_epoch),
            "-se",
            int(save_every_epoch),
            "-l",
            1 if save_latest_only else 0,
            "-c",
            1 if cache_dataset_in_gpu else 0,
            "-sw",
            1 if save_every_weights else 0,
            "-v",
            version,
            "-od",
            str(output_dir),
            "-on",
            output_name,
        ]
        if gpus:
            train_args.extend(["-g", gpus])
        if pretrained_G:
            train_args.extend(["-pg", pretrained_G])
        if pretrained_D:
            train_args.extend(["-pd", pretrained_D])
        training_progress_start = progress_value
        epoch_pattern = re.compile(r"====>\s+Epoch:\s+(\d+)\b")

        def update_training_progress(line):
            match = epoch_pattern.search(line or "")
            if match:
                set_progress(training_progress_start + min(int(match.group(1)), int(total_epoch)))

        _run_rvc_command(
            train_args,
            train_log,
            allow_training_exit=True,
            output_callback=update_training_progress,
        )
        set_progress(training_progress_start + max(1, int(total_epoch)))

        exported_model = output_model_path
        if not exported_model.exists():
            tail = _read_text_tail(exp_dir / "train.log", max_chars=4000) or _read_text_tail(train_log)
            raise RuntimeError(f"训练结束但没有找到导出的模型: {exported_model}\n{tail}")
        info.append(f"model={exported_model}")

        index_path = ""
        if train_index:
            added_index, index_info = _train_index_for_experiment(
                exp_name, version, output_index_path
            )
            index_path = str(added_index)
            info.append(index_info)
            set_progress(progress_value + 1)

        model_output, index_output, zip_output = _package_training_outputs(
            output_name, exported_model, index_path
        )
        set_progress(progress_total)
        info.append(f"output_model={model_output}")
        info.append(f"output_index={index_output or 'not generated'}")
        info.append(f"output_zip={zip_output}")
        info.append(_read_training_log_summary(train_log))
        if trainset_tempdir is not None:
            trainset_tempdir.cleanup()
        return {
            "ui": {"rvc_zip": [_output_file_info(zip_output)]},
            "result": ("\n".join(part for part in info if part),),
        }


NODE_CLASS_MAPPINGS = {
    "RunningHubRVCModelLoader": RunningHubRVCModelLoader,
    "RunningHubRVCZipModelLoader": RunningHubRVCZipModelLoader,
    "RunningHubRVCVoiceConversion": RunningHubRVCVoiceConversion,
    "RunningHubRVCOneClickTrain": RunningHubRVCOneClickTrain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHubRVCModelLoader": "RunningHub RVC Model Loader",
    "RunningHubRVCZipModelLoader": "RunningHub RVC ZIP Model Loader",
    "RunningHubRVCVoiceConversion": "RunningHub RVC Voice Conversion",
    "RunningHubRVCOneClickTrain": "RunningHub RVC One-Click Train",
}
