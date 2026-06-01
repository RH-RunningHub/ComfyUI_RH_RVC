# ComfyUI RH RVC

- [RunningHub 中国站](https://www.runninghub.cn/?inviteCode=rh-v1367)
- [RunningHub 国际站](https://www.runninghub.ai/?inviteCode=rh-v1367)

![License](https://img.shields.io/badge/License-Apache%202.0-green)

这是一个用于 Retrieval-based Voice Conversion (RVC) 的 ComfyUI 自定义节点插件。插件封装了 [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) 的推理和训练代码，并在 ComfyUI 中提供模型加载、ZIP 模型上传、音频变声和一键训练节点。

## 功能特点

- 从 `ComfyUI/models/RVC` 加载已有 RVC `.pth` 模型。
- 在 ComfyUI 节点界面上传 RVC 模型 ZIP，并把 `.pth` 与可选 `.index` 解压到 `ComfyUI/models/RVC/_uploaded`。
- 接收 ComfyUI `AUDIO` 输入，输出转换后的 ComfyUI `AUDIO`，可继续连接保存、混音或视频节点。
- 从本地训练集目录执行 RVC 预处理、F0/特征提取、模型训练和索引训练。
- 训练节点会把 `<save_name>.pth`、`<save_name>.index` 和 `<save_name>.zip` 保存到 `ComfyUI/output/RVC/<save_name>/`。
- 从 `ComfyUI/models/RVC/_assets/hubert` 使用 HuBERT。
- 可选支持 RMVPE 音高提取，模型路径为 `ComfyUI/models/RVC/_assets/rmvpe/rmvpe.pt`。
- 模型二进制文件不放进插件仓库。

## 安装

把本仓库克隆到 `ComfyUI/custom_nodes`：

```bash
cd ComfyUI/custom_nodes
git clone <repository-url> ComfyUI_RH_RVC
cd ComfyUI_RH_RVC
pip install -r requirements.txt
```

安装后重启 ComfyUI。

插件内置 RVC 推理源码：

```text
ComfyUI_RH_RVC/rvc_source
```

只有在你明确想使用其他 RVC 源码目录时，才需要设置 `RVC_PROJECT_ROOT=/absolute/path/to/rvc_source`。

## 模型下载与安装

插件仓库不包含模型二进制文件。所有 RVC 资源都应放到 `ComfyUI/models/RVC`。

### 模型目录结构

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

必需文件：

| 文件 | 是否必需 | 目标路径 | 说明 |
| --- | --- | --- | --- |
| RVC `.pth` 声音模型 | 是 | `ComfyUI/models/RVC/<model-folder>/` | 主变声模型。 |
| HuBERT | 是 | `ComfyUI/models/RVC/_assets/hubert/hubert_base.pt` | 内容特征模型。 |
| RVC `.index` | 可选 | 与 `.pth` 同目录，或 `models/RVC` 下任意子目录 | 用于改善音色匹配。 |
| RMVPE | 可选 | `ComfyUI/models/RVC/_assets/rmvpe/rmvpe.pt` | 仅在 `f0_method=rmvpe` 时需要。 |
| 预训练 G/D | 可选 | `ComfyUI/models/RVC/_assets/pretrained` 或 `_assets/pretrained_v2` | 训练时留空会自动尝试这些路径，缺失则从头训练。 |

### 下载方式

#### 方法 1：使用 aria2 下载核心资产

在 `ComfyUI/models/RVC` 目录下执行：

```bash
mkdir -p _assets/hubert _assets/rmvpe
aria2c -x 16 -s 16 -k 1M -c -o _assets/hubert/hubert_base.pt \
  https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt
aria2c -x 16 -s 16 -k 1M -c -o _assets/rmvpe/rmvpe.pt \
  https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
```

#### 方法 2：使用 hf 下载核心资产

```bash
cd ComfyUI/models/RVC
mkdir -p _assets/hubert _assets/rmvpe
hf download lj1995/VoiceConversionWebUI hubert_base.pt --local-dir _assets/hubert
hf download lj1995/VoiceConversionWebUI rmvpe.pt --local-dir _assets/rmvpe
```

#### 方法 3：手动下载

| 模型 | 链接 | 目标路径 |
| --- | --- | --- |
| HuBERT | https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt | `ComfyUI/models/RVC/_assets/hubert/hubert_base.pt` |
| RMVPE | https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt | `ComfyUI/models/RVC/_assets/rmvpe/rmvpe.pt` |

RVC `.pth` 和可选 `.index` 声音模型需要你自行准备。`.pth` 与 `.index` 文件名不强制相同，但推荐使用相同 stem，自动匹配会更稳定。

### 模型选择指南

| 使用场景 | 推荐配置 | 说明 |
| --- | --- | --- |
| 基础变声 | `.pth` + HuBERT | 没有 index 也能运行，但质量可能较低。 |
| 更好音色匹配 | `.pth` + 匹配 `.index` + HuBERT | 把 `.index` 放在 `.pth` 附近，或使用 ZIP Loader。 |
| RMVPE 音高提取 | `.pth` + HuBERT + RMVPE | 仅 `f0_method=rmvpe` 时需要。 |

### F0 方法说明

| 方法 | 额外要求 |
| --- | --- |
| `harvest` | 使用 `pyworld`，由 `requirements.txt` 安装。 |
| `dio` | 使用 `pyworld`，由 `requirements.txt` 安装。 |
| `pm` | 使用 `praat-parselmouth`，由 `requirements.txt` 安装。 |
| `crepe` | 使用 `torchcrepe`，由 `requirements.txt` 安装。 |
| `rmvpe` | 需要 `models/RVC/_assets/rmvpe/rmvpe.pt`。 |

## 使用方法

### 示例工作流

导入以下工作流到 ComfyUI：

- [`examples/rvc_voice_conversion_basic_api.json`](examples/rvc_voice_conversion_basic_api.json)
- [`examples/rvc_training_basic_api.json`](examples/rvc_training_basic_api.json)

该工作流演示：

1. 使用 `RunningHub RVC ZIP Model Loader` 加载或上传 RVC 模型 ZIP。
2. 加载音频，默认裁剪到 20 秒，并通过 `AudioSeparation` 分离音轨。
3. 使用 RVC 转换人声 stem。
4. 把转换后的人声与其他 stems 混合。
5. 使用标准音频保存节点保存最终音频。

训练示例工作流使用 `RunningHub RVC One-Click Train`。训练数据有两种输入方式：

1. `trainset_dir`：训练音频目录，默认留空；填写时优先使用该目录。
2. `audio`：可选音频输入，`trainset_dir` 为空时使用。支持 ComfyUI `AUDIO`、`AUDIO` 列表、音频文件路径或路径列表，可连接能输出音频/路径列表的上游节点。

`trainset_dir` 示例：

- 绝对路径：`/workspace/ComfyUI/input/my_rvc_trainset`
- ComfyUI input 相对路径：`my_rvc_trainset`

训练节点的 `save_name` 会决定输出文件名。完成后会生成：

```text
ComfyUI/output/RVC/<save_name>/<save_name>.pth
ComfyUI/output/RVC/<save_name>/<save_name>.index
ComfyUI/output/RVC/<save_name>/<save_name>.zip
```

## 节点说明

### RunningHub RVC Model Loader

从 `ComfyUI/models/RVC` 加载已有 `.pth` 模型，并返回 `RVC_MODEL` 句柄。

### RunningHub RVC ZIP Model Loader

从 ComfyUI input 上传或选择 `.zip` 文件，并把 `.pth` 与可选 `.index` 解压到 `ComfyUI/models/RVC/_uploaded`。

上传限制：每个 ZIP 最大 150 MB。

### RunningHub RVC Voice Conversion

使用 `RVC_MODEL` 转换 ComfyUI `AUDIO` 输入，返回转换后的 `AUDIO` 和运行信息。文件输出请连接 ComfyUI 的音频保存节点。

### RunningHub RVC One-Click Train

从训练集目录或可选音频输入执行完整 RVC 训练流程，返回 `info`，其中包含 `.pth`、`.index`、`.zip` 路径和日志摘要。节点还会把最终 `.zip` 作为 ComfyUI 输出文件上报，便于 RunningHub 后处理上传到 COS。`save_name` 是最终输出文件名；`experiment_name` 是训练日志和中间产物目录名。

注意事项：

- `f0_method=rmvpe` 需要 `ComfyUI/models/RVC/_assets/rmvpe/rmvpe.pt`，否则可先用 `harvest`。
- 训练需要 HuBERT：`ComfyUI/models/RVC/_assets/hubert/hubert_base.pt`。
- 训练耗时和显存占用取决于训练集长度、`batch_size`、`total_epoch` 和 GPU。
- 输出 zip 可直接作为 ZIP Model Loader 的输入包，里面包含同名 `.pth` 和可选 `.index`。

## 许可证

本项目使用 [Apache License 2.0](LICENSE)。分发或修改版本时必须保留许可证和署名信息，包括 [`NOTICE`](NOTICE) 中的声明。

内置 RVC 推理源码基于 [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)，其上游项目使用 MIT License 发布。

## 相关链接

- [RunningHub 中国站](https://www.runninghub.cn/?inviteCode=rh-v1367)
- [RunningHub 国际站](https://www.runninghub.ai/?inviteCode=rh-v1367)
- [RVC WebUI 原项目](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [RVC 核心资产 Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI)

## 致谢

本项目基于 RVC Project 贡献者开发的 [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)。
