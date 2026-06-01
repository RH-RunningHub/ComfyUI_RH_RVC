import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ZIP_UPLOAD_TARGETS = {
    RunningHubRVCZipModelLoader: {
        widgetName: "zip_file",
        label: "RVC模型ZIP",
        buttonName: "选择RVC模型ZIP上传",
        maxBytes: 150 * 1024 * 1024,
        errorPrefix: "RVC 模型 ZIP 上传失败",
    },
};

async function uploadZipFile(file, maxBytes) {
    if (!file.name.toLowerCase().endsWith(".zip")) {
        throw new Error("文件必须是 ZIP 格式");
    }
    if (file.size > maxBytes) {
        throw new Error(`文件大小超过限制（最大 ${(maxBytes / (1024 * 1024)).toFixed(0)}MB），当前 ${(file.size / (1024 * 1024)).toFixed(2)}MB`);
    }

    const form = new FormData();
    form.append("image", file);
    form.append("type", "input");

    const response = await api.fetchApi("/upload/image", {
        method: "POST",
        body: form,
    });

    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.error || result.msg || `上传失败: ${response.status}`);
    }
    return result;
}

app.registerExtension({
    name: "RunningHub.RVC.ZipUpload",
    rh: {
        type: "nodes",
        nodes: Object.keys(ZIP_UPLOAD_TARGETS),
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const target = ZIP_UPLOAD_TARGETS[nodeData.name];
        if (!target) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            const zipWidget = this.widgets?.find((widget) => widget.name === target.widgetName);
            if (!zipWidget) {
                return;
            }
            zipWidget.label = target.label;

            let input = this[`_rhRvcZipUploadInput_${target.widgetName}`];
            if (!input) {
                input = document.createElement("input");
                input.type = "file";
                input.accept = ".zip,application/zip";
                input.style.display = "none";
                document.body.appendChild(input);
                this[`_rhRvcZipUploadInput_${target.widgetName}`] = input;
            }

            let uploadButton = (this.widgets || []).find((widget) => widget?.__rhRvcZipUpload === target.widgetName);
            if (!uploadButton) {
                uploadButton = this.addWidget("button", target.buttonName, null, () => input.click(), {
                    serialize: false,
                });
                uploadButton.__rhRvcZipUpload = target.widgetName;
            }

            input.onchange = async () => {
                const file = input.files?.[0];
                if (!file) {
                    input.value = "";
                    return;
                }

                const originalName = uploadButton.name || target.buttonName;
                try {
                    uploadButton.name = "上传中...";
                    uploadButton.disabled = true;

                    const result = await uploadZipFile(file, target.maxBytes);
                    const uploadedPath = result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
                    if (zipWidget.options?.values && !zipWidget.options.values.includes(uploadedPath)) {
                        zipWidget.options.values.push(uploadedPath);
                        zipWidget.options.values.sort();
                    }
                    zipWidget.value = uploadedPath;
                    app.graph.setDirtyCanvas(true, true);

                    uploadButton.name = "上传成功";
                    setTimeout(() => {
                        uploadButton.name = originalName;
                        uploadButton.disabled = false;
                    }, 2000);
                } catch (error) {
                    const message = error?.message || "上传失败";
                    uploadButton.name = "上传失败";
                    alert(`${target.errorPrefix}: ${message}`);
                    setTimeout(() => {
                        uploadButton.name = originalName;
                        uploadButton.disabled = false;
                    }, 3000);
                } finally {
                    input.value = "";
                }
            };
        };
    },
});
