import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MAX_ZIP_UPLOAD_BYTES = 150 * 1024 * 1024;

async function uploadZipModel(file) {
    if (!file.name.toLowerCase().endsWith(".zip")) {
        throw new Error("文件必须是 ZIP 格式");
    }
    if (file.size > MAX_ZIP_UPLOAD_BYTES) {
        throw new Error(`文件大小超过限制（最大 150MB），当前 ${(file.size / (1024 * 1024)).toFixed(2)}MB`);
    }

    const form = new FormData();
    form.append("file", file);

    const response = await api.fetchApi("/extensions/ComfyUI_RH_RVC/upload_zip_model", {
        method: "POST",
        body: form,
    });

    const result = await response.json().catch(() => ({}));
    if (!response.ok || !result.success) {
        throw new Error(result.error || `上传失败: ${response.status}`);
    }
    return result;
}

app.registerExtension({
    name: "RunningHub.RVC.ZipUpload",
    rh: {
        type: "nodes",
        nodes: ["RunningHubRVCZipModelLoader"],
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "RunningHubRVCZipModelLoader") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            const zipWidget = this.widgets?.find((widget) => widget.name === "zip_file");
            if (!zipWidget) {
                return;
            }
            zipWidget.label = "RVC模型ZIP";

            let input = this._rhRvcZipUploadInput;
            if (!input) {
                input = document.createElement("input");
                input.type = "file";
                input.accept = ".zip,application/zip";
                input.style.display = "none";
                document.body.appendChild(input);
                this._rhRvcZipUploadInput = input;
            }

            let uploadButton = (this.widgets || []).find((widget) => widget?.__rhRvcZipUpload === true);
            if (!uploadButton) {
                uploadButton = this.addWidget("button", "选择RVC模型ZIP上传", null, () => input.click(), {
                    serialize: false,
                });
                uploadButton.__rhRvcZipUpload = true;
            }

            input.onchange = async () => {
                const file = input.files?.[0];
                if (!file) {
                    input.value = "";
                    return;
                }

                const originalName = uploadButton.name || "选择RVC模型ZIP上传";
                try {
                    uploadButton.name = "上传中...";
                    uploadButton.disabled = true;

                    const result = await uploadZipModel(file);
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
                    alert(`RVC 模型 ZIP 上传失败: ${message}`);
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
