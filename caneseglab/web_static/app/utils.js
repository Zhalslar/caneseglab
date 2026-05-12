export function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("show"), 2800);
}

export function setValue(selector, value) {
  const input = document.querySelector(selector);
  if (!input) return;
  if (input.type === "checkbox") {
    input.checked = Boolean(value);
    return;
  }
  input.value = value ?? "";
}

export function readForm(form) {
  const data = new FormData(form);
  const entries = Object.fromEntries(data.entries());
  for (const input of form.querySelectorAll('input[type="checkbox"]')) {
    entries[input.name] = input.checked;
  }
  return entries;
}

export function toNumber(value) {
  if (value === "" || value === null || value === undefined) return undefined;
  return Number(value);
}

export function fileUrl(path, stamp = "") {
  const suffix = stamp ? `&t=${encodeURIComponent(stamp)}` : "";
  return `/api/file?path=${encodeURIComponent(path)}${suffix}`;
}

export function formatSize(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

export function statusText(status) {
  return {
    queued: "排队中",
    running: "执行中",
    succeeded: "已完成",
    failed: "失败",
  }[status] || status;
}

export function kindText(kind) {
  return {
    mask: "标注转掩码",
    "auto-label": "打标",
    "infer-dataset": "推理",
    train: "训练模型",
    "export-onnx": "导出 ONNX",
    "export-trt": "导出 TensorRT",
    "verify-onnx": "ONNX 推理",
    "infer-trt": "TensorRT 推理",
    "navigate-onnx": "ONNX 导航",
    "navigate-trt": "TensorRT 导航",
    "navigate-dataset": "批量导航",
    benchmark: "推理测速",
    "export-paper-figures": "论文图导出",
    "audit-navigation": "导航结果审计",
  }[kind] || kind;
}

export function datasetLabel(dataset) {
  return `${dataset.name} (${dataset.image_count} 张图像)`;
}

function fileNameFromPath(path) {
  return String(path || "").split(/[\\/]/).pop() || "";
}

export function inferenceModelFilename(method) {
  return method === "infer-trt" || method === "navigate-trt" ? "model.trt" : "model.onnx";
}

export function listInferenceModels(artifacts, method, configuredPath = "") {
  const filename = inferenceModelFilename(method).toLowerCase();
  const items = [];
  const seen = new Set();

  for (const file of artifacts?.root_files || []) {
    if (String(file.name || "").toLowerCase() !== filename) {
      continue;
    }
    if (seen.has(file.path)) {
      continue;
    }
    seen.add(file.path);
    items.push({
      path: file.path,
      name: file.name,
      runName: "产物根目录",
      runPath: artifacts?.root_dir || file.path,
      label: `产物根目录 / ${file.name}`,
    });
  }

  for (const run of artifacts?.runs || []) {
    for (const file of run.files || []) {
      if (String(file.name || "").toLowerCase() !== filename) {
        continue;
      }
      if (seen.has(file.path)) {
        continue;
      }
      seen.add(file.path);
      items.push({
        path: file.path,
        name: file.name,
        runName: run.name,
        runPath: run.path,
        label: `${run.name} / ${file.name}`,
      });
    }
  }

  if (configuredPath && !seen.has(configuredPath)) {
    items.unshift({
      path: configuredPath,
      name: fileNameFromPath(configuredPath) || inferenceModelFilename(method),
      runName: "当前配置",
      runPath: configuredPath,
      label: `当前配置 / ${fileNameFromPath(configuredPath) || configuredPath}`,
    });
  }

  return items;
}
