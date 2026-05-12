import { state } from "./state.js?v=20260325c";
import {
  datasetLabel,
  escapeHtml,
  fileUrl,
  formatSize,
  inferenceModelFilename,
  kindText,
  listInferenceModels,
  statusText,
} from "./utils.js?v=20260325c";

function renderImageCompare(label, path, stamp = "", extraClass = "") {
  if (!path) return "";
  const url = fileUrl(path, stamp);
  return `
    <figure class="compare-card ${extraClass}">
      <figcaption>${escapeHtml(label)}</figcaption>
      <button
        type="button"
        class="image-preview-trigger"
        data-preview-src="${url}"
        data-preview-label="${escapeHtml(label)}"
      >
        <img class="compare-image" src="${url}" alt="${escapeHtml(label)}" loading="lazy">
      </button>
      <div class="compare-actions">
        <button
          type="button"
          class="ghost-button image-preview-inline"
          data-preview-src="${url}"
          data-preview-label="${escapeHtml(label)}"
        >放大查看</button>
        <a class="file-link" href="${url}" target="_blank" rel="noreferrer">新窗口打开</a>
      </div>
      <div class="meta compare-path">${escapeHtml(path)}</div>
    </figure>
  `;
}

function imageLabel(image) {
  return image?.relative_path || image?.name || "";
}

function inferenceSelectionKey(method) {
  return method === "infer-trt" || method === "navigate-trt" ? "enginePath" : "onnxPath";
}

function renderResultLink(label, path, stamp = "") {
  if (!path) return "";
  const url = fileUrl(path, stamp);
  return `<a class="file-link" href="${url}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`;
}

function renderResultLinks(title, items, stamp = "") {
  const rows = (items || []).filter((item) => item?.path);
  if (!rows.length) return "";
  return `
    <section class="job-detail-section">
      <h3>${escapeHtml(title)}</h3>
      <div class="result-link-list">
        ${rows.map((item) => `
          <div class="result-link-row">
            <span>${escapeHtml(item.label)}</span>
            ${renderResultLink(item.label, item.path, stamp)}
          </div>
        `).join("")}
      </div>
    </section>
  `;
}

function setSelectOptions(selectId, items, getValue, getLabel, options = {}) {
  const select = document.getElementById(selectId);
  if (!select) return "";

  const previousValue = select.value;
  const placeholderLabel = options.placeholderLabel || "暂无可选项";
  const placeholderValue = options.placeholderValue || "";
  const includePlaceholder = options.includePlaceholder !== false;
  const nextItems = items || [];

  const fragments = [];
  if (includePlaceholder) {
    fragments.push(`<option value="${escapeHtml(placeholderValue)}">${escapeHtml(placeholderLabel)}</option>`);
  }
  fragments.push(
    ...nextItems.map((item) => `
      <option value="${escapeHtml(getValue(item))}">${escapeHtml(getLabel(item))}</option>
    `),
  );
  select.innerHTML = fragments.join("");

  const candidateValues = new Set(nextItems.map((item) => getValue(item)));
  let nextValue = previousValue;
  if (!candidateValues.has(nextValue)) {
    nextValue = options.preferredValue;
  }
  if (!candidateValues.has(nextValue)) {
    nextValue = includePlaceholder ? placeholderValue : (nextItems[0] ? getValue(nextItems[0]) : "");
  }

  select.value = nextValue || "";
  select.disabled = !nextItems.length && !(includePlaceholder && placeholderValue === "");
  return select.value;
}

function renderBenchmarkResult(job) {
  const result = job.result || {};
  const cards = (result.results || []).map((item) => `
    <article class="metric-result-card">
      <strong>${escapeHtml(item.backend || "-")}</strong>
      <div class="meta">状态：${escapeHtml(item.status || "-")}</div>
      <div class="metric-result-grid">
        <span>平均耗时</span><strong>${escapeHtml(String(item.avg_ms ?? "-"))} ms</strong>
        <span>P50</span><strong>${escapeHtml(String(item.p50_ms ?? "-"))} ms</strong>
        <span>P95</span><strong>${escapeHtml(String(item.p95_ms ?? "-"))} ms</strong>
        <span>FPS</span><strong>${escapeHtml(String(item.fps ?? "-"))}</strong>
      </div>
      ${item.error ? `<pre>${escapeHtml(item.error)}</pre>` : ""}
    </article>
  `).join("");

  const summary = [
    `图像数量：${result.image_count ?? "-"}`,
    `预热次数：${result.warmup_runs ?? "-"}`,
    `计时次数：${result.timed_runs ?? "-"}`,
    `计时范围：${result.timing_scope || "-"}`,
  ].join("\n");

  return `
    <section class="job-detail-section">
      <h3>测速摘要</h3>
      <pre>${escapeHtml(summary)}</pre>
    </section>
    <section class="job-detail-section">
      <h3>后端对比</h3>
      <div class="metric-result-list">${cards || '<div class="empty">当前没有可展示的测速结果。</div>'}</div>
    </section>
    ${renderResultLinks("测速导出", [
      { label: "测速汇总 JSON", path: result.summary_path },
      { label: "测速表 CSV", path: result.table_path },
    ], job.finished_at || job.created_at || "")}
  `;
}

function renderPaperFiguresResult(job) {
  const result = job.result || {};
  const stamp = job.finished_at || job.created_at || "";
  const figureLabelMap = {
    training_loss: "损失曲线",
    validation_metrics: "验证指标",
    learning_rate: "学习率",
    navigation_status: "导航结果统计",
    navigation_metrics: "导航指标分布",
  };
  const previewCards = [];
  for (const paths of Object.values(result.figures || {})) {
    for (const path of paths || []) {
      if (!String(path).toLowerCase().endsWith(".png")) continue;
      previewCards.push(renderImageCompare(String(path).split(/[\\/]/).pop() || "figure", path, stamp));
    }
  }

  return `
    ${previewCards.length ? `
      <section class="job-detail-section">
        <h3>论文图预览</h3>
        <div class="compare-grid">${previewCards.join("")}</div>
      </section>` : ""}
    ${renderResultLinks("论文图输出", [
      { label: "导出清单", path: `${result.output_dir || ""}\\figure_manifest.json` },
      ...Object.entries(result.figures || {}).flatMap(([name, paths]) =>
        (paths || []).map((path, index) => ({
          label: `${figureLabelMap[name] || name}${(paths || []).length > 1 ? ` #${index + 1}` : ""}`,
          path,
        })),
      ),
    ], stamp)}
  `;
}

function renderNavigationAuditResult(job) {
  const result = job.result || {};
  const lines = [
    `导航 JSON 总数：${result.total_nav_json ?? "-"}`,
    ...Object.entries(result.status_counts || {}).map(([key, value]) => `${key}：${value}`),
  ].join("\n");

  return `
    <section class="job-detail-section">
      <h3>审计摘要</h3>
      <pre>${escapeHtml(lines)}</pre>
    </section>
    ${renderResultLinks("审计输出", [
      { label: "审计汇总 JSON", path: result.summary_path },
      { label: "失败样本 CSV", path: result.failure_table_path },
    ], job.finished_at || job.created_at || "")}
  `;
}

function renderInferenceResult(job) {
  const result = job.result || {};
  const stamp = job.finished_at || job.created_at || "";
  const originalPath = result.image_path || job.payload?.image_path || "";
  const compareHtml = [
    renderImageCompare("原图", originalPath, stamp, "primary"),
    renderImageCompare("叠加结果", result.overlay_output, stamp),
    renderImageCompare("掩码结果", result.mask_output, stamp),
  ].filter(Boolean).join("");

  const detail = [];
  if (originalPath) detail.push(`原图：${originalPath}`);
  if (result.overlay_output) detail.push(`叠加图：${result.overlay_output}`);
  if (result.mask_output) detail.push(`掩码图：${result.mask_output}`);

  return `
    <section class="job-detail-section">
      <h3>推理结果对比</h3>
      <div class="compare-grid">${compareHtml || '<div class="empty">推理结果已生成，但当前没有可展示的图片。</div>'}</div>
    </section>
    ${detail.length ? `<section class="job-detail-section"><h3>结果路径</h3><pre>${escapeHtml(detail.join("\n"))}</pre></section>` : ""}
  `;
}

function renderNavigationResult(job) {
  const result = job.result || {};
  const navigation = result.result || {};
  const stamp = job.finished_at || job.created_at || "";
  const originalPath = result.image_path || job.payload?.image_path || "";
  const compareHtml = [
    renderImageCompare("原图", originalPath, stamp, "primary"),
    renderImageCompare("分割覆盖图", result.overlay_path || result.overlay_output, stamp),
    renderImageCompare("掩码结果", result.mask_path || result.mask_output, stamp),
    renderImageCompare("导航覆盖图", result.nav_overlay_output, stamp),
    renderImageCompare("BEV 俯视图", result.bev_overlay_output, stamp),
  ].filter(Boolean).join("");

  const summary = [
    `状态：${navigation.status || "-"}`,
    `横向误差(px)：${navigation.lateral_error_px ?? navigation.center_offset_px ?? "-"}`,
    `航向角(deg)：${navigation.heading_angle_deg ?? "-"}`,
    `前视误差(px)：${navigation.lookahead_error_px ?? "-"}`,
    `通道宽度(px)：${navigation.corridor_width_px ?? "-"}`,
    `置信度：${navigation.confidence ?? "-"}`,
    `拟合误差：${navigation.fit_rmse ?? "-"}`,
    `有效点数：${navigation.point_count ?? "-"}`,
  ].join("\n");

  const detail = [];
  if (result.overlay_path || result.overlay_output) detail.push(`分割覆盖图：${result.overlay_path || result.overlay_output}`);
  if (result.mask_path || result.mask_output) detail.push(`掩码图：${result.mask_path || result.mask_output}`);
  if (result.nav_overlay_output) detail.push(`导航图：${result.nav_overlay_output}`);
  if (result.bev_overlay_output) detail.push(`BEV 图：${result.bev_overlay_output}`);
  if (result.nav_json_output) detail.push(`导航数据：${result.nav_json_output}`);

  return `
    <section class="job-detail-section">
      <h3>导航结果对比</h3>
      <div class="compare-grid">${compareHtml || '<div class="empty">导航结果已生成，但当前没有可展示的图片。</div>'}</div>
    </section>
    <section class="job-detail-section">
      <h3>导航摘要</h3>
      <pre>${escapeHtml(summary)}</pre>
    </section>
    ${detail.length ? `<section class="job-detail-section"><h3>结果路径</h3><pre>${escapeHtml(detail.join("\n"))}</pre></section>` : ""}
  `;
}

function renderCommonResult(job) {
  if (!job.result) {
    return '<div class="empty">任务正在等待结果。</div>';
  }

  if (job.kind === "verify-onnx" || job.kind === "infer-trt") {
    return renderInferenceResult(job);
  }

  if (job.kind === "navigate-onnx" || job.kind === "navigate-trt") {
    return renderNavigationResult(job);
  }

  if (job.kind === "benchmark") {
    return renderBenchmarkResult(job);
  }

  if (job.kind === "export-paper-figures") {
    return renderPaperFiguresResult(job);
  }

  if (job.kind === "audit-navigation") {
    return renderNavigationAuditResult(job);
  }

  return `
    <section class="job-detail-section">
      <h3>执行结果</h3>
      <pre>${escapeHtml(JSON.stringify(job.result, null, 2))}</pre>
    </section>
  `;
}

function renderJobPayload(job) {
  if (!job.payload || !Object.keys(job.payload).length) {
    return "";
  }
  return `
    <section class="job-detail-section">
      <h3>提交参数</h3>
      <pre>${escapeHtml(JSON.stringify(job.payload, null, 2))}</pre>
    </section>
  `;
}

function renderJobError(job) {
  if (!job.error) {
    return "";
  }
  return `
    <section class="job-detail-section">
      <h3>错误信息</h3>
      <pre>${escapeHtml(job.error)}</pre>
    </section>
  `;
}

export function populateConfig(config) {
  document.getElementById("page-title").textContent = config.web.title;
  document.title = config.web.title;
  document.getElementById("single-job-mode").textContent = config.web.single_job_mode ? "单任务" : "多任务";
  document.getElementById("artifacts-root").textContent = config.inference.artifacts_dir;
}

export function renderPageTabs() {
  for (const button of document.querySelectorAll(".tab-button")) {
    button.classList.toggle("active", button.dataset.page === state.activePage);
  }
  for (const panel of document.querySelectorAll("[data-page-panel]")) {
    panel.classList.toggle("active", panel.dataset.pagePanel === state.activePage);
  }
}

export function renderDatasetOptions(selectId, selectedName = "") {
  const select = document.getElementById(selectId);
  if (!select) return;

  if (!state.datasets.length) {
    select.innerHTML = '<option value="">暂无可用数据集</option>';
    select.disabled = true;
    return;
  }

  const selected = selectedName || select.value || state.datasets[0].name;
  select.innerHTML = state.datasets
    .map((dataset) => `<option value="${escapeHtml(dataset.name)}">${escapeHtml(datasetLabel(dataset))}</option>`)
    .join("");
  select.value = state.datasetMap[selected] ? selected : state.datasets[0].name;
  select.disabled = false;
}

export function renderDatasetOverview(rootDir, datasets) {
  document.getElementById("dataset-root").textContent = rootDir || "-";

  const container = document.getElementById("dataset-cards");
  if (!datasets.length) {
    container.innerHTML = '<div class="empty">当前没有发现可用数据集。</div>';
    return;
  }

  container.innerHTML = datasets.map((dataset) => {
    const imageCount = Number.isFinite(dataset.image_count) ? dataset.image_count : 0;
    const annotationCount = Number.isFinite(dataset.annotation_count) ? dataset.annotation_count : 0;
    const maskCount = Number.isFinite(dataset.mask_count) ? dataset.mask_count : 0;
    const overlayCount = Number.isFinite(dataset.overlay_count) ? dataset.overlay_count : 0;
    const navigationCount = Number.isFinite(dataset.navigation_count) ? dataset.navigation_count : 0;
    const showInfer = imageCount > 0 && overlayCount < imageCount;
    const showNavigate = imageCount > 0 && navigationCount < imageCount;
    const showAutoLabel = annotationCount < imageCount;
    const showGenerateMasks = annotationCount > 0 && maskCount < annotationCount;
    return `
      <article class="dataset-card">
        <div class="dataset-card-head">
          <strong>${escapeHtml(dataset.name)}</strong>
        </div>
        <div class="dataset-metrics">
          <button type="button" class="metric-chip metric-action" data-dataset-open="${escapeHtml(dataset.name)}" data-dataset-kind="images">
            <span>图像</span>
            <strong>${escapeHtml(String(imageCount))}</strong>
          </button>
          <button type="button" class="metric-chip metric-action" data-dataset-open="${escapeHtml(dataset.name)}" data-dataset-kind="annotations">
            <span>标注</span>
            <strong>${escapeHtml(`${annotationCount}/${imageCount}`)}</strong>
          </button>
          <button type="button" class="metric-chip metric-action" data-dataset-open="${escapeHtml(dataset.name)}" data-dataset-kind="masks">
            <span>掩码</span>
            <strong>${escapeHtml(`${maskCount}/${imageCount}`)}</strong>
          </button>
          <button type="button" class="metric-chip metric-action" data-dataset-open="${escapeHtml(dataset.name)}" data-dataset-kind="overlays">
            <span>覆盖图</span>
            <strong>${escapeHtml(`${overlayCount}/${imageCount}`)}</strong>
          </button>
          <button type="button" class="metric-chip metric-action" data-dataset-open="${escapeHtml(dataset.name)}" data-dataset-kind="navigation">
            <span>导航</span>
            <strong>${escapeHtml(`${navigationCount}/${imageCount}`)}</strong>
          </button>
        </div>
        ${(showInfer || showNavigate || showAutoLabel || showGenerateMasks) ? `
        <div class="dataset-card-actions">
          ${showInfer ? `
          <button
            type="button"
            class="ghost-button"
            data-infer-dataset="${escapeHtml(dataset.name)}"
          >推理</button>` : ""}
          ${showNavigate ? `
          <button
            type="button"
            class="ghost-button"
            data-navigate-dataset="${escapeHtml(dataset.name)}"
          >导航</button>` : ""}
          ${showAutoLabel ? `
          <button
            type="button"
            class="dataset-autolabel-button ghost-button"
            data-autolabel-dataset="${escapeHtml(dataset.name)}"
          >打标</button>` : ""}
          ${showGenerateMasks ? `
          <button
            type="button"
            class="dataset-mask-button"
            data-mask-dataset="${escapeHtml(dataset.name)}"
          >掩码</button>` : ""}
        </div>` : ""}
      </article>
    `;
  }).join("");
}

export function renderTrainHint(dataset) {
  const el = document.getElementById("train-paths");
  if (!dataset) {
    el.innerHTML = "图像目录：-<br>掩码目录：-";
    return;
  }
  el.innerHTML = `图像目录：${escapeHtml(dataset.image_dir || "未找到")}<br>掩码目录：${escapeHtml(dataset.mask_dir || "未找到")}`;
}

export function renderInferencePanel() {
  const method = state.inferSelection.method || "verify-onnx";
  const datasetName = state.inferSelection.datasetName || state.datasets[0]?.name || "";
  const dataset = datasetName ? state.datasetMap[datasetName] : null;
  const images = dataset ? state.datasetImages[dataset.name] || [] : [];
  const selectedImage = images.find((item) => item.path === state.inferSelection.imagePath) || null;
  const configuredPath = method === "infer-trt"
    ? state.config?.inference?.engine_path || ""
    : state.config?.inference?.onnx_path || "";
  const modelOptions = listInferenceModels(state.artifacts, method, configuredPath);
  const modelKey = inferenceSelectionKey(method);
  const selectedModelPath = state.inferSelection[modelKey] || modelOptions[0]?.path || "";

  renderDatasetOptions("infer-dataset", datasetName);

  const methodSelect = document.getElementById("infer-method");
  const datasetSelect = document.getElementById("infer-dataset");
  const modelSelect = document.getElementById("infer-model");
  const hiddenInput = document.getElementById("infer-image-path");
  const summary = document.getElementById("infer-image-summary");
  const card = document.getElementById("infer-picked-card");
  const submit = document.getElementById("infer-submit");

  methodSelect.value = method;
  datasetSelect.value = dataset ? dataset.name : "";
  modelSelect.innerHTML = modelOptions.length
    ? modelOptions.map((item) => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.label)}</option>`).join("")
    : `<option value="">暂无可用 ${escapeHtml(inferenceModelFilename(method))}</option>`;
  modelSelect.disabled = !modelOptions.length;
  modelSelect.value = modelOptions.some((item) => item.path === selectedModelPath)
    ? selectedModelPath
    : (modelOptions[0]?.path || "");
  hiddenInput.value = state.inferSelection.imagePath || "";
  submit.textContent = method === "infer-trt" ? "执行 TensorRT 推理" : "执行 ONNX 推理";
  submit.disabled = !modelOptions.length;

  if (!dataset) {
    summary.textContent = "当前没有可用数据集。";
    card.className = "infer-picked-card empty";
    card.textContent = "请先准备可用数据集。";
    if (state.artifacts) {
      updateModelPathHints(state.artifacts);
    }
    return;
  }

  summary.textContent = images.length
    ? `${dataset.name} 共 ${images.length} 张图像，点击右侧按钮从网格中挑选。`
    : `${dataset.name} 当前没有可用图像。`;

  if (!selectedImage) {
    card.className = "infer-picked-card empty";
    card.textContent = images.length ? "当前还没有选中的图片。" : "当前数据集没有可选图片。";
    if (state.artifacts) {
      updateModelPathHints(state.artifacts);
    }
    return;
  }

  const label = imageLabel(selectedImage) || state.inferSelection.imageLabel || "已选图片";
  const url = fileUrl(selectedImage.path);
  card.className = "infer-picked-card";
  card.innerHTML = `
    <div class="infer-picked-preview">
      <img class="infer-picked-thumb" src="${url}" alt="${escapeHtml(label)}" loading="lazy">
      <div class="infer-picked-meta">
        <div>
          <strong>${escapeHtml(label)}</strong>
          <div class="meta">${escapeHtml(selectedImage.path)}</div>
        </div>
        <div class="infer-picked-actions">
          <button
            type="button"
            class="ghost-button"
            data-preview-src="${url}"
            data-preview-label="${escapeHtml(label)}"
          >放大查看</button>
          <button type="button" class="ghost-button" id="infer-repick-button">重新选择</button>
        </div>
      </div>
    </div>
  `;

  if (state.artifacts) {
    updateModelPathHints(state.artifacts);
  }
}

export function renderNavigationPanel() {
  const methodSelect = document.getElementById("navigation-method");
  const datasetSelect = document.getElementById("navigation-dataset");
  const modelSelect = document.getElementById("navigation-model");
  const hiddenInput = document.getElementById("navigation-image-path");
  const summary = document.getElementById("navigation-image-summary");
  const card = document.getElementById("navigation-picked-card");
  const submit = document.getElementById("navigation-submit");
  const batchButton = document.getElementById("navigation-batch-submit");
  const outputHint = document.getElementById("navigation-output-hint");

  if (
    !methodSelect
    || !datasetSelect
    || !modelSelect
    || !hiddenInput
    || !summary
    || !card
    || !submit
    || !batchButton
    || !outputHint
  ) {
    return;
  }

  const method = state.navigationSelection.method || "navigate-onnx";
  const datasetName = state.navigationSelection.datasetName || state.datasets[0]?.name || "";
  const dataset = datasetName ? state.datasetMap[datasetName] : null;
  const images = dataset ? state.datasetImages[dataset.name] || [] : [];
  const selectedImage = images.find((item) => item.path === state.navigationSelection.imagePath) || null;
  const configuredPath = method === "navigate-trt"
    ? state.config?.inference?.engine_path || ""
    : state.config?.inference?.onnx_path || "";
  const modelOptions = listInferenceModels(state.artifacts, method, configuredPath);
  const batchModelOptions = listInferenceModels(
    state.artifacts,
    "navigate-onnx",
    state.config?.inference?.onnx_path || "",
  );
  const modelKey = inferenceSelectionKey(method);
  const selectedModelPath = state.navigationSelection[modelKey] || modelOptions[0]?.path || "";

  renderDatasetOptions("navigation-dataset", datasetName);

  methodSelect.value = method;
  datasetSelect.value = dataset ? dataset.name : "";
  modelSelect.innerHTML = modelOptions.length
    ? modelOptions.map((item) => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.label)}</option>`).join("")
    : `<option value="">暂无可用 ${escapeHtml(inferenceModelFilename(method))}</option>`;
  modelSelect.disabled = !modelOptions.length;
  modelSelect.value = modelOptions.some((item) => item.path === selectedModelPath)
    ? selectedModelPath
    : (modelOptions[0]?.path || "");
  hiddenInput.value = state.navigationSelection.imagePath || "";
  submit.textContent = method === "navigate-trt" ? "执行 TensorRT 导航" : "执行 ONNX 导航";
  submit.disabled = !modelOptions.length;
  batchButton.disabled = !dataset || !batchModelOptions.length;

  if (!dataset) {
    summary.textContent = "当前没有可用数据集。";
    card.className = "infer-picked-card empty";
    card.textContent = "请先准备可用数据集。";
    outputHint.textContent = "导航输出目录：-";
    if (state.artifacts) {
      updateModelPathHints(state.artifacts, "navigation-model-path", method, state.navigationSelection[modelKey]);
    }
    return;
  }

  outputHint.textContent = `导航输出目录：${dataset.navigation_dir || "-"}`;
  summary.textContent = images.length
    ? `${dataset.name} 共 ${images.length} 张图像，可单图调试或批量生成导航结果。`
    : `${dataset.name} 当前没有可用图像。`;

  if (!selectedImage) {
    card.className = "infer-picked-card empty";
    card.textContent = images.length ? "当前还没有选中的图片。" : "当前数据集没有可选图片。";
    if (state.artifacts) {
      updateModelPathHints(state.artifacts, "navigation-model-path", method, state.navigationSelection[modelKey]);
    }
    return;
  }

  const label = imageLabel(selectedImage) || state.navigationSelection.imageLabel || "已选图片";
  const url = fileUrl(selectedImage.path);
  card.className = "infer-picked-card";
  card.innerHTML = `
    <div class="infer-picked-preview">
      <img class="infer-picked-thumb" src="${url}" alt="${escapeHtml(label)}" loading="lazy">
      <div class="infer-picked-meta">
        <div>
          <strong>${escapeHtml(label)}</strong>
          <div class="meta">${escapeHtml(selectedImage.path)}</div>
        </div>
        <div class="infer-picked-actions">
          <button
            type="button"
            class="ghost-button"
            data-preview-src="${url}"
            data-preview-label="${escapeHtml(label)}"
          >放大查看</button>
          <button type="button" class="ghost-button" id="navigation-repick-button">重新选择</button>
        </div>
      </div>
    </div>
  `;

  if (state.artifacts) {
    updateModelPathHints(state.artifacts, "navigation-model-path", method, state.navigationSelection[modelKey]);
  }
}

export function renderAnalysisPanel() {
  renderDatasetOptions("benchmark-dataset");

  const runs = state.analysisOptions.trainRuns || [];

  const benchmarkRunValue = setSelectOptions(
    "benchmark-run",
    runs,
    (item) => item.path,
    (item) => `${item.name}${item.is_latest ? " / 最新" : ""}${item.onnx_path ? "" : " / 无ONNX"}${item.trt_path ? "" : " / 无TRT"}`,
    {
      placeholderLabel: "使用当前默认配置 / 最近产物",
      placeholderValue: "",
      includePlaceholder: true,
    },
  );
  const historyValue = setSelectOptions(
    "figure-history-path",
    runs,
    (item) => item.history_path,
    (item) => `${item.name}${item.is_latest ? " / 最新" : ""}`,
    {
      placeholderLabel: "留空则读取最近一次训练 history.json",
      placeholderValue: "",
      includePlaceholder: true,
    },
  );

  const benchmarkDataset = state.datasetMap[document.getElementById("benchmark-dataset")?.value || ""] || null;
  const selectedBenchmarkRun = runs.find((item) => item.path === benchmarkRunValue) || null;
  const selectedHistory = runs.find((item) => item.history_path === historyValue) || null;

  const benchmarkHint = document.getElementById("benchmark-run-hint");
  if (benchmarkHint) {
    benchmarkHint.innerHTML = benchmarkDataset
      ? `测速图像目录：${escapeHtml(benchmarkDataset.image_dir || "-")}<br>训练产物：${escapeHtml(selectedBenchmarkRun?.path || "默认 / 最新")}`
      : "测速图像目录：-";
  }

  const figureHint = document.getElementById("figure-history-hint");
  if (figureHint) {
    figureHint.innerHTML = selectedHistory
      ? `history：${escapeHtml(selectedHistory.history_path)}<br>run：${escapeHtml(selectedHistory.path)}`
      : "history：留空时自动使用最近一次训练结果。";
  }

  const figureNavigationRootInput = document.getElementById("figure-navigation-root");
  if (figureNavigationRootInput && !figureNavigationRootInput.value) {
    figureNavigationRootInput.value = state.config?.dataset?.root_dir || "";
  }

  const defaultOnnxReady = listInferenceModels(
    state.artifacts,
    "verify-onnx",
    state.config?.inference?.onnx_path || "",
  ).length > 0;
  const defaultTrtReady = listInferenceModels(
    state.artifacts,
    "infer-trt",
    state.config?.inference?.engine_path || "",
  ).length > 0;

  document.getElementById("benchmark-backend-pytorch").disabled = false;
  document.getElementById("benchmark-backend-onnx").disabled = selectedBenchmarkRun
    ? !selectedBenchmarkRun.onnx_path
    : !defaultOnnxReady;
  document.getElementById("benchmark-backend-trt").disabled = selectedBenchmarkRun
    ? !selectedBenchmarkRun.trt_path
    : !defaultTrtReady;

}

export function renderArtifacts(payload) {
  document.getElementById("artifacts-root").textContent = payload.root_dir || "-";
  updateModelPathHints(payload);

  const container = document.getElementById("artifacts");
  if (!payload.runs.length) {
    container.innerHTML = '<div class="empty">还没有训练产物目录。先跑一次训练，页面会自动展示结果。</div>';
    return;
  }

  container.innerHTML = payload.runs.map((run) => {
    const files = run.files.map((file) => {
      const link = fileUrl(file.path, file.modified_at);
      const preview = file.is_image ? `<img class="thumb" src="${link}" alt="${escapeHtml(file.name)}">` : "";
      return `
        <div>
          <a href="${link}" target="_blank" rel="noreferrer">${escapeHtml(file.name)}</a>
          <div class="meta">大小：${formatSize(file.size)}，更新时间：${escapeHtml(file.modified_at)}</div>
          ${preview}
        </div>
      `;
    }).join("");

    return `
      <article class="run-card">
        <div class="run-head">
          <div>
            <strong>${escapeHtml(run.name)}</strong>
            <div class="meta">${escapeHtml(run.path)}</div>
          </div>
          ${run.is_latest ? '<span class="badge running">最新</span>' : ""}
        </div>
        <div class="run-files">${files || '<span class="meta">目录中暂无文件。</span>'}</div>
      </article>
    `;
  }).join("");
}

export function updateModelPathHints(
  payload,
  elementId = "infer-model-path",
  method = state.inferSelection.method || "verify-onnx",
  selectedPath = state.inferSelection[inferenceSelectionKey(method)],
) {
  const hint = document.getElementById(elementId);
  const configuredPath = method === "infer-trt" || method === "navigate-trt"
    ? state.config?.inference?.engine_path || ""
    : state.config?.inference?.onnx_path || "";
  const models = listInferenceModels(payload, method, configuredPath);
  const selectedModel = models.find((item) => item.path === selectedPath) || models[0] || null;

  if (!hint) return;

  if (!selectedModel) {
    hint.textContent = method === "infer-trt" || method === "navigate-trt"
      ? "当前 TensorRT 模型：暂无，请先生成引擎。"
      : "当前 ONNX 模型：暂无，请先训练或导出。";
    return;
  }

  const methodLabel = method === "infer-trt" || method === "navigate-trt" ? "TensorRT" : "ONNX";
  hint.innerHTML = `当前 ${methodLabel} 模型：${escapeHtml(selectedModel.path)}<br>来源：${escapeHtml(selectedModel.runName)}<br>位置：${escapeHtml(selectedModel.runPath)}`;
}

export function renderTaskSummary() {
  const total = state.jobs.length;
  const running = state.jobs.filter((job) => job.status === "queued" || job.status === "running").length;
  document.getElementById("task-running-badge").textContent = String(running);
  document.getElementById("task-running-badge").hidden = running <= 0;
  document.getElementById("task-summary").textContent = running ? `${running} 个进行中` : "当前无运行任务";
  document.getElementById("task-count").textContent = `${total} 个任务`;
}

function jobProgressValue(job) {
  if (job.status === "succeeded") return 100;
  const percent = Number(job.progress_percent);
  if (!Number.isFinite(percent)) return 0;
  return Math.max(0, Math.min(100, percent));
}

function jobProgressLabel(job) {
  const completed = Number(job.progress_completed);
  const total = Number(job.progress_total);
  const text = job.progress_text ? ` · ${job.progress_text}` : "";
  if (Number.isFinite(total) && total > 0) {
    return `${completed}/${total}${text}`;
  }
  if (job.progress_text) {
    return job.progress_text;
  }
  return statusText(job.status);
}

export function renderJobList() {
  renderTaskSummary();

  const container = document.getElementById("job-list");
  if (!state.jobs.length) {
    container.innerHTML = '<div class="empty">还没有后台任务，提交一个试试看。</div>';
    return;
  }

  container.innerHTML = state.jobs.map((job) => `
    <article class="job-item ${job.job_id === state.selectedJobId ? "active" : ""}" data-job-id="${escapeHtml(job.job_id)}">
      <div class="job-item-head">
        <div>
          <div class="job-item-title">${escapeHtml(kindText(job.kind))}</div>
          <div class="meta">任务号：${escapeHtml(job.job_id)}</div>
        </div>
        <span class="badge ${job.status}">${statusText(job.status)}</span>
      </div>
      <div class="meta job-item-meta-line">
        <span>创建：${escapeHtml(job.created_at)}</span>
        ${job.finished_at ? `<span>结束：${escapeHtml(job.finished_at)}</span>` : ""}
      </div>
      <div class="job-progress">
        <div class="job-progress-bar">
          <span style="width: ${jobProgressValue(job)}%;"></span>
        </div>
        <div class="meta">${escapeHtml(jobProgressLabel(job))}</div>
      </div>
    </article>
  `).join("");
}

export function renderJobDetail() {
  const container = document.getElementById("job-detail");
  const hint = document.getElementById("task-detail-hint");
  const title = document.getElementById("detail-modal-title");
  const job = state.selectedJob;

  if (!job) {
    title.textContent = "点击任务查看详情";
    hint.textContent = "图片支持继续放大查看。";
    container.className = "job-detail empty";
    container.textContent = "点击右上角任务中心中的任务查看详情。";
    container.dataset.signature = "";
    return;
  }

  const signature = JSON.stringify({
    job_id: job.job_id,
    kind: job.kind,
    status: job.status,
    created_at: job.created_at,
    started_at: job.started_at,
    finished_at: job.finished_at,
    progress_completed: job.progress_completed,
    progress_total: job.progress_total,
    progress_percent: job.progress_percent,
    progress_text: job.progress_text,
    payload: job.payload || null,
    result: job.result || null,
    error: job.error || null,
  });

  title.textContent = kindText(job.kind);
  hint.textContent = `${statusText(job.status)} / 任务号 ${job.job_id}`;

  if (container.dataset.signature === signature) {
    container.className = "job-detail";
    return;
  }

  container.className = "job-detail";
  container.innerHTML = `
    <article class="job-detail-card">
      <div class="job-detail-head">
        <div>
          <strong>${escapeHtml(kindText(job.kind))}</strong>
          <div class="meta">任务号：${escapeHtml(job.job_id)}</div>
        </div>
        <span class="badge ${job.status}">${statusText(job.status)}</span>
      </div>

      <section class="job-detail-section">
        <h3>任务时间</h3>
        <div class="meta">创建时间：${escapeHtml(job.created_at)}</div>
        ${job.started_at ? `<div class="meta">开始时间：${escapeHtml(job.started_at)}</div>` : ""}
        ${job.finished_at ? `<div class="meta">结束时间：${escapeHtml(job.finished_at)}</div>` : ""}
      </section>

      <section class="job-detail-section">
        <h3>任务进度</h3>
        <div class="job-progress">
          <div class="job-progress-bar">
            <span style="width: ${jobProgressValue(job)}%;"></span>
          </div>
          <div class="meta">${escapeHtml(jobProgressLabel(job))}</div>
        </div>
      </section>

      ${renderCommonResult(job)}
      ${renderJobPayload(job)}
      ${renderJobError(job)}
    </article>
  `;
  container.dataset.signature = signature;
}

export function renderTaskCenter() {
  renderJobList();
}

export function renderDetailModalVisibility() {
  const panel = document.getElementById("detail-modal");
  panel.classList.toggle("open", state.detailModalOpen);
  panel.setAttribute("aria-hidden", state.detailModalOpen ? "false" : "true");
}

export function renderTaskCenterVisibility() {
  const panel = document.getElementById("task-center-modal");
  panel.classList.toggle("open", state.taskCenterOpen);
  panel.setAttribute("aria-hidden", state.taskCenterOpen ? "false" : "true");
}

export function renderDatasetGallery() {
  const title = document.getElementById("dataset-gallery-title");
  const hint = document.getElementById("dataset-gallery-hint");
  const grid = document.getElementById("dataset-gallery-grid");
  const dataset = state.selectedDatasetName ? state.datasetMap[state.selectedDatasetName] : null;
  const pickerMode = state.galleryMode === "infer" || state.galleryMode === "navigation";
  const kind = pickerMode ? "images" : (state.galleryKind || "images");
  const files = dataset
    ? (pickerMode ? (state.datasetImages[dataset.name] || []) : (state.datasetFiles[dataset.name]?.[kind] || []))
    : [];
  const kindLabelMap = {
    images: "图像",
    annotations: "标注",
    masks: "掩码",
    overlays: "覆盖图",
    navigation: "导航结果",
  };
  const kindLabel = kindLabelMap[kind] || "文件";

  if (!dataset) {
    title.textContent = "数据集图片";
    hint.textContent = pickerMode ? "先选择数据集，再从网格里挑选图片。" : "点击图像数量块查看图片。";
    grid.className = "dataset-gallery-grid empty";
    grid.textContent = "当前未选择数据集。";
    return;
  }

  title.textContent = pickerMode ? `${dataset.name} · 选择输入图像` : `${dataset.name} · ${kindLabel}`;
  hint.textContent = pickerMode
    ? `${dataset.image_count} 张图像`
    : `${files.length} 个${kindLabel}文件`;

  if (!files.length) {
    grid.className = "dataset-gallery-grid empty";
    grid.textContent = `当前数据集没有可展示的${kindLabel}。`;
    return;
  }

  grid.className = "dataset-gallery-grid";
  grid.innerHTML = files.map((file) => {
    const url = fileUrl(file.path);
    const selectedPath = state.galleryMode === "navigation"
      ? state.navigationSelection.imagePath
      : state.inferSelection.imagePath;
    const isSelected = pickerMode && file.path === selectedPath;
    const label = imageLabel(file);
    const attrs = pickerMode
      ? `data-image-pick="${escapeHtml(file.path)}" data-image-label="${escapeHtml(label)}"`
      : file.is_image
        ? `data-preview-src="${url}" data-preview-label="${escapeHtml(label)}"`
        : `data-file-open="${url}"`;

    return `
      <button
        type="button"
        class="gallery-tile ${isSelected ? "selected" : ""} ${file.is_image ? "" : "gallery-tile-file"}"
        ${attrs}
      >
        ${file.is_image ? `<img class="gallery-thumb" src="${url}" alt="${escapeHtml(label)}" loading="lazy">` : '<div class="gallery-file-icon">JSON</div>'}
        <span class="gallery-name">${escapeHtml(file.name)}</span>
      </button>
    `;
  }).join("");
}

export function renderDatasetGalleryVisibility() {
  const panel = document.getElementById("dataset-gallery-modal");
  panel.classList.toggle("open", state.datasetGalleryOpen);
  panel.setAttribute("aria-hidden", state.datasetGalleryOpen ? "false" : "true");
}
