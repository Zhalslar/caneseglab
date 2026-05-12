import {
  fetchAnalysisOptions,
  fetchArtifacts,
  fetchConfig,
  fetchDatasetFiles,
  fetchDatasetImages,
  fetchDatasets,
  fetchJobDetail,
  fetchJobs,
  openJobEvents,
  submitJob as postJob,
} from "./api.js?v=20260325c";
import {
  populateConfig,
  renderAnalysisPanel,
  renderArtifacts,
  renderDatasetGallery,
  renderDatasetGalleryVisibility,
  renderDatasetOptions,
  renderDatasetOverview,
  renderDetailModalVisibility,
  renderInferencePanel,
  renderJobDetail,
  renderNavigationPanel,
  renderPageTabs,
  renderTaskCenter,
  renderTaskCenterVisibility,
  renderTrainHint,
} from "./render.js?v=20260511a";
import {
  datasetByName,
  selectDataset,
  selectJob,
  setActivePage,
  setAnalysisOptions,
  setArtifacts,
  setDatasets,
  setDatasetGalleryOpen,
  setDetailModalOpen,
  setGalleryMode,
  setGalleryKind,
  setInferSelection,
  setJobs,
  setNavigationSelection,
  setTaskCenterOpen,
  state,
} from "./state.js?v=20260325c";
import {
  listInferenceModels,
  readForm,
  setValue,
  showToast,
  toNumber,
} from "./utils.js?v=20260325c";

let reconnectJobEventsHandle = 0;

function inferenceSelectionKey(method) {
  return method === "infer-trt" || method === "navigate-trt" ? "enginePath" : "onnxPath";
}

function configuredInferenceModelPath(method) {
  if (method === "infer-trt" || method === "navigate-trt") {
    return state.config?.inference?.engine_path || "";
  }
  return state.config?.inference?.onnx_path || "";
}

function syncInferenceModelSelection(method = state.inferSelection.method || "verify-onnx") {
  const key = inferenceSelectionKey(method);
  const models = listInferenceModels(state.artifacts, method, configuredInferenceModelPath(method));
  let nextPath = state.inferSelection[key] || "";
  if (!models.length) {
    nextPath = "";
  } else if (!models.some((item) => item.path === nextPath)) {
    nextPath = models[0].path;
  }
  setInferSelection({ [key]: nextPath });
  return nextPath;
}

function syncNavigationModelSelection(method = state.navigationSelection.method || "navigate-onnx") {
  const key = inferenceSelectionKey(method);
  const models = listInferenceModels(state.artifacts, method, configuredInferenceModelPath(method));
  let nextPath = state.navigationSelection[key] || "";
  if (!models.length) {
    nextPath = "";
  } else if (!models.some((item) => item.path === nextPath)) {
    nextPath = models[0].path;
  }
  setNavigationSelection({ [key]: nextPath });
  return nextPath;
}

function applyConfig(config) {
  state.config = config;
  populateConfig(config);

  setValue('#train-form [name="model_name"]', config.train.model_name);
  setValue('#train-form [name="backbone"]', config.train.backbone);
  setValue('#train-form [name="epochs"]', config.train.epochs);
  setValue('#train-form [name="learning_rate"]', config.train.learning_rate);
  setValue('#train-form [name="train_batch_size"]', config.train.train_batch_size);
  setValue('#train-form [name="valid_batch_size"]', config.train.valid_batch_size);
  setValue('#train-form [name="device"]', config.train.device);
  setValue('#train-form [name="export_onnx"]', config.train.export_onnx);
  setInferSelection({
    method: "verify-onnx",
    onnxPath: config.inference.onnx_path || "",
    enginePath: config.inference.engine_path || "",
  });
  setNavigationSelection({
    method: "navigate-onnx",
    onnxPath: config.inference.onnx_path || "",
    enginePath: config.inference.engine_path || "",
  });
}

function syncBodyModalState() {
  const taskCenterOpen = document.getElementById("task-center-modal")?.classList.contains("open") || false;
  const galleryOpen = document.getElementById("dataset-gallery-modal")?.classList.contains("open") || false;
  const detailOpen = document.getElementById("detail-modal")?.classList.contains("open") || false;
  const previewOpen = document.getElementById("preview-modal")?.classList.contains("open") || false;
  document.body.classList.toggle("modal-open", taskCenterOpen || galleryOpen || detailOpen || previewOpen);
}

function onId(id, eventName, handler) {
  const element = document.getElementById(id);
  if (!element) return null;
  element.addEventListener(eventName, handler);
  return element;
}

function invalidateDatasetFiles(datasetName = "") {
  if (!datasetName) {
    state.datasetFiles = {};
    return;
  }
  delete state.datasetFiles[datasetName];
}

function renderJobPanels() {
  renderTaskCenter();
  if (state.detailModalOpen) {
    renderJobDetail();
  }
}

async function refreshOpenDatasetGallery() {
  if (!state.datasetGalleryOpen || !state.selectedDatasetName) return;

  if (state.galleryMode === "infer" || state.galleryMode === "navigation") {
    await ensureDatasetImages(state.selectedDatasetName);
  } else {
    await ensureDatasetFiles(state.selectedDatasetName, state.galleryKind || "images");
  }
  renderDatasetGallery();
}

async function syncJobEffects(nextJobs, previousStatusMap) {
  let needsDatasetRefresh = false;
  let needsArtifactRefresh = false;
  let needsAnalysisRefresh = false;

  for (const job of nextJobs) {
    if (job.status !== "succeeded") continue;
    if (previousStatusMap.get(job.job_id) === "succeeded") continue;

    if (job.kind === "mask" || job.kind === "auto-label" || job.kind === "infer-dataset") {
      needsDatasetRefresh = true;
    }
    if (job.kind === "navigate-onnx" || job.kind === "navigate-trt" || job.kind === "navigate-dataset") {
      needsDatasetRefresh = true;
    }
    if (job.kind === "train" || job.kind === "export-onnx" || job.kind === "export-trt") {
      needsArtifactRefresh = true;
      needsAnalysisRefresh = true;
    }
  }

  if (needsDatasetRefresh) {
    invalidateDatasetFiles();
    await refreshDatasets();
    await refreshOpenDatasetGallery();
  }
  if (needsArtifactRefresh) {
    await refreshArtifacts();
  }
  if (needsAnalysisRefresh) {
    await refreshAnalysisOptions();
  }
}

function applyJobSnapshot(jobs, runEffects = true) {
  const previousStatusMap = new Map(state.jobs.map((job) => [job.job_id, job.status]));
  setJobs(jobs || []);

  if (state.selectedJobId) {
    const selectedFromList = state.jobs.find((item) => item.job_id === state.selectedJobId) || null;
    if (selectedFromList) {
      selectJob(selectedFromList);
    }
  }

  renderJobPanels();
  if (runEffects) {
    void syncJobEffects(state.jobs, previousStatusMap);
  }
}

function connectJobEvents() {
  window.clearTimeout(reconnectJobEventsHandle);
  state.jobEventSource?.close();

  const source = openJobEvents();
  state.jobEventSource = source;

  source.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data || "{}");
      applyJobSnapshot(payload.jobs || []);
    } catch (error) {
      console.error(error);
    }
  };

  source.onerror = () => {
    source.close();
    if (state.jobEventSource === source) {
      state.jobEventSource = null;
    }
    reconnectJobEventsHandle = window.setTimeout(connectJobEvents, 1500);
  };
}

export async function refreshDatasets() {
  const payload = await fetchDatasets();
  setDatasets(payload.datasets || []);
  invalidateDatasetFiles();

  renderDatasetOverview(payload.root_dir || "-", state.datasets);
  renderDatasetOptions("train-dataset");
  renderAnalysisPanel();

  syncTrainDataset();
  await syncInferenceDataset(state.inferSelection.datasetName || state.datasets[0]?.name || "");
  await syncNavigationDataset(state.navigationSelection.datasetName || state.datasets[0]?.name || "");
}

export async function ensureDatasetImages(datasetName) {
  if (!datasetName) return [];
  if (state.datasetImages[datasetName]) {
    return state.datasetImages[datasetName];
  }
  const payload = await fetchDatasetImages(datasetName);
  state.datasetImages[datasetName] = payload.images || [];
  return state.datasetImages[datasetName];
}

export async function ensureDatasetFiles(datasetName, kind) {
  if (!datasetName || !kind) return [];
  state.datasetFiles[datasetName] ||= {};
  if (state.datasetFiles[datasetName][kind]) {
    return state.datasetFiles[datasetName][kind];
  }
  const payload = await fetchDatasetFiles(datasetName, kind);
  state.datasetFiles[datasetName][kind] = payload.files || [];
  return state.datasetFiles[datasetName][kind];
}

export function syncTrainDataset() {
  renderTrainHint(datasetByName(document.getElementById("train-dataset").value));
}

export async function syncInferenceDataset(datasetName) {
  const nextDatasetName = datasetName || state.datasets[0]?.name || "";

  if (!nextDatasetName) {
    setInferSelection({
      datasetName: null,
      imagePath: "",
      imageLabel: "",
    });
    renderInferencePanel();
    return;
  }

  const images = await ensureDatasetImages(nextDatasetName);
  const defaultImagePath = state.config?.inference?.image_path || "";

  let imagePath = state.inferSelection.imagePath || "";
  if (!images.length) {
    imagePath = "";
  } else if (!images.some((item) => item.path === imagePath)) {
    imagePath = images.some((item) => item.path === defaultImagePath)
      ? defaultImagePath
      : images[0].path;
  }

  const image = images.find((item) => item.path === imagePath) || null;
  setInferSelection({
    datasetName: nextDatasetName,
    imagePath,
    imageLabel: image ? (image.relative_path || image.name) : "",
  });
  renderInferencePanel();
}

export async function syncNavigationDataset(datasetName) {
  const nextDatasetName = datasetName || state.datasets[0]?.name || "";

  if (!nextDatasetName) {
    setNavigationSelection({
      datasetName: null,
      imagePath: "",
      imageLabel: "",
    });
    renderNavigationPanel();
    return;
  }

  const images = await ensureDatasetImages(nextDatasetName);
  const defaultImagePath = state.config?.inference?.image_path || "";

  let imagePath = state.navigationSelection.imagePath || "";
  if (!images.length) {
    imagePath = "";
  } else if (!images.some((item) => item.path === imagePath)) {
    imagePath = images.some((item) => item.path === defaultImagePath)
      ? defaultImagePath
      : images[0].path;
  }

  const image = images.find((item) => item.path === imagePath) || null;
  setNavigationSelection({
    datasetName: nextDatasetName,
    imagePath,
    imageLabel: image ? (image.relative_path || image.name) : "",
  });
  renderNavigationPanel();
}

export async function refreshJobs() {
  const payload = await fetchJobs();
  applyJobSnapshot(payload.jobs || [], false);
}

export async function refreshArtifacts() {
  const payload = await fetchArtifacts();
  setArtifacts(payload);
  syncInferenceModelSelection();
  syncNavigationModelSelection();
  renderArtifacts(payload);
  renderInferencePanel();
  renderNavigationPanel();
  renderAnalysisPanel();
}

export async function refreshAnalysisOptions() {
  const payload = await fetchAnalysisOptions();
  setAnalysisOptions(payload);
  renderAnalysisPanel();
}

export async function selectJobById(jobId) {
  if (!jobId) return;
  const existing = state.jobs.find((item) => item.job_id === jobId) || null;
  if (existing) {
    selectJob(existing);
    renderJobPanels();
    return;
  }
  const detail = await fetchJobDetail(jobId);
  selectJob(detail);
  renderJobPanels();
}

export function openDetailModal() {
  setDetailModalOpen(true);
  renderDetailModalVisibility();
  syncBodyModalState();
}

export function closeDetailModal() {
  setDetailModalOpen(false);
  renderDetailModalVisibility();
  syncBodyModalState();
}

export async function openDatasetGallery(datasetName, mode = "browse", kind = "images") {
  selectDataset(datasetName);
  setGalleryMode(mode);
  setGalleryKind(kind);
  if (mode === "infer" || mode === "navigation") {
    await ensureDatasetImages(datasetName);
  } else {
    await ensureDatasetFiles(datasetName, kind);
  }
  renderDatasetGallery();
  setDatasetGalleryOpen(true);
  renderDatasetGalleryVisibility();
  syncBodyModalState();
}

export function closeDatasetGallery() {
  setDatasetGalleryOpen(false);
  setGalleryMode("browse");
  setGalleryKind("images");
  renderDatasetGalleryVisibility();
  syncBodyModalState();
}

export function openTaskCenter() {
  setTaskCenterOpen(true);
  renderTaskCenterVisibility();
  syncBodyModalState();
}

export function closeTaskCenter() {
  setTaskCenterOpen(false);
  renderTaskCenterVisibility();
  syncBodyModalState();
}

export function openPreviewModal(src, label = "图片预览") {
  const modal = document.getElementById("preview-modal");
  const image = document.getElementById("preview-image");
  const caption = document.getElementById("preview-caption");

  image.src = src;
  image.alt = label;
  caption.textContent = label;
  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
  syncBodyModalState();
}

export function closePreviewModal() {
  const modal = document.getElementById("preview-modal");
  const image = document.getElementById("preview-image");

  modal.classList.remove("open");
  modal.setAttribute("aria-hidden", "true");
  image.removeAttribute("src");
  image.alt = "";
  syncBodyModalState();
}

export function setPage(page) {
  const changed = state.activePage !== page;
  setActivePage(page);
  renderPageTabs();
  if (page === "data" && changed) {
    void refreshDatasets();
  }
}

async function submitJob(url, payload, successText, page = "", options = {}) {
  const job = await postJob(url, payload);
  showToast(`${successText}，任务号：${job.job_id}`);
  setJobs([job, ...state.jobs.filter((item) => item.job_id !== job.job_id)]);
  selectJob(job);
  renderJobPanels();
  if (page) {
    setPage(page);
  }
  if (options.autoOpenDetail) {
    closeTaskCenter();
    openDetailModal();
  }
}

async function submitMaskForDataset(datasetName) {
  const dataset = datasetByName(datasetName);
  const annotationDir = dataset?.annotation_dir || (dataset?.path ? `${dataset.path}\\annotations` : "");
  if (!annotationDir) {
    throw new Error("当前数据集没有可用标注目录。");
  }
  await submitJob(
    "/api/jobs/mask",
    {
      input_dir: annotationDir,
      output_dir: null,
    },
    `已提交 ${dataset.name} 的掩码任务`,
    "data",
  );
}

async function submitAutoLabelForDataset(datasetName) {
  const dataset = datasetByName(datasetName);
  const imageDir = dataset?.image_dir || (dataset?.path ? `${dataset.path}\\images` : "");
  const annotationDir = dataset?.annotation_dir || (dataset?.path ? `${dataset.path}\\annotations` : "");
  const overlayDir = dataset?.overlay_dir || (dataset?.path ? `${dataset.path}\\overlays` : "");
  if (!imageDir || !annotationDir) {
    throw new Error("当前数据集缺少自动打标所需目录。");
  }
  await submitJob(
    "/api/jobs/auto-label",
    {
      image_dir: imageDir,
      annotation_dir: annotationDir,
      overlay_dir: overlayDir,
    },
    `已提交 ${dataset.name} 的打标任务`,
    "data",
  );
}

async function submitInferForDataset(datasetName) {
  const dataset = datasetByName(datasetName);
  const imageDir = dataset?.image_dir || (dataset?.path ? `${dataset.path}\\images` : "");
  const overlayDir = dataset?.overlay_dir || (dataset?.path ? `${dataset.path}\\overlays` : "");
  if (!imageDir || !overlayDir) {
    throw new Error("当前数据集缺少批量推理所需目录。");
  }
  await submitJob(
    "/api/jobs/infer-dataset",
    {
      image_dir: imageDir,
      overlay_dir: overlayDir,
    },
    `已提交 ${dataset.name} 的推理任务`,
    "data",
  );
}

async function submitNavigateForDataset(datasetName) {
  const dataset = datasetByName(datasetName);
  const imageDir = dataset?.image_dir || (dataset?.path ? `${dataset.path}\\images` : "");
  const onnxModels = listInferenceModels(
    state.artifacts,
    "navigate-onnx",
    state.config?.inference?.onnx_path || "",
  );
  const onnxPath = state.navigationSelection.onnxPath
    || state.inferSelection.onnxPath
    || state.config?.inference?.onnx_path
    || onnxModels[0]?.path
    || null;
  if (!imageDir) {
    throw new Error("当前数据集缺少批量导航所需目录。");
  }
  if (!onnxPath) {
    throw new Error("当前没有可用 ONNX 模型。");
  }
  await submitJob(
    "/api/jobs/navigate-dataset",
    {
      image_dir: imageDir,
      config: {
        inference: {
          onnx_path: onnxPath,
        },
      },
    },
    `已提交 ${dataset.name} 的导航任务`,
    "navigation",
  );
}

async function handleTrainSubmit(event) {
  event.preventDefault();
  const values = readForm(event.currentTarget);
  const dataset = datasetByName(values.dataset_name);
  if (!dataset?.image_dir || !dataset?.mask_dir) {
    throw new Error("当前数据集缺少 images 或 masks 目录。");
  }
  await submitJob(
    "/api/jobs/train",
    {
      image_dir: dataset.image_dir,
      mask_dir: dataset.mask_dir,
      config: {
        train: {
          model_name: values.model_name,
          backbone: values.backbone,
          epochs: toNumber(values.epochs),
          learning_rate: toNumber(values.learning_rate),
          train_batch_size: toNumber(values.train_batch_size),
          valid_batch_size: toNumber(values.valid_batch_size),
          device: values.device,
          export_onnx: values.export_onnx,
        },
      },
    },
    "训练任务已提交",
    "train",
  );
}

async function handleExportSubmit(event) {
  event.preventDefault();
  const values = readForm(event.currentTarget);
  await submitJob(
    "/api/jobs/export-onnx",
    {
      output_path: values.output_path || null,
    },
    "导出任务已提交",
    "train",
  );
}

async function handleExportTrtSubmit(event) {
  event.preventDefault();
  const values = readForm(event.currentTarget);
  await submitJob(
    "/api/jobs/export-trt",
    {
      onnx_path: values.onnx_path || null,
      output_path: values.output_path || null,
    },
    "TensorRT 导出任务已提交",
    "train",
  );
}

async function handleInferSubmit(event) {
  event.preventDefault();
  const values = readForm(event.currentTarget);
  if (!values.image_path) {
    throw new Error("请先从图片网格中选择一张图片。");
  }
  if (!values.model_path) {
    throw new Error("当前没有可用模型，请先训练、导出或生成对应模型文件。");
  }

  const isTensorRt = values.method === "infer-trt";
  await submitJob(
    isTensorRt ? "/api/jobs/infer-trt" : "/api/jobs/verify-onnx",
    {
      image_path: values.image_path,
      config: {
        inference: isTensorRt
          ? { engine_path: values.model_path }
          : { onnx_path: values.model_path },
      },
    },
    isTensorRt ? "TensorRT 推理任务已提交" : "ONNX 推理任务已提交",
    "infer",
    { autoOpenDetail: true },
  );
}

async function handleNavigationSubmit(event) {
  event.preventDefault();
  const values = readForm(event.currentTarget);
  if (!values.image_path) {
    throw new Error("请先从图片网格中选择一张图片。");
  }
  if (!values.model_path) {
    throw new Error("当前没有可用模型，请先训练、导出或生成对应模型文件。");
  }

  const isTensorRt = values.method === "navigate-trt";
  await submitJob(
    isTensorRt ? "/api/jobs/navigate-trt" : "/api/jobs/navigate-onnx",
    {
      image_path: values.image_path,
      config: {
        inference: isTensorRt
          ? { engine_path: values.model_path }
          : { onnx_path: values.model_path },
      },
    },
    isTensorRt ? "TensorRT 导航任务已提交" : "ONNX 导航任务已提交",
    "navigation",
    { autoOpenDetail: true },
  );
}

async function handleNavigationBatchSubmit(event) {
  event.preventDefault();
  const datasetName = state.navigationSelection.datasetName || document.getElementById("navigation-dataset")?.value || "";
  const onnxModels = listInferenceModels(
    state.artifacts,
    "navigate-onnx",
    state.config?.inference?.onnx_path || "",
  );
  const hasOnnxModel = Boolean(
    state.navigationSelection.onnxPath
      || state.inferSelection.onnxPath
      || state.config?.inference?.onnx_path
      || onnxModels[0]?.path,
  );
  if (!datasetName) {
    throw new Error("请先选择数据集。");
  }
  if (!hasOnnxModel) {
    throw new Error("当前没有可用 ONNX 模型。");
  }
  await submitNavigateForDataset(datasetName);
}

function selectedTrainRun(runDir) {
  return state.analysisOptions.trainRuns.find((item) => item.path === runDir) || null;
}

async function handleBenchmarkSubmit(event) {
  event.preventDefault();
  const values = readForm(event.currentTarget);
  const dataset = datasetByName(values.dataset_name);
  if (!dataset?.image_dir) {
    throw new Error("当前数据集缺少测速所需图像目录。");
  }

  const backends = [];
  if (values.backend_pytorch && !document.getElementById("benchmark-backend-pytorch")?.disabled) backends.push("pytorch");
  if (values.backend_onnx && !document.getElementById("benchmark-backend-onnx")?.disabled) backends.push("onnx");
  if (values.backend_trt && !document.getElementById("benchmark-backend-trt")?.disabled) backends.push("tensorrt");
  if (!backends.length) {
    throw new Error("请至少勾选一个测速后端。");
  }

  const run = selectedTrainRun(values.run_dir);
  const config = {};
  if (run) {
    config.train = { output_dir: run.path };
    config.inference = {
      onnx_path: run.onnx_path || undefined,
      engine_path: run.trt_path || undefined,
    };
  }

  await submitJob(
    "/api/jobs/benchmark",
    {
      image_dir: dataset.image_dir,
      output_dir: values.output_dir || null,
      warmup_runs: toNumber(values.warmup_runs),
      timed_runs: toNumber(values.timed_runs),
      max_images: toNumber(values.max_images),
      backends,
      config,
    },
    "测速任务已提交",
    "analysis",
  );
}

async function handleExportPaperFiguresSubmit(event) {
  event.preventDefault();
  const values = readForm(event.currentTarget);
  await submitJob(
    "/api/jobs/export-paper-figures",
    {
      history_path: values.history_path || null,
      navigation_root: values.navigation_root || null,
      output_dir: values.output_dir || null,
    },
    "论文图导出任务已提交",
    "analysis",
    { autoOpenDetail: true },
  );
}

async function handleAuditNavigationSubmit(event) {
  event.preventDefault();
  const values = readForm(event.currentTarget);
  await submitJob(
    "/api/jobs/audit-navigation",
    {
      navigation_root: values.navigation_root || null,
      output_dir: values.output_dir || null,
    },
    "导航审计任务已提交",
    "analysis",
    { autoOpenDetail: true },
  );
}

async function wrapSubmit(handler, event) {
  try {
    await handler(event);
  } catch (error) {
    showToast(error.message || "提交失败");
  }
}

export function bindEvents() {
  onId("train-form", "submit", (event) => void wrapSubmit(handleTrainSubmit, event));
  onId("export-form", "submit", (event) => void wrapSubmit(handleExportSubmit, event));
  onId("export-trt-form", "submit", (event) => void wrapSubmit(handleExportTrtSubmit, event));
  onId("infer-form", "submit", (event) => void wrapSubmit(handleInferSubmit, event));
  onId("navigation-form", "submit", (event) => void wrapSubmit(handleNavigationSubmit, event));
  onId("navigation-batch-form", "submit", (event) => void wrapSubmit(handleNavigationBatchSubmit, event));
  onId("benchmark-form", "submit", (event) => void wrapSubmit(handleBenchmarkSubmit, event));
  onId("paper-figures-form", "submit", (event) => void wrapSubmit(handleExportPaperFiguresSubmit, event));
  onId("navigation-audit-form", "submit", (event) => void wrapSubmit(handleAuditNavigationSubmit, event));

  onId("train-dataset", "change", syncTrainDataset);
  onId("benchmark-dataset", "change", renderAnalysisPanel);
  onId("benchmark-run", "change", renderAnalysisPanel);
  onId("figure-history-path", "change", renderAnalysisPanel);
  onId("infer-dataset", "change", (event) => {
    setInferSelection({ datasetName: event.currentTarget.value });
    void syncInferenceDataset(event.currentTarget.value);
  });
  onId("infer-method", "change", (event) => {
    const method = event.currentTarget.value;
    setInferSelection({ method });
    syncInferenceModelSelection(method);
    renderInferencePanel();
  });
  onId("infer-model", "change", (event) => {
    const method = state.inferSelection.method || "verify-onnx";
    setInferSelection({ [inferenceSelectionKey(method)]: event.currentTarget.value });
    renderInferencePanel();
  });
  onId("infer-image-picker", "click", () => {
    const inferDataset = document.getElementById("infer-dataset");
    const datasetName = state.inferSelection.datasetName || inferDataset?.value || "";
    if (!datasetName) {
      showToast("请先选择数据集。");
      return;
    }
    void openDatasetGallery(datasetName, "infer");
  });
  onId("navigation-dataset", "change", (event) => {
    setNavigationSelection({ datasetName: event.currentTarget.value });
    void syncNavigationDataset(event.currentTarget.value);
  });
  onId("navigation-method", "change", (event) => {
    const method = event.currentTarget.value;
    setNavigationSelection({ method });
    syncNavigationModelSelection(method);
    renderNavigationPanel();
  });
  onId("navigation-model", "change", (event) => {
    const method = state.navigationSelection.method || "navigate-onnx";
    setNavigationSelection({ [inferenceSelectionKey(method)]: event.currentTarget.value });
    renderNavigationPanel();
  });
  onId("navigation-image-picker", "click", () => {
    const navDataset = document.getElementById("navigation-dataset");
    const datasetName = state.navigationSelection.datasetName || navDataset?.value || "";
    if (!datasetName) {
      showToast("请先选择数据集。");
      return;
    }
    void openDatasetGallery(datasetName, "navigation");
  });
  onId("task-toggle", "click", openTaskCenter);
  onId("task-center-close", "click", closeTaskCenter);
  onId("task-center-backdrop", "click", closeTaskCenter);

  onId("dataset-cards", "click", (event) => {
    const inferTarget = event.target.closest("[data-infer-dataset]");
    if (inferTarget) {
      event.stopPropagation();
      void wrapSubmit(() => submitInferForDataset(inferTarget.dataset.inferDataset), event);
      return;
    }

    const navigateTarget = event.target.closest("[data-navigate-dataset]");
    if (navigateTarget) {
      event.stopPropagation();
      void wrapSubmit(() => submitNavigateForDataset(navigateTarget.dataset.navigateDataset), event);
      return;
    }

    const autoLabelTarget = event.target.closest("[data-autolabel-dataset]");
    if (autoLabelTarget) {
      event.stopPropagation();
      void wrapSubmit(() => submitAutoLabelForDataset(autoLabelTarget.dataset.autolabelDataset), event);
      return;
    }

    const maskTarget = event.target.closest("[data-mask-dataset]");
    if (maskTarget) {
      event.stopPropagation();
      void wrapSubmit(() => submitMaskForDataset(maskTarget.dataset.maskDataset), event);
      return;
    }

    const openTarget = event.target.closest("[data-dataset-open]");
    if (!openTarget) return;
    const datasetName = openTarget.dataset.datasetOpen;
    const kind = openTarget.dataset.datasetKind || "images";
    void openDatasetGallery(datasetName, "browse", kind);
  });

  onId("job-list", "click", (event) => {
    const target = event.target.closest("[data-job-id]");
    if (!target) return;

    const job = state.jobs.find((item) => item.job_id === target.dataset.jobId) || null;
    if (job) {
      selectJob(job);
      renderTaskCenter();
      renderJobDetail();
    }

    closeTaskCenter();
    openDetailModal();
    void selectJobById(target.dataset.jobId);
  });

  onId("job-detail", "click", (event) => {
    const target = event.target.closest("[data-preview-src]");
    if (!target) return;
    openPreviewModal(target.dataset.previewSrc, target.dataset.previewLabel || "图片预览");
  });

  onId("infer-picked-card", "click", (event) => {
    const previewTarget = event.target.closest("[data-preview-src]");
    if (previewTarget) {
      openPreviewModal(previewTarget.dataset.previewSrc, previewTarget.dataset.previewLabel || "图片预览");
      return;
    }

    const repickTarget = event.target.closest("#infer-repick-button");
    if (!repickTarget) return;

    const inferDataset = document.getElementById("infer-dataset");
    const datasetName = state.inferSelection.datasetName || inferDataset?.value || "";
    if (!datasetName) {
      showToast("请先选择数据集。");
      return;
    }
    void openDatasetGallery(datasetName, "infer");
  });
  onId("navigation-picked-card", "click", (event) => {
    const previewTarget = event.target.closest("[data-preview-src]");
    if (previewTarget) {
      openPreviewModal(previewTarget.dataset.previewSrc, previewTarget.dataset.previewLabel || "图片预览");
      return;
    }

    const repickTarget = event.target.closest("#navigation-repick-button");
    if (!repickTarget) return;

    const navDataset = document.getElementById("navigation-dataset");
    const datasetName = state.navigationSelection.datasetName || navDataset?.value || "";
    if (!datasetName) {
      showToast("请先选择数据集。");
      return;
    }
    void openDatasetGallery(datasetName, "navigation");
  });

  onId("dataset-gallery-grid", "click", (event) => {
    const pickTarget = event.target.closest("[data-image-pick]");
    if (pickTarget) {
      if (state.galleryMode === "navigation") {
        setNavigationSelection({
          imagePath: pickTarget.dataset.imagePick,
          imageLabel: pickTarget.dataset.imageLabel || "",
        });
        renderNavigationPanel();
      } else {
        setInferSelection({
          imagePath: pickTarget.dataset.imagePick,
          imageLabel: pickTarget.dataset.imageLabel || "",
        });
        renderInferencePanel();
      }
      closeDatasetGallery();
      return;
    }

    const fileTarget = event.target.closest("[data-file-open]");
    if (fileTarget) {
      window.open(fileTarget.dataset.fileOpen, "_blank", "noopener,noreferrer");
      return;
    }

    const previewTarget = event.target.closest("[data-preview-src]");
    if (!previewTarget) return;
    openPreviewModal(previewTarget.dataset.previewSrc, previewTarget.dataset.previewLabel || "图片预览");
  });

  for (const button of document.querySelectorAll(".tab-button")) {
    button.addEventListener("click", () => setPage(button.dataset.page));
  }

  onId("dataset-gallery-close", "click", closeDatasetGallery);
  onId("dataset-gallery-backdrop", "click", closeDatasetGallery);
  onId("detail-close", "click", closeDetailModal);
  onId("detail-backdrop", "click", closeDetailModal);
  onId("preview-close", "click", closePreviewModal);
  onId("preview-backdrop", "click", closePreviewModal);

  window.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (document.getElementById("preview-modal").classList.contains("open")) {
      closePreviewModal();
      return;
    }
    if (state.datasetGalleryOpen) {
      closeDatasetGallery();
      return;
    }
    if (state.detailModalOpen) {
      closeDetailModal();
      return;
    }
    if (state.taskCenterOpen) {
      closeTaskCenter();
    }
  });

  window.addEventListener("beforeunload", () => {
    state.jobEventSource?.close();
    state.jobEventSource = null;
    window.clearTimeout(reconnectJobEventsHandle);
  });
}

export async function boot() {
  try {
    bindEvents();
    renderPageTabs();
    renderTaskCenterVisibility();
    renderDatasetGalleryVisibility();
    renderDetailModalVisibility();

    const config = await fetchConfig();
    applyConfig(config);
    await refreshDatasets();
    await refreshAnalysisOptions();
    renderInferencePanel();
    renderNavigationPanel();
    renderAnalysisPanel();
    await refreshJobs();
    await refreshArtifacts();
    connectJobEvents();
  } catch (error) {
    showToast(error.message || "初始化失败");
  }
}
