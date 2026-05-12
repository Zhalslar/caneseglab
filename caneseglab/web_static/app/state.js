export const state = {
  config: null,
  datasets: [],
  datasetMap: {},
  datasetImages: {},
  datasetFiles: {},
  artifacts: null,
  analysisOptions: {
    trainRuns: [],
  },
  jobs: [],
  selectedJobId: null,
  selectedJob: null,
  selectedDatasetName: null,
  jobEventSource: null,
  activePage: "data",
  taskCenterOpen: false,
  detailModalOpen: false,
  datasetGalleryOpen: false,
  galleryMode: "browse",
  galleryKind: "images",
  inferSelection: {
    method: "verify-onnx",
    datasetName: null,
    onnxPath: "",
    enginePath: "",
    imagePath: "",
    imageLabel: "",
  },
  navigationSelection: {
    method: "navigate-onnx",
    datasetName: null,
    onnxPath: "",
    enginePath: "",
    imagePath: "",
    imageLabel: "",
  },
};

function joinDatasetPath(root, leaf) {
  if (!root) return "";
  const separator = root.includes("\\") ? "\\" : "/";
  return `${root}${separator}${leaf}`;
}

function toCount(value) {
  return Number.isFinite(value) ? value : 0;
}

function normalizeDataset(dataset) {
  const rootPath = dataset?.path || "";

  return {
    ...dataset,
    path: rootPath,
    image_dir: dataset?.image_dir || joinDatasetPath(rootPath, "images"),
    annotation_dir: dataset?.annotation_dir || joinDatasetPath(rootPath, "annotations"),
    mask_dir: dataset?.mask_dir || joinDatasetPath(rootPath, "masks"),
    overlay_dir: dataset?.overlay_dir || joinDatasetPath(rootPath, "overlays"),
    navigation_dir: dataset?.navigation_dir || joinDatasetPath(rootPath, "navigation"),
    image_count: toCount(dataset?.image_count),
    annotation_count: toCount(dataset?.annotation_count),
    mask_count: toCount(dataset?.mask_count),
    overlay_count: toCount(dataset?.overlay_count),
    navigation_count: toCount(dataset?.navigation_count),
  };
}

export function datasetByName(name) {
  return state.datasetMap[name] || null;
}

export function setDatasets(datasets) {
  state.datasets = (datasets || []).map(normalizeDataset);
  state.datasetMap = Object.fromEntries(state.datasets.map((dataset) => [dataset.name, dataset]));
}

export function setArtifacts(payload) {
  state.artifacts = payload;
}

export function setAnalysisOptions(payload) {
  state.analysisOptions = {
    trainRuns: payload?.train_runs || [],
    trainRoot: payload?.train_root || "",
  };
}

export function setJobs(jobs) {
  state.jobs = jobs || [];

  if (!state.jobs.length) {
    state.selectedJobId = null;
    state.selectedJob = null;
    return;
  }

  if (!state.selectedJobId) {
    state.selectedJobId = state.jobs[0].job_id;
  }

  const selected = state.jobs.find((item) => item.job_id === state.selectedJobId) || null;
  state.selectedJob = selected;

  if (!selected) {
    state.selectedJobId = state.jobs[0].job_id;
    state.selectedJob = state.jobs[0];
  }
}

export function selectJob(job) {
  state.selectedJobId = job?.job_id || null;
  state.selectedJob = job || null;
}

export function setActivePage(page) {
  state.activePage = page;
}

export function setTaskCenterOpen(open) {
  state.taskCenterOpen = open;
}

export function setDetailModalOpen(open) {
  state.detailModalOpen = open;
}

export function selectDataset(name) {
  state.selectedDatasetName = name || null;
}

export function setDatasetGalleryOpen(open) {
  state.datasetGalleryOpen = open;
}

export function setGalleryMode(mode) {
  state.galleryMode = mode || "browse";
}

export function setGalleryKind(kind) {
  state.galleryKind = kind || "images";
}

export function setInferSelection(patch) {
  state.inferSelection = {
    ...state.inferSelection,
    ...(patch || {}),
  };
}

export function setNavigationSelection(patch) {
  state.navigationSelection = {
    ...state.navigationSelection,
    ...(patch || {}),
  };
}
