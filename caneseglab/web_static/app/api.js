function withNoCache(url) {
  const joiner = url.includes("?") ? "&" : "?";
  return `${url}${joiner}_=${Date.now()}`;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    ...options,
  });

  const isJson = response.headers.get("content-type")?.includes("application/json");
  const payload = isJson ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof payload === "string" ? payload : payload?.error || response.statusText;
    throw new Error(message);
  }
  return payload;
}

export function fetchConfig() {
  return requestJson(withNoCache("/api/config"));
}

export function fetchDatasets() {
  return requestJson(withNoCache("/api/datasets"));
}

export function fetchDatasetImages(datasetName) {
  return requestJson(withNoCache(`/api/datasets/${encodeURIComponent(datasetName)}/images`));
}

export function fetchDatasetFiles(datasetName, kind) {
  return requestJson(
    withNoCache(`/api/datasets/${encodeURIComponent(datasetName)}/files?kind=${encodeURIComponent(kind)}`),
  );
}

export function fetchJobs() {
  return requestJson(withNoCache("/api/jobs"));
}

export function fetchJobDetail(jobId) {
  return requestJson(withNoCache(`/api/jobs/${encodeURIComponent(jobId)}`));
}

export function fetchArtifacts() {
  return requestJson(withNoCache("/api/artifacts"));
}

export function fetchAnalysisOptions() {
  return requestJson(withNoCache("/api/analysis/options"));
}

export function openJobEvents() {
  return new EventSource("/api/events/jobs");
}

export function submitJob(url, payload) {
  return requestJson(url, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
