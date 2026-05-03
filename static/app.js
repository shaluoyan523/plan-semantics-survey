const state = {
  catalog: null,
  task: null,
  imageIndex: 0,
  visibleImages: [],
  results: null,
};

const els = {
  surveyView: document.getElementById("surveyView"),
  resultsView: document.getElementById("resultsView"),
  viewTabs: document.querySelectorAll(".viewTabs button"),
  datasetSelect: document.getElementById("datasetSelect"),
  methodSelect: document.getElementById("methodSelect"),
  sampleText: document.getElementById("sampleText"),
  targetText: document.getElementById("targetText"),
  instructionText: document.getElementById("instructionText"),
  prevImage: document.getElementById("prevImage"),
  nextImage: document.getElementById("nextImage"),
  imageTitle: document.getElementById("imageTitle"),
  imageCounter: document.getElementById("imageCounter"),
  imagePager: document.getElementById("imagePager"),
  screenshot: document.getElementById("screenshot"),
  imageStage: document.querySelector(".imageStage"),
  imageCaption: document.getElementById("imageCaption"),
  beforePlan: document.getElementById("beforePlan"),
  afterPlan: document.getElementById("afterPlan"),
  scoreGrid: document.getElementById("scoreGrid"),
  collapseButtons: document.querySelectorAll(".collapsePlan"),
  currentRating: document.getElementById("currentRating"),
  noteInput: document.getElementById("noteInput"),
  saveRating: document.getElementById("saveRating"),
  nextSample: document.getElementById("nextSample"),
  saveStatus: document.getElementById("saveStatus"),
  refreshResults: document.getElementById("refreshResults"),
  metricTotal: document.getElementById("metricTotal"),
  metricSame: document.getElementById("metricSame"),
  metricNear: document.getElementById("metricNear"),
  metricUser: document.getElementById("metricUser"),
  confusionMatrix: document.getElementById("confusionMatrix"),
  differenceChart: document.getElementById("differenceChart"),
  matrixNote: document.getElementById("matrixNote"),
  differenceNote: document.getElementById("differenceNote"),
};

function ratingKey(task = state.task) {
  if (!task) return "";
  return `rating:${task.sampleToken}`;
}

function setStatus(text, mode = "") {
  els.saveStatus.textContent = text;
  els.saveStatus.className = `status ${mode}`.trim();
}

function textOrEmpty(value) {
  return value && String(value).trim() ? value : "No plan text available.";
}

function formatPct(value) {
  return `${Number(value || 0).toFixed(1)}%`;
}

function populateMethods(selectedMethod = null) {
  const dataset = els.datasetSelect.value;
  const methods = state.catalog.datasets[dataset]?.methods || [];
  els.methodSelect.innerHTML = "";
  for (const method of methods) {
    const option = document.createElement("option");
    option.value = method;
    option.textContent = method.replace(/_/g, " ");
    els.methodSelect.appendChild(option);
  }
  const preferred = selectedMethod || state.catalog.default?.method || methods[0] || "";
  els.methodSelect.value = methods.includes(preferred) ? preferred : methods[0] || "";
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

async function loadCatalog() {
  state.catalog = await fetchJson("/api/catalog");
  const preferred = state.catalog.default || {};
  els.datasetSelect.value = preferred.dataset || "pc";
  populateMethods(preferred.method);
}

async function loadTask() {
  const params = new URLSearchParams({
    dataset: els.datasetSelect.value,
    method: els.methodSelect.value,
  });
  if (state.task?.sampleToken) {
    params.set("avoid", state.task.sampleToken);
  }
  setStatus("Loading");
  state.task = await fetchJson(`/api/random-task?${params.toString()}`);
  updateVisibleImages();
  state.imageIndex = Math.max(0, state.visibleImages.length - 1);
  renderTask();
  setStatus("Ready");
}

async function loadResults() {
  state.results = await fetchJson("/api/results");
  renderResults();
}

function updateVisibleImages() {
  const images = state.task?.images || [];
  const targetIndex = Math.max(0, Math.min(state.task?.currentImageIndex || 0, images.length - 1));
  state.visibleImages = images.slice(0, targetIndex + 1);
}

function restoreRating() {
  const saved = localStorage.getItem(ratingKey());
  const score = saved ? JSON.parse(saved).score : null;
  document.querySelectorAll('input[name="score"]').forEach((input) => {
    input.checked = Number(input.value) === score;
  });
  els.noteInput.value = saved ? JSON.parse(saved).note || "" : "";
  els.currentRating.textContent = score === null || score === undefined ? "No score selected" : `Selected score: ${score}`;
}

function renderTask() {
  const task = state.task;
  els.sampleText.textContent = task.sampleLabel;
  els.targetText.textContent = `Step ${task.step} of ${task.totalSteps}`;
  els.instructionText.textContent = task.instruction;
  els.beforePlan.textContent = textOrEmpty(task.beforePlan);
  els.afterPlan.textContent = textOrEmpty(task.afterPlan);
  renderImage();
  restoreRating();
}

function renderImage() {
  const task = state.task;
  const images = state.visibleImages || [];
  if (!images.length) {
    els.imageStage.classList.add("isEmpty");
    els.screenshot.removeAttribute("src");
    els.imageTitle.textContent = "Screenshot";
    els.imageCounter.textContent = "";
    els.imagePager.textContent = "0 / 0";
    els.imageCaption.textContent = "";
    els.prevImage.disabled = true;
    els.nextImage.disabled = true;
    return;
  }

  state.imageIndex = Math.max(0, Math.min(state.imageIndex, images.length - 1));
  const image = images[state.imageIndex];
  els.imageStage.classList.remove("isEmpty");
  els.screenshot.src = image.url;
  els.imageTitle.textContent = state.imageIndex === images.length - 1 ? "Target screenshot" : "Previous screenshot";
  const runLabel = task.runLabel || task.run || "";
  els.imageCounter.textContent = `${task.dataset.toUpperCase()} | ${runLabel} | ${task.method.replace(/_/g, " ")} | target step ${task.step}`;
  els.imagePager.textContent = `${state.imageIndex + 1} / ${images.length}`;
  els.imageCaption.textContent = "";
  els.prevImage.disabled = state.imageIndex <= 0;
  els.nextImage.disabled = state.imageIndex >= images.length - 1;
}

function togglePlan(target) {
  const panel = document.querySelector(target === "before" ? ".planPanel.baseline" : ".planPanel.comparison");
  const button = document.querySelector(`.collapsePlan[data-target="${target}"]`);
  if (!panel || !button) return;
  const isCollapsed = panel.classList.toggle("collapsed");
  button.textContent = isCollapsed ? "Expand" : "Collapse";
}

function selectedScore() {
  const checked = document.querySelector('input[name="score"]:checked');
  return checked ? Number(checked.value) : null;
}

async function saveRating() {
  const score = selectedScore();
  if (score === null) {
    setStatus("Pick a score", "error");
    return;
  }
  const payload = {
    sampleToken: state.task.sampleToken,
    score,
    note: els.noteInput.value.trim(),
  };
  setStatus("Saving");
  await fetchJson("/api/rating", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  localStorage.setItem(ratingKey(), JSON.stringify({ score, note: payload.note }));
  els.currentRating.textContent = `Selected score: ${score}`;
  setStatus("Saved", "saved");
  loadResults().catch(() => {});
}

function showView(view) {
  const isResults = view === "results";
  els.surveyView.hidden = isResults;
  els.resultsView.hidden = !isResults;
  els.viewTabs.forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  if (isResults) {
    loadResults().catch((error) => setStatus(error.message, "error"));
  }
}

function renderResults() {
  const results = state.results;
  if (!results) return;
  const combined = results.combined;
  const user = results.user;
  els.metricTotal.textContent = String(combined.summary.total);
  els.metricSame.textContent = formatPct(combined.summary.samePct);
  els.metricNear.textContent = formatPct(combined.summary.samePct + combined.summary.plusMinus1Pct);
  els.metricUser.textContent = String(user.summary.total);
  renderMatrix(combined.confusion, results.scores);
  renderDifferenceChart(combined.differenceCounts, results.diffs, combined.summary.total);
  els.matrixNote.textContent = `Initial data: ${results.initial.summary.total} samples. User-added data: ${user.summary.total} ratings.`;
  els.differenceNote.textContent = `Exact: ${combined.summary.same} (${formatPct(combined.summary.samePct)}), within +/-1: ${combined.summary.same + combined.summary.plusMinus1} (${formatPct(combined.summary.samePct + combined.summary.plusMinus1Pct)}).`;
}

function renderMatrix(confusion, scores) {
  const maxCount = Math.max(1, ...confusion.flat());
  els.confusionMatrix.innerHTML = "";
  els.confusionMatrix.appendChild(document.createElement("div"));
  for (const score of scores) {
    const label = document.createElement("div");
    label.className = "matrixLabel";
    label.textContent = `LLM ${score}`;
    els.confusionMatrix.appendChild(label);
  }
  confusion.forEach((row, humanScore) => {
    const rowLabel = document.createElement("div");
    rowLabel.className = "matrixLabel";
    rowLabel.textContent = String(humanScore);
    els.confusionMatrix.appendChild(rowLabel);
    const rowTotal = row.reduce((sum, value) => sum + value, 0);
    row.forEach((count) => {
      const cell = document.createElement("div");
      cell.className = "matrixCell";
      const intensity = count / maxCount;
      cell.style.background = `rgba(37, 99, 235, ${0.08 + intensity * 0.78})`;
      cell.innerHTML = `${count}<small>${rowTotal ? formatPct((count / rowTotal) * 100) : "0.0%"}</small>`;
      els.confusionMatrix.appendChild(cell);
    });
  });
}

function renderDifferenceChart(counts, diffs, total) {
  const maxCount = Math.max(1, ...diffs.map((diff) => counts[String(diff)] || 0));
  els.differenceChart.innerHTML = "";
  for (const diff of diffs) {
    const count = counts[String(diff)] || 0;
    const row = document.createElement("div");
    row.className = "barRow";
    const label = document.createElement("div");
    label.className = "barLabel";
    label.textContent = diff > 0 ? `+${diff}` : String(diff);
    const track = document.createElement("div");
    track.className = "barTrack";
    const fill = document.createElement("div");
    fill.className = "barFill";
    fill.style.width = `${Math.max(1, (count / maxCount) * 100)}%`;
    track.appendChild(fill);
    const value = document.createElement("div");
    value.className = "barValue";
    value.textContent = `${count} (${total ? formatPct((count / total) * 100) : "0.0%"})`;
    row.append(label, track, value);
    els.differenceChart.appendChild(row);
  }
}

function goToNextSample() {
  loadTask().catch((error) => setStatus(error.message, "error"));
}

els.datasetSelect.addEventListener("change", () => {
  populateMethods();
  loadTask().catch((error) => setStatus(error.message, "error"));
});

els.methodSelect.addEventListener("change", () => {
  loadTask().catch((error) => setStatus(error.message, "error"));
});

els.prevImage.addEventListener("click", () => {
  state.imageIndex -= 1;
  renderImage();
});

els.nextImage.addEventListener("click", () => {
  state.imageIndex += 1;
  renderImage();
});

els.scoreGrid.addEventListener("change", () => {
  const score = selectedScore();
  els.currentRating.textContent = score === null ? "No score selected" : `Selected score: ${score}`;
});

els.collapseButtons.forEach((button) => {
  button.addEventListener("click", () => togglePlan(button.dataset.target));
});

els.saveRating.addEventListener("click", () => {
  saveRating().catch((error) => setStatus(error.message, "error"));
});

els.nextSample.addEventListener("click", goToNextSample);

els.viewTabs.forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view));
});

els.refreshResults.addEventListener("click", () => {
  loadResults().catch((error) => setStatus(error.message, "error"));
});

document.addEventListener("keydown", (event) => {
  if (!state.task) return;
  if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement) return;
  if (event.key === "ArrowLeft") {
    state.imageIndex -= 1;
    renderImage();
  } else if (event.key === "ArrowRight") {
    state.imageIndex += 1;
    renderImage();
  } else if (/^[0-4]$/.test(event.key)) {
    const input = document.querySelector(`input[name="score"][value="${event.key}"]`);
    if (input) {
      input.checked = true;
      els.currentRating.textContent = `Selected score: ${event.key}`;
    }
  }
});

loadCatalog()
  .then(() => loadTask())
  .then(() => loadResults())
  .catch((error) => setStatus(error.message, "error"));
