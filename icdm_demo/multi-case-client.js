Object.assign(labelNames, {
  "住宅开发Ⅲ": "Residential Development",
  "物业经营Ⅲ": "Property Operations",
  "仓储物流Ⅲ": "Warehousing & Logistics",
  "商业地产Ⅲ": "Commercial Real Estate",
  "畜禽养殖Ⅲ": "Livestock Breeding",
  "基础设施建设Ⅲ": "Infrastructure Construction",
  "新能源发电Ⅲ": "Renewable Power Generation",
  "环保设备Ⅲ": "Environmental Equipment",
  "酒店及餐饮Ⅲ": "Hotels & Catering",
  "旅游综合Ⅲ": "Integrated Tourism"
});

let selectedCaseId = "gree-2021";
let selectedCaseResult = null;
const evidenceCursor = { text: 0, table: 0 };

const evidenceNav = document.createElement("div");
evidenceNav.className = "evidence-nav";
evidenceNav.innerHTML = `<button type="button" data-evidence-step="-1" aria-label="Previous evidence">←</button><span id="evidence-position">0 / 0</span><button type="button" data-evidence-step="1" aria-label="Next evidence">→</button>`;
document.querySelector(".evidence-panel .panel-heading").after(evidenceNav);

const caseSelect = document.createElement("select");
caseSelect.className = "case-select";
caseSelect.disabled = true;
caseSelect.innerHTML = "<option>Loading cases…</option>";
document.querySelector(".case-line").insertBefore(caseSelect, backendBadge);

const priorRunButton = apiRunButton;
const caseRunButton = priorRunButton.cloneNode(true);
priorRunButton.replaceWith(caseRunButton);

function displayLabelName(tag) {
  return language === "en" ? (labelNames[tag] || tag) : tag;
}

function getEvidenceItems(channel) {
  if (!selectedCaseResult) return [];
  if (channel === "text") {
    const items = selectedCaseResult.text_evidence?.items;
    return Array.isArray(items) && items.length ? items : (selectedCaseResult.text_evidence?.excerpt ? [selectedCaseResult.text_evidence] : []);
  }
  return selectedCaseResult.table_evidence?.items || [];
}

function currentEvidence(channel) {
  const items = getEvidenceItems(channel);
  if (!items.length) return null;
  evidenceCursor[channel] %= items.length;
  return items[evidenceCursor[channel]];
}

function updateEvidenceNav(channel) {
  const items = getEvidenceItems(channel);
  const position = items.length ? evidenceCursor[channel] + 1 : 0;
  document.getElementById("evidence-position").textContent = `${position} / ${items.length}`;
  evidenceNav.querySelectorAll("button").forEach(button => {
    button.disabled = items.length <= 1;
    button.title = backendText(button.dataset.evidenceStep === "-1" ? "Previous evidence" : "Next evidence", button.dataset.evidenceStep === "-1" ? "上一条证据" : "下一条证据");
  });
}

function renderCaseText() {
  const container = document.querySelector("#text-evidence blockquote");
  const item = currentEvidence("text");
  container.textContent = item?.excerpt || backendText("No text evidence was retained for this case.", "该案例没有保留可展示的文本证据。");
}

function renderEvidence(channel) {
  if (!selectedCaseResult) return;
  if (channel === "text") renderCaseText();
  else renderCaseTable(selectedCaseResult);
  updateEvidenceProvenance(channel);
  updateEvidenceNav(channel);
}

function updateCaseIdentity(result, resetEvidence = true) {
  if (resetEvidence) evidenceCursor.text = evidenceCursor.table = 0;
  selectedCaseResult = result;
  const company = result.company;
  document.querySelector(".case-line strong").textContent = company.name;
  document.querySelector(".case-line small").textContent = `${company.ticker} · ${language === "zh" ? `${company.year} 年年度报告` : `Annual Report ${company.year}`}`;
  const cleanName = company.name.replace(/\s+/g, "");
  document.querySelector(".avatar").textContent = cleanName.slice(0, 2).toUpperCase();
  const firstLabel = result.labels?.[0];
  if (firstLabel) document.querySelector(".evidence-foot b").textContent = displayLabelName(firstLabel.Tag);
  const activeChannel = document.querySelector("[data-channel].active")?.dataset.channel || "text";
  renderEvidence(activeChannel);
}

function updateEvidenceProvenance(channel) {
  if (!selectedCaseResult) return;
  const isText = channel === "text";
  const evidence = currentEvidence(channel);
  const supports = evidence?.supports || [];
  const supportNode = document.querySelector(".evidence-foot b");
  if (supportNode) {
    supportNode.textContent = supports.length
      ? supports.map(displayLabelName).join(" · ")
      : backendText("Retrieved evidence", "检索证据");
  }
  const sourceNode = document.getElementById("source-name");
  if (!sourceNode) return;
  if (isText) {
    const rank = evidence?.retrieval_rank ? ` #${evidence.retrieval_rank}` : "";
    sourceNode.textContent = `${backendText("Text retrieval · supporting passage", "文本检索 · 支持段落")}${rank}`;
  } else {
    const rank = evidence?.retrieval_rank ? ` #${evidence.retrieval_rank}` : "";
    const title = evidence?.title || evidence?.heading || backendText("retrieved table", "检索表格");
    sourceNode.textContent = `${backendText("Table retrieval", "表格检索")} · ${title}${rank}`;
  }
}

function parseMarkdownTable(text) {
  if (!text) return [];
  return text.split("\n")
    .map(line => line.trim())
    .filter(line => line.startsWith("|") && line.endsWith("|"))
    .map(line => line.slice(1, -1).split("|").map(cell => cell.trim()))
    .filter(row => !row.every(cell => /^:?-{2,}:?$/.test(cell)));
}

function renderCaseTable(result) {
  const container = document.getElementById("table-evidence");
  const item = currentEvidence("table");
  if (!item) {
    container.innerHTML = `<p class="table-note">${backendText("No table evidence was retained for this case.", "该案例没有保留可展示的表格证据。")}</p>`;
    return;
  }
  const rows = parseMarkdownTable(item.table_text);
  if (rows.length >= 2) {
    const header = rows[0].slice(0, 4);
    const body = rows.slice(1, 6);
    const noteTitle = item.title || backendText("Retrieved table", "检索表格");
    const noteReason = backendText("Table retrieval evidence", "表格检索证据");
    container.innerHTML = `<table><thead><tr>${header.map(cell => `<th>${cell}</th>`).join("")}</tr></thead><tbody>${body.map((row, index) => `<tr class="${index === 0 ? "focus" : ""}">${header.map((_, cellIndex) => `<td>${row[cellIndex] || "—"}</td>`).join("")}</tr>`).join("")}</tbody></table><p class="table-note"><b>${noteTitle}</b> · ${noteReason}</p>`;
  } else {
    const trace = item.graph_text || item.reasoning_trace || item.title || backendText("Table retrieval evidence", "表格检索证据");
    container.innerHTML = `<p class="table-note">${presentationText(trace).slice(0, 900)}</p>`;
  }
}

async function loadCase(caseId) {
  selectedCaseId = caseId;
  caseSelect.disabled = true;
  document.querySelector(".demo-card").classList.add("case-loading");
  try {
    const response = await fetch(`/api/demo/cases/${encodeURIComponent(caseId)}?lang=${encodeURIComponent(language)}`);
    if (!response.ok) throw new Error("Case API failed");
    const result = await response.json();
    applyFeatured(result);
    updateCaseIdentity(result);
    runState = "idle";
    updateCaseRunButton();
    document.querySelectorAll(".step").forEach(step => step.classList.remove("active"));
  } catch (error) {
    console.error(error);
    backendBadge.className = "backend-badge error";
    backendBadge.querySelector("span").textContent = backendText("Case load failed", "案例加载失败");
  } finally {
    caseSelect.disabled = false;
    document.querySelector(".demo-card").classList.remove("case-loading");
  }
}

function updateCaseRunButton() {
  caseRunButton.textContent = runState === "idle" ? copy[language].run : runState === "running" ? copy[language].running : copy[language].complete;
}

caseSelect.addEventListener("change", () => loadCase(caseSelect.value));

document.querySelectorAll("[data-channel]").forEach(button => button.addEventListener("click", () => {
  renderEvidence(button.dataset.channel);
}));

evidenceNav.querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
  const channel = document.querySelector("[data-channel].active")?.dataset.channel || "text";
  const items = getEvidenceItems(channel);
  if (items.length <= 1) return;
  evidenceCursor[channel] = (evidenceCursor[channel] + Number(button.dataset.evidenceStep) + items.length) % items.length;
  renderEvidence(channel);
}));

caseRunButton.addEventListener("click", async () => {
  caseRunButton.disabled = true;
  caseSelect.disabled = true;
  runState = "running";
  updateCaseRunButton();
  runDetail.classList.add("visible");
  runDetail.innerHTML = `<span>${backendText("Running selected case from real artifacts…", "正在运行所选案例的真实产物流程…")}</span><strong>${selectedCaseId}</strong>`;
  try {
    const response = await fetch(`/api/demo/cases/${encodeURIComponent(selectedCaseId)}/runs`, { method: "POST" });
    if (!response.ok) throw new Error("Unable to create case run");
    const created = await response.json();
    const run = await pollRun(created.run_id);
    applyFeatured(run.result);
    updateCaseIdentity(run.result);
    runState = "complete";
    updateCaseRunButton();
  } catch (error) {
    console.error(error);
    runState = "idle";
    caseRunButton.textContent = backendText("Retry backend", "重试后端");
  } finally {
    caseRunButton.disabled = false;
    caseSelect.disabled = false;
  }
});

document.querySelectorAll("[data-language]").forEach(button => button.addEventListener("click", () => {
  updateCaseRunButton();
  loadCaseCatalog();
}));

async function loadCaseCatalog() {
  try {
    const response = await fetch(`/api/demo/cases?lang=${encodeURIComponent(language)}`);
    if (!response.ok) throw new Error("Case catalog unavailable");
    const cases = await response.json();
    caseSelect.innerHTML = cases.map(item => `<option value="${item.id}">${language === "zh" ? item.name.replace(/\s+/g, "") : item.name} · ${item.year}</option>`).join("");
    caseSelect.value = selectedCaseId;
    caseSelect.disabled = false;
    backendBadge.title = `${backendBadge.title || "FastAPI"} · ${cases.length} cases`;
    await loadCase(selectedCaseId);
  } catch (error) {
    console.error(error);
    caseSelect.innerHTML = "<option>Gree Electric · 2021</option>";
  }
}

loadCaseCatalog();
