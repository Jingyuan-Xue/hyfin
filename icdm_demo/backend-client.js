const backendBadge = document.createElement("span");
backendBadge.className = "backend-badge";
backendBadge.innerHTML = "<i></i><span>Connecting API…</span>";
document.querySelector(".case-line").insertBefore(backendBadge, document.getElementById("run-demo"));

const oldRunButton = document.getElementById("run-demo");
const apiRunButton = oldRunButton.cloneNode(true);
oldRunButton.replaceWith(apiRunButton);

const runDetail = document.createElement("div");
runDetail.className = "run-detail";
runDetail.innerHTML = "<span>Execution</span><strong>artifact-backed pipeline</strong>";
document.querySelector(".steps").after(runDetail);

let featuredResult = null;

function backendText(en, zh) { return language === "zh" ? zh : en; }

function presentationText(value) {
  return String(value || "")
    .replace(/A2RAG/gi, backendText("text retrieval", "文本检索"))
    .replace(/TabGR/gi, backendText("table retrieval", "表格检索"));
}

function displayLanguageQuery() { return "lang=" + encodeURIComponent(language); }

const labelRationalesEnglish = {
  "产业用房运营": "The retained evidence identifies recurring operation of industrial properties.",
  "住宅开发": "The retained evidence identifies residential property development as a material business.",
  "医疗医美": "The retained evidence supports healthcare and medical-aesthetics activities.",
  "咨询服务": "The retained evidence supports a distinct consulting-services business.",
  "商业园区运营": "The retained evidence supports recurring commercial-park operations.",
  "园区开发": "The retained evidence supports industrial-park development activities.",
  "城市更新土地开发": "The retained evidence supports urban-renewal and land-development activities.",
  "工程施工承包": "The retained evidence supports engineering and construction contracting.",
  "投资资产管理": "The retained evidence supports investment and asset-management activities.",
  "新能源电力": "The retained evidence supports renewable-power generation or investment.",
  "水务工程投资": "The retained evidence supports investment in water infrastructure projects.",
  "物业租赁运营": "The retained evidence supports recurring property leasing operations.",
  "物业管理服务": "The retained evidence supports property-management services.",
  "物业运营": "The retained evidence supports recurring property operations.",
  "环保再生资源": "The retained evidence supports environmental and resource-recycling activities.",
  "电子信息产品": "The retained evidence supports electronic information products.",
  "节能服务": "The retained evidence supports energy-efficiency services.",
  "装饰设计": "The retained evidence supports interior design and decoration services.",
  "贸易批发": "The retained evidence supports wholesale trading activities.",
  "资产租赁": "The retained evidence supports asset-leasing activities.",
  "酒店经营": "The retained evidence supports hotel operations.",
  "金属制品": "The retained evidence supports metal-products manufacturing.",
  "风机钢混塔筒制造": "The retained evidence supports wind-turbine steel–concrete tower manufacturing.",
  "空调制造": "The evidence identifies air conditioners as the company’s principal product business.",
  "生活电器制造": "The evidence supports a distinct home-appliance manufacturing business.",
  "智能装备制造": "The evidence supports a distinct intelligent-equipment manufacturing business."
};

function englishLabelReason(label) {
  const definition = String(label.Definition || "");
  if (labelRationalesEnglish[label.Tag]) return labelRationalesEnglish[label.Tag];
  return /[\u3400-\u9fff]/.test(definition) ? "Supported by the retained text and table evidence." : (definition || "Supported by the retained text and table evidence.");
}

function updateVisibleRunButton() {
  apiRunButton.textContent = runState === "idle"
    ? copy[language].run
    : runState === "running"
      ? copy[language].running
      : copy[language].complete;
}

function applyFeatured(result) {
  featuredResult = result;
  const excerpt = result.text_evidence?.excerpt || "";
  if (excerpt) {
    document.querySelector("#text-evidence blockquote").textContent = excerpt.length > 650 ? `${excerpt.slice(0, 650)}…` : excerpt;
  }
  const labels = result.labels || [];
  if (labels.length) {
    labelSets.zh.hybrid = labels.map((label, index) => [
      index === 0 ? "主标签" : "次标签",
      label.Tag,
      Math.round(Number(label.Confidence) * 100),
      presentationText(label.Reason),
    ]);
    labelSets.en.hybrid = labels.map((label, index) => [
      index === 0 ? "PRIMARY" : "SECONDARY",
      labelNames[label.Tag] || label.Tag,
      Math.round(Number(label.Confidence) * 100),
      language === "en" ? englishLabelReason(label) : label.Reason,
    ]);
    renderLabels();
  }
  const execution = result.execution || {};
  runDetail.innerHTML = `<span>${backendText("Evidence sources", "证据来源")}: <b>${backendText("Text retrieval", "文本检索")} + ${backendText("Table retrieval", "表格检索")}</b></span><strong>${backendText("Evidence aligned", "证据已对齐")}</strong>`;
}

async function loadBackend() {
  try {
    const [healthResponse, experimentResponse, featuredResponse] = await Promise.all([
      fetch("/api/health"), fetch("/api/experiments"), fetch("/api/demo/featured?" + displayLanguageQuery()),
    ]);
    if (!healthResponse.ok || !experimentResponse.ok || !featuredResponse.ok) throw new Error("API response failed");
    const health = await healthResponse.json();
    data = await experimentResponse.json();
    const featured = await featuredResponse.json();
    applyFeatured(featured);
    renderMetrics();
    renderDataTable();
    backendBadge.className = "backend-badge online";
    backendBadge.querySelector("span").textContent = backendText("Backend connected", "后端已连接");
    backendBadge.title = `FastAPI · ${backendText("Text retrieval", "文本检索")} ${health.services.a2rag ? "online" : "cached"} · ${backendText("Table retrieval", "表格检索")} ${health.services.tabgr ? "online" : "cached"}`;
    runDetail.classList.add("visible");
  } catch (error) {
    console.error(error);
    backendBadge.className = "backend-badge error";
    backendBadge.querySelector("span").textContent = backendText("Snapshot fallback", "使用前端快照");
  }
}

function setStages(stages) {
  document.querySelectorAll(".step").forEach((node, index) => {
    const stage = stages[index];
    node.classList.toggle("active", stage && (stage.status === "running" || stage.status === "complete"));
  });
}

async function pollRun(runId) {
  for (;;) {
    const response = await fetch(`/api/demo/runs/${runId}?lang=${encodeURIComponent(language)}`);
    if (!response.ok) throw new Error("Run polling failed");
    const run = await response.json();
    setStages(run.stages || []);
    if (run.status === "complete") return run;
    if (run.status === "failed") throw new Error(run.error || "Pipeline run failed");
    await new Promise(resolve => setTimeout(resolve, 220));
  }
}

apiRunButton.addEventListener("click", async () => {
  apiRunButton.disabled = true;
  runState = "running";
  updateVisibleRunButton();
  runDetail.classList.add("visible");
  runDetail.innerHTML = `<span>${backendText("Reading real pipeline artifacts…", "正在读取真实 Pipeline 产物…")}</span><strong>FastAPI</strong>`;
  try {
    const response = await fetch("/api/demo/runs", { method: "POST" });
    if (!response.ok) throw new Error("Unable to create backend run");
    const created = await response.json();
    const run = await pollRun(created.run_id);
    applyFeatured(run.result);
    runState = "complete";
    updateVisibleRunButton();
  } catch (error) {
    console.error(error);
    runState = "idle";
    apiRunButton.textContent = backendText("Retry backend", "重试后端");
    runDetail.innerHTML = `<span>${backendText("Backend run failed", "后端运行失败")}</span><strong>${error.message}</strong>`;
  } finally {
    apiRunButton.disabled = false;
  }
});

document.querySelectorAll("[data-language]").forEach(button => button.addEventListener("click", () => {
  updateVisibleRunButton();
  if (backendBadge.classList.contains("online")) backendBadge.querySelector("span").textContent = backendText("Backend connected", "后端已连接");
  if (featuredResult) applyFeatured(featuredResult);
  loadBackend();
}));

loadBackend();
