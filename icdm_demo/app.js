
let data = window.EXPERIMENT_DATA;
let language = localStorage.getItem("icdm-demo-language") || "en";
let activeMode = "hybrid";
let activeDataTab = "companies";
let searchTerm = "";
let runState = "idle";
let demoTimer;

const copy = {
  en: {
    demoGroup: "DEMO", resultsGroup: "RESULTS", liveNav: "Industry Analytics", qaNav: "QA", riskNav: "Risk Exposure", experimentsNav: "Experiments",
    qaTitle: "Annual report Q&A",
    qaSubtitle: "Ask a company-specific question and verify the answer against cited disclosure.",
    riskTitle: "Risk exposure",
    riskSubtitle: "Review the most material company risks and open the annual-report evidence behind each conclusion.",
    paper: "Paper", code: "Code", event: "ICDM Demo",
    liveTitle: "Industry Analytics", liveSubtitle: "Identify a company’s fine-grained business lines from narrative and financial evidence.",
    experimentTitle: "Experimental results.", experimentSubtitle: "Fine-grained industry evaluation on SW3 430101.",
    company: "Gree Electric Appliances", companyMeta: "000651.SZ · Annual Report 2021",
    run: "▶ Run demo", running: "Running…", complete: "✓ Complete",
    steps: ["Parse", "Retrieve", "Align", "Generate"],
    textEvidence: "Text evidence", tableEvidence: "Table evidence", mode: "Mode",
    modes: { text: "Text", table: "Table", hybrid: "Hybrid" },
    evidence: "EVIDENCE", generated: "GENERATED LABELS", supports: "Supports",
    textSource: "Management discussion · Main business", tableSource: "Revenue composition · Product segments",
    tableHeads: ["Segment", "Revenue", "Share", "YoY"],
    tableRows: ["Air conditioner", "Home appliances", "Intelligent equipment"],
    tableNote: "Air conditioners contribute 70.11% of total revenue and are the economically dominant business line.",
    metricLabels: ["Official SW3 companies", "Companies retained after filtering", "Companies with price data", "Discovered groups", "Random partitions"],
    chartEyebrow: "MARKET-NEUTRAL COMOVEMENT", chartTitle: "Finer groups move closer together.",
    baselineEyebrow: "RANDOM BASELINE", baselineTitle: "Significant against random partitions.",
    pValue: "p-value · 1,000 trials",
    baselineBody: "The final dense-hybrid partition is significantly above the random baseline (1,000 trials).",
    bars: ["Official SW3", "L4 within-group", "L4 cross-group"],
    dataEyebrow: "FULL EXPERIMENT DATA", dataTitle: "Explore the evaluation output.", search: "Search company or label…",
    dataTabs: ["Companies", "Clusters", "Labels"], source: "Source: final dense-hybrid run · conf_ge_050_max15",
    rowsShown: "rows shown",
    companyHeads: ["Company ID", "Company", "Cluster", "Fine-grained labels", "Count"],
    clusterHeads: ["Cluster", "Companies", "Within comovement", "Top labels", "Members"],
    labelHeads: ["Canonical label", "Company frequency", "IDF", "Status"], kept: "KEPT"
  },
  zh: {
    demoGroup: "现场演示", resultsGroup: "实验结果", liveNav: "行业分析", qaNav: "QA问答", riskNav: "风险暴露", experimentsNav: "完整实验",
    qaTitle: "年报问答",
    qaSubtitle: "围绕一家公司提问，并用年报引用核查回答。",
    riskTitle: "风险暴露",
    riskSubtitle: "优先查看重要风险，并展开每项结论背后的年报原文。",
    paper: "论文", code: "代码", event: "ICDM 系统演示",
    liveTitle: "行业分析", liveSubtitle: "结合叙述与财务证据，识别企业的细粒度业务构成。",
    experimentTitle: "完整实验结果。", experimentSubtitle: "申万三级行业 430101 的细粒度行业评估。",
    company: "珠海格力电器股份有限公司", companyMeta: "000651.SZ · 2021 年年度报告",
    run: "▶ 运行演示", running: "正在运行…", complete: "✓ 运行完成",
    steps: ["解析", "检索", "对齐", "生成"],
    textEvidence: "文字证据", tableEvidence: "表格证据", mode: "模式",
    modes: { text: "仅文本", table: "仅表格", hybrid: "混合" },
    evidence: "证据", generated: "生成的行业标签", supports: "支持标签",
    textSource: "管理层讨论与分析 · 主要业务", tableSource: "营业收入构成 · 分产品",
    tableHeads: ["业务板块", "营业收入", "收入占比", "同比增长"],
    tableRows: ["空调", "生活电器", "智能装备"],
    tableNote: "空调业务贡献了 70.11% 的营业收入，是公司经济意义上最重要的核心业务。",
    metricLabels: ["官方 SW3 公司", "过滤后保留的公司", "具有价格数据的公司", "发现的业务群组", "随机分组实验"],
    chartEyebrow: "市场中性共振", chartTitle: "细粒度组内公司具有更强共振。",
    baselineEyebrow: "随机基线", baselineTitle: "显著优于随机分组。",
    pValue: "p 值 · 1,000 次随机实验",
    baselineBody: "最终 dense-hybrid 分组在 1,000 次随机实验中显著优于随机基线。",
    bars: ["官方 SW3", "L4 组内", "L4 跨组"],
    dataEyebrow: "完整实验数据", dataTitle: "浏览评估输出。", search: "搜索公司或标签…",
    dataTabs: ["公司", "群组", "标签"], source: "数据来源：最终 dense-hybrid 运行 · conf_ge_050_max15",
    rowsShown: "行数据",
    companyHeads: ["公司代码", "公司名称", "所属群组", "细粒度标签", "数量"],
    clusterHeads: ["群组", "公司数", "组内共振", "主要标签", "成员公司"],
    labelHeads: ["标准标签", "公司频次", "IDF", "状态"], kept: "保留"
  }
};

const labelNames = {
  "物业管理服务": "Property management services", "物业租赁运营": "Property leasing operations",
  "工程施工承包": "Engineering construction", "酒店经营": "Hotel operations",
  "商业园区运营": "Commercial park operations", "投资资产管理": "Investment & asset management",
  "城市更新土地开发": "Urban renewal & land development", "贸易批发": "Wholesale trade",
  "新能源电力": "New energy power", "环保再生资源": "Environmental recycling",
  "租赁业务": "Leasing services", "房地产经纪": "Real-estate brokerage",
  "房地产销售代理": "Real-estate sales agency", "文旅演艺": "Culture, tourism & performance",
  "产业用房运营": "Industrial property operations", "住宅开发": "Residential development",
  "医疗医美": "Healthcare & medical aesthetics", "咨询服务": "Consulting services",
  "园区开发": "Industrial park development", "水务工程投资": "Water infrastructure investment",
  "物业运营": "Property operations", "电子信息产品": "Electronic information products",
  "节能服务": "Energy-efficiency services", "装饰设计": "Interior design & decoration",
  "资产租赁": "Asset leasing", "金属制品": "Metal products",
  "风机钢混塔筒制造": "Wind-turbine steel–concrete tower manufacturing",
  "空调制造": "Air conditioner manufacturing",
  "生活电器制造": "Home appliance manufacturing",
  "智能装备制造": "Intelligent equipment manufacturing"
};

const labelSets = {
  en: {
    text: [["PRIMARY", "Air Conditioner Manufacturing", 82, "Repeatedly described as the flagship product."], ["SECONDARY", "Home Appliance Manufacturing", 78, "Broad portfolio explicitly described."], ["CANDIDATE", "Smart Home Solutions", 62, "Narrative support without revenue evidence."]],
    table: [["PRIMARY", "Air Conditioner Manufacturing", 91, "Largest reported product segment."], ["SECONDARY", "Home Appliance Manufacturing", 68, "Distinct ¥4.88B revenue segment."], ["CANDIDATE", "Intelligent Equipment", 58, "Fast-growing but economically small."]],
    hybrid: [["PRIMARY", "Air Conditioner Manufacturing", 95, "Narrative leadership + revenue dominance."], ["SECONDARY", "Home Appliance Manufacturing", 75, "Product breadth + reported revenue."], ["SECONDARY", "Intelligent Equipment", 65, "Expansion narrative + 42.77% growth."]]
  },
  zh: {
    text: [["主标签", "空调制造", 82, "在文字材料中被反复描述为旗舰产品。"], ["次标签", "生活电器制造", 78, "年报明确描述了丰富的产品组合。"], ["候选", "智能家居解决方案", 62, "具有文字支持，但缺少收入证据。"]],
    table: [["主标签", "空调制造", 91, "在披露的产品板块中收入规模最大。"], ["次标签", "生活电器制造", 68, "形成独立的 48.8 亿元收入板块。"], ["候选", "智能装备制造", 58, "增长较快，但当前经济规模较小。"]],
    hybrid: [["主标签", "空调制造", 95, "行业领导地位与收入主导性相互印证。"], ["次标签", "生活电器制造", 75, "产品广度与实际收入共同支持。"], ["次标签", "智能装备制造", 65, "业务拓展描述与 42.77% 增长共同支持。"]]
  }
};

const decisions = {
  en: { text: "Text identifies rich business meaning, but cannot reliably rank economic materiality.", table: "Tables rank business materiality, but abbreviated rows provide limited semantic context.", hybrid: "Text identifies the product and market position; tables confirm 70.11% of total revenue." },
  zh: { text: "文字能够识别丰富的业务语义，但难以可靠判断各业务的经济重要性。", table: "表格能够衡量业务重要性，但简短的行标题缺少完整语义背景。", hybrid: "文字确认产品及行业地位，表格进一步证明空调贡献了 70.11% 的营业收入。" }
};

const languageSwitch = document.createElement("div");
languageSwitch.className = "language-switch";
languageSwitch.innerHTML = '<button data-language="en">EN</button><button data-language="zh">中文</button>';
document.querySelector(".brand").after(languageSwitch);

document.querySelectorAll("[data-language]").forEach(button => button.addEventListener("click", () => {
  language = button.dataset.language;
  localStorage.setItem("icdm-demo-language", language);
  applyLanguage();
}));

document.querySelectorAll("[data-view]").forEach(button => button.addEventListener("click", () => {
  const view = button.dataset.view;
  document.querySelectorAll(".nav-item").forEach(item => item.classList.toggle("active", item === button));
  document.querySelectorAll(".view").forEach(section => section.classList.remove("active"));
  document.getElementById(`${view}-view`).classList.add("active");
  document.body.classList.toggle("qa-page-active", view === "qa");
  if (view === "qa") window.scrollTo(0, 0);
}));

document.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => {
  activeMode = button.dataset.mode;
  renderLabels();
}));

document.querySelectorAll("[data-channel]").forEach(button => button.addEventListener("click", () => {
  const channel = button.dataset.channel;
  document.querySelectorAll("[data-channel]").forEach(item => item.classList.toggle("active", item === button));
  document.getElementById("text-evidence").classList.toggle("hidden", channel !== "text");
  document.getElementById("table-evidence").classList.toggle("hidden", channel !== "table");
  document.getElementById("source-name").textContent = channel === "text" ? copy[language].textSource : copy[language].tableSource;
}));

const runButton = document.getElementById("run-demo");
runButton.addEventListener("click", () => {
  clearInterval(demoTimer);
  const steps = [...document.querySelectorAll(".step")];
  steps.forEach(step => step.classList.remove("active"));
  let index = 0;
  steps[0].classList.add("active");
  runState = "running";
  runButton.disabled = true;
  runButton.textContent = copy[language].running;
  demoTimer = setInterval(() => {
    index += 1;
    if (index >= steps.length) {
      clearInterval(demoTimer);
      runState = "complete";
      runButton.disabled = false;
      runButton.textContent = copy[language].complete;
      activeMode = "hybrid";
      renderLabels();
      return;
    }
    steps[index].classList.add("active");
  }, 550);
});

document.querySelectorAll("[data-data-tab]").forEach(button => button.addEventListener("click", () => {
  activeDataTab = button.dataset.dataTab;
  document.querySelectorAll("[data-data-tab]").forEach(item => item.classList.toggle("active", item === button));
  renderDataTable();
}));

document.getElementById("data-search").addEventListener("input", event => {
  searchTerm = event.target.value.trim().toLowerCase();
  renderDataTable();
});

function translateLabel(label) { return language === "en" ? (labelNames[label] || label) : label; }
function translateLabelString(value) { return language === "en" ? value.split(" · ").map(translateLabel).join(" · ") : value; }
function includesSearch(values) { return !searchTerm || values.flat(Infinity).join(" ").toLowerCase().includes(searchTerm); }

function renderLabels() {
  const labels = labelSets[language][activeMode];
  const container = document.getElementById("generated-labels");
  document.getElementById("active-mode").textContent = language === "zh" ? `${labels.length} 个标签` : `${labels.length} labels`;
  container.innerHTML = labels.map(label => `<article class="label-card"><div class="label-top"><span>${label[0]}</span><b>${label[2]}%</b></div><h3>${label[1]}</h3><p>${label[3]}</p><div class="confidence"><i style="width:${label[2]}%"></i></div></article>`).join("");
  container.scrollTop = 0;
  document.getElementById("decision-note").textContent = decisions[language][activeMode];
  document.querySelectorAll("[data-mode]").forEach(button => button.classList.toggle("active", button.dataset.mode === activeMode));
}

function renderMetrics() {
  const values = [data.summary.officialCompanies, data.summary.labeledCompanies, data.summary.priceCompanies, data.summary.groups, data.summary.randomTrials.toLocaleString()];
  document.getElementById("metric-strip").innerHTML = values.map((value, index) => `<article><strong>${value}</strong><span>${copy[language].metricLabels[index]}</span></article>`).join("");
  const resultValues = [data.summary.officialComovement, data.summary.withinComovement, data.summary.crossComovement];
  const kinds = ["official", "within", "cross"];
  document.getElementById("result-bars").innerHTML = resultValues.map((value, index) => `<div class="bar-row ${kinds[index]}"><span>${copy[language].bars[index]}</span><div class="bar-track"><i style="width:${value * 205}%"></i></div><strong>${value.toFixed(4)}</strong></div>`).join("");
  const improvement = data.summary.withinComovement - data.summary.officialComovement;
  document.querySelector(".card-title > b").textContent = `+${improvement.toFixed(4)}`;
  document.querySelector(".finding-card > div strong").textContent = Number(data.summary.pValue).toFixed(3);
  const tabCounts = [data.companies.length, data.clusters.length, data.labels.length];
  document.querySelectorAll("[data-data-tab] b").forEach((node, index) => { node.textContent = tabCounts[index]; });
}

function renderDataTable() {
  const c = copy[language];
  const head = document.getElementById("data-thead");
  const body = document.getElementById("data-tbody");
  let rows = [];
  if (activeDataTab === "companies") {
    rows = data.companies.filter(includesSearch);
    head.innerHTML = `<tr>${c.companyHeads.map(value => `<th>${value}</th>`).join("")}</tr>`;
    body.innerHTML = rows.map(row => `<tr><td class="code">${row[0]}</td><td><strong>${row[1]}</strong></td><td><span class="cluster-pill">${row[2]}</span></td><td><div class="tag-list">${row[3].map(tag => `<span>${translateLabel(tag)}</span>`).join("")}</div></td><td>${row[3].length}</td></tr>`).join("");
  } else if (activeDataTab === "clusters") {
    rows = data.clusters.filter(includesSearch);
    head.innerHTML = `<tr>${c.clusterHeads.map(value => `<th>${value}</th>`).join("")}</tr>`;
    body.innerHTML = rows.map(row => `<tr><td><span class="cluster-pill">${row[0]}</span></td><td>${row[1]}</td><td><strong>${row[2] == null ? "—" : row[2].toFixed(4)}</strong></td><td>${translateLabelString(row[3])}</td><td>${row[4]}</td></tr>`).join("");
  } else {
    rows = data.labels.filter(includesSearch);
    head.innerHTML = `<tr>${c.labelHeads.map(value => `<th>${value}</th>`).join("")}</tr>`;
    body.innerHTML = rows.map(row => `<tr><td><strong>${translateLabel(row[0])}</strong></td><td>${row[1]}</td><td class="code">${row[2].toFixed(3)}</td><td><span class="cluster-pill">${c.kept}</span></td></tr>`).join("");
  }
  document.getElementById("row-count").textContent = `${rows.length} ${c.rowsShown}`;
}

function applyLanguage() {
  const c = copy[language];
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-language]").forEach(button => button.classList.toggle("active", button.dataset.language === language));
  const groupLabels = document.querySelectorAll(".sidebar nav small");
  groupLabels[0].textContent = c.demoGroup; groupLabels[1].textContent = c.resultsGroup;
  document.querySelector('[data-view="live"] span').textContent = c.liveNav;
  document.querySelector('[data-view="qa"] span').textContent = c.qaNav;
  document.querySelector('[data-view="risk"] span').textContent = c.riskNav;
  document.querySelector('[data-view="experiments"] span').textContent = c.experimentsNav;
  const footerLinks = document.querySelectorAll(".side-footer a");
  footerLinks[0].textContent = c.paper; footerLinks[1].textContent = c.code;
  document.querySelector(".side-footer span").textContent = c.event;
  document.querySelector("#live-view .title-block h1").textContent = c.liveTitle;
  document.querySelector("#live-view .title-block p").textContent = c.liveSubtitle;
  const qaTitle = document.querySelector("#qa-view .title-block h1");
  const qaSubtitle = document.querySelector("#qa-view .title-block p");
  if (qaTitle) qaTitle.textContent = c.qaTitle;
  if (qaSubtitle) qaSubtitle.textContent = c.qaSubtitle;
  document.querySelector("#risk-view .title-block h1").textContent = c.riskTitle;
  document.querySelector("#risk-view .title-block p").textContent = c.riskSubtitle;
  document.querySelector("#experiments-view .title-block h1").textContent = c.experimentTitle;
  document.querySelector("#experiments-view .title-block p").textContent = c.experimentSubtitle;
  document.querySelector(".case-line strong").textContent = c.company;
  document.querySelector(".case-line small").textContent = c.companyMeta;
  runButton.textContent = runState === "idle" ? c.run : c[runState];
  document.querySelectorAll(".step span").forEach((node, index) => { node.textContent = c.steps[index]; });
  const channelButtons = document.querySelectorAll("[data-channel]");
  channelButtons[0].textContent = c.textEvidence; channelButtons[1].textContent = c.tableEvidence;
  document.querySelector(".mode-tabs > span").textContent = c.mode;
  document.querySelectorAll("[data-mode]").forEach(button => { button.textContent = c.modes[button.dataset.mode]; });
  document.querySelector(".evidence-panel .panel-heading > span").textContent = c.evidence;
  document.querySelector(".label-panel .panel-heading > span").textContent = c.generated;
  document.querySelector(".evidence-foot span").textContent = c.supports;
  const tableHeads = document.querySelectorAll("#table-evidence th");
  tableHeads.forEach((node, index) => { node.textContent = c.tableHeads[index]; });
  document.querySelectorAll("#table-evidence tbody tr td:first-child").forEach((node, index) => { node.textContent = c.tableRows[index]; });
  document.querySelector(".table-note").textContent = c.tableNote;
  const selectedChannel = document.querySelector("[data-channel].active").dataset.channel;
  document.getElementById("source-name").textContent = selectedChannel === "text" ? c.textSource : c.tableSource;
  document.querySelector(".card-title span").textContent = c.chartEyebrow;
  document.querySelector(".card-title h2").textContent = c.chartTitle;
  document.querySelector(".finding-card > span").textContent = c.baselineEyebrow;
  document.querySelector(".finding-card h2").textContent = c.baselineTitle;
  document.querySelector(".finding-card small").textContent = c.pValue;
  document.querySelector(".finding-card p").textContent = c.baselineBody;
  document.querySelector(".data-head span").textContent = c.dataEyebrow;
  document.querySelector(".data-head h2").textContent = c.dataTitle;
  document.getElementById("data-search").placeholder = c.search;
  document.querySelectorAll("[data-data-tab]").forEach((button, index) => { button.childNodes[0].textContent = `${c.dataTabs[index]} `; });
  document.querySelector(".table-footer span:last-child").textContent = c.source;
  renderLabels(); renderMetrics(); renderDataTable();
  if (typeof window.renderFinGLMQA === "function") window.renderFinGLMQA();
  if (typeof window.renderRiskExposure === "function") window.renderRiskExposure();
}

applyLanguage();
