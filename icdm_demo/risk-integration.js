(function () {
  "use strict";

  const API_BASE = "/api/risk";
  const categories = [
    "市场价格", "需求周期", "供应链", "政策监管", "财务流动性",
    "汇率利率", "客户信用/集中度", "运营安全环保", "技术替代", "诉讼合规治理"
  ];
  const categoryEnglish = {
    "市场价格": "Market price",
    "需求周期": "Demand cycle",
    "供应链": "Supply chain",
    "政策监管": "Policy & regulation",
    "财务流动性": "Financial liquidity",
    "汇率利率": "FX & interest rate",
    "客户信用/集中度": "Customer credit / concentration",
    "运营安全环保": "Operations, safety & environment",
    "技术替代": "Technology substitution",
    "诉讼合规治理": "Litigation, compliance & governance"
  };

  const copy = {
    en: {
      serviceTitle: "Evidence-preserving risk exposure",
      unavailable: "Service unavailable",
      coverage: "{available}/{target} companies · {records} exposures",
      catalogTitle: "Choose a company",
      catalogDescription: "2023 annual reports · SW3 430101",
      search: "Search company or stock code",
      loadingCompanies: "Loading companies…",
      companyCount: "Showing {filtered} of {total} companies",
      noCompanies: "No matching company.",
      loadFailed: "Risk artifacts are unavailable.",
      placeholderTitle: "Select one company",
      placeholderCopy: "Its category scores, risk conclusions, and source evidence will appear here.",
      loadingDetail: "Loading company evidence…",
      qualitySufficient: "Evidence sufficient",
      qualityReview: "Review required",
      companyMeta: "{id} · {year} Annual Report · source text retained in Chinese",
      riskFactors: "Risk factors",
      maxExposure: "Maximum exposure",
      evidenceQuotes: "Evidence quotes",
      categoryProfile: "Category profile",
      legend: "0 none · 3 high",
      conclusions: "Risk conclusions and evidence",
      reason: "ASSESSMENT",
      evidence: "ANNUAL-REPORT EVIDENCE",
      interpretation: "Why this supports the exposure",
      mitigants: "DISCLOSED MITIGATION",
      noMitigants: "No explicit mitigation was extracted.",
      exposure: "Exposure {score} / 3",
      detailFailed: "Could not load this company’s risk evidence.",
      riskFactor: "{category} risk factor {index}",
      assessmentSummary: "The retained annual-report evidence supports an exposure score of {score}/3.",
      evidenceInterpretation: "This passage was retained as direct support for the category score."
    },
    zh: {
      serviceTitle: "保留原始证据的企业风险暴露",
      unavailable: "服务不可用",
      coverage: "已覆盖 {available}/{target} 家公司 · {records} 条风险暴露",
      catalogTitle: "选择一家公司",
      catalogDescription: "2023 年年度报告 · 申万三级 430101",
      search: "搜索公司或股票代码",
      loadingCompanies: "正在加载公司…",
      companyCount: "显示 {filtered} / 共 {total} 家公司",
      noCompanies: "没有找到匹配的公司。",
      loadFailed: "风险暴露产物暂时不可用。",
      placeholderTitle: "请选择一家公司",
      placeholderCopy: "这里将展示类别得分、风险结论及其年报原文证据。",
      loadingDetail: "正在加载公司风险证据…",
      qualitySufficient: "证据充分",
      qualityReview: "需要人工复核",
      companyMeta: "{id} · {year} 年年度报告",
      riskFactors: "风险因子",
      maxExposure: "最高暴露等级",
      evidenceQuotes: "原文证据",
      categoryProfile: "风险类别画像",
      legend: "0 无 · 3 高暴露",
      conclusions: "风险结论与证据",
      reason: "风险判断",
      evidence: "年报原文证据",
      interpretation: "证据解释",
      mitigants: "披露的缓释措施",
      noMitigants: "未提取到明确的缓释措施。",
      exposure: "暴露等级 {score} / 3",
      detailFailed: "无法加载该公司的风险证据。",
      riskFactor: "{category}风险因子 {index}",
      assessmentSummary: "年报证据支持该风险 {score}/3 的暴露等级。",
      evidenceInterpretation: "该段内容被保留为风险类别评分的直接依据。"
    }
  };

  const elements = {
    coverage: document.getElementById("risk-coverage"),
    serviceTitle: document.getElementById("risk-service-title"),
    catalogTitle: document.getElementById("risk-catalog-title"),
    catalogDescription: document.getElementById("risk-catalog-description"),
    search: document.getElementById("risk-company-search"),
    count: document.getElementById("risk-company-count"),
    list: document.getElementById("risk-company-list"),
    detail: document.getElementById("risk-detail"),
    placeholder: document.getElementById("risk-detail-placeholder"),
    placeholderTitle: document.getElementById("risk-placeholder-title"),
    placeholderCopy: document.getElementById("risk-placeholder-copy"),
    content: document.getElementById("risk-detail-content"),
    companyName: document.getElementById("risk-company-name"),
    companyMeta: document.getElementById("risk-company-meta"),
    quality: document.getElementById("risk-quality"),
    statCount: document.getElementById("risk-stat-count"),
    statCountLabel: document.getElementById("risk-stat-count-label"),
    statMax: document.getElementById("risk-stat-max"),
    statMaxLabel: document.getElementById("risk-stat-max-label"),
    statEvidence: document.getElementById("risk-stat-evidence"),
    statEvidenceLabel: document.getElementById("risk-stat-evidence-label"),
    categoryTitle: document.getElementById("risk-category-title"),
    legend: document.getElementById("risk-score-legend"),
    categoryGrid: document.getElementById("risk-category-grid"),
    listTitle: document.getElementById("risk-list-title"),
    exposureCount: document.getElementById("risk-exposure-count"),
    exposureList: document.getElementById("risk-exposure-list")
  };

  let companies = [];
  let coverage = null;
  let selectedCompany = null;
  let selectedDetail = null;
  let catalogLoaded = false;
  let catalogFailed = false;
  let requestSequence = 0;
  let loadedLanguage = null;
  let loadingCatalog = false;

  function language() {
    return document.documentElement.lang.toLowerCase().startsWith("zh") ? "zh" : "en";
  }

  function t(key, values) {
    let value = copy[language()][key] || copy.en[key] || key;
    Object.entries(values || {}).forEach(function (entry) {
      value = value.replace("{" + entry[0] + "}", String(entry[1]));
    });
    return value;
  }

  function categoryName(value) {
    return language() === "en" ? (categoryEnglish[value] || value) : value;
  }

  function displayCompanyName(company) {
    return language() === "en" ? (company.display_name || company.company_name) : (company.company_name_original || company.company_name);
  }

  function node(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text !== undefined) value.textContent = text;
    return value;
  }

  async function getJson(url, timeoutMs) {
    const controller = new AbortController();
    const timeout = window.setTimeout(function () { controller.abort(); }, timeoutMs || 12000);
    try {
      const response = await fetch(url, {signal: controller.signal, headers: {"Accept": "application/json"}});
      const data = await response.json();
      if (!response.ok) {
        const error = new Error(data.detail || "REQUEST_FAILED");
        error.status = response.status;
        throw error;
      }
      return data;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function normalized(value) {
    return String(value || "").toLocaleLowerCase("zh-CN").replace(/\s+/g, "");
  }

  function filteredCompanies() {
    const query = normalized(elements.search.value);
    return companies.filter(function (company) {
      return !query || normalized(company.company_id + company.company_name + (company.display_name || "")).includes(query);
    });
  }

  function renderCoverage() {
    elements.coverage.textContent = coverage ? t("coverage", {
      available: coverage.available_company_count,
      target: coverage.target_company_count,
      records: coverage.record_count
    }) : "";
  }

  function renderCompanies() {
    elements.list.replaceChildren();
    if (!catalogLoaded) {
      elements.count.textContent = catalogFailed ? t("loadFailed") : t("loadingCompanies");
      elements.list.appendChild(node("div", "risk-empty", catalogFailed ? t("loadFailed") : t("loadingCompanies")));
      return;
    }
    const filtered = filteredCompanies();
    elements.count.textContent = t("companyCount", {filtered: filtered.length, total: companies.length});
    if (!filtered.length) {
      elements.list.appendChild(node("div", "risk-empty", t("noCompanies")));
      return;
    }
    filtered.forEach(function (company) {
      const active = selectedCompany && selectedCompany.company_id === company.company_id;
      const button = node("button", "risk-company-card" + (active ? " active" : ""));
      button.type = "button";
      button.setAttribute("aria-pressed", active ? "true" : "false");
      button.appendChild(node("strong", "", displayCompanyName(company)));
      button.appendChild(node("small", "", company.company_id.replace(/^A/, "") + " · " + company.report_year));
      button.appendChild(node("span", "", company.risk_count));
      button.addEventListener("click", function () { selectCompany(company); });
      elements.list.appendChild(button);
    });
  }

  function setPlaceholder(title, description) {
    elements.detail.classList.add("empty");
    elements.placeholder.classList.remove("hidden");
    elements.content.classList.add("hidden");
    elements.placeholderTitle.textContent = title;
    elements.placeholderCopy.textContent = description;
  }

  function renderCategories(detail) {
    elements.categoryGrid.replaceChildren();
    const scores = detail.summary.category_scores || {};
    categories.forEach(function (category) {
      const score = Number(scores[category] || 0);
      const card = node("article", "risk-category-item" + (score ? " active" : ""));
      const top = node("div");
      top.appendChild(node("span", "", categoryName(category)));
      top.appendChild(node("b", "", String(score)));
      card.appendChild(top);
      const bar = node("div", "risk-category-bar");
      const fill = node("i");
      fill.style.width = (score / 3 * 100) + "%";
      bar.appendChild(fill);
      card.appendChild(bar);
      elements.categoryGrid.appendChild(card);
    });
  }

  function renderEvidence(exposure) {
    const wrapper = node("div");
    wrapper.appendChild(node("span", "risk-content-label", t("evidence")));
    const list = node("div", "risk-evidence-list");
    (exposure.Evidence || []).forEach(function (evidence) {
      const item = node("article", "risk-evidence-item");
      const quote = node("p", "", evidence.EvidenceQuote);
      item.appendChild(quote);
      if (evidence.Interpretation) {
        item.appendChild(node("small", "", t("interpretation") + " · " + evidence.Interpretation));
      }
      list.appendChild(item);
    });
    wrapper.appendChild(list);
    return wrapper;
  }

  function renderMitigants(exposure) {
    const wrapper = node("div");
    wrapper.appendChild(node("span", "risk-content-label", t("mitigants")));
    const mitigants = Array.isArray(exposure.Mitigants) ? exposure.Mitigants : [];
    if (!mitigants.length) {
      wrapper.appendChild(node("p", "risk-reason", t("noMitigants")));
      return wrapper;
    }
    const list = node("ul", "risk-mitigants");
    mitigants.forEach(function (value) {
      const item = node("li", "", value);
      list.appendChild(item);
    });
    wrapper.appendChild(list);
    return wrapper;
  }

  function renderExposures(detail) {
    elements.exposureList.replaceChildren();
    const exposures = Array.isArray(detail.risk_exposures) ? detail.risk_exposures.slice() : [];
    exposures.sort(function (a, b) {
      return Number(b.ExposureScore) - Number(a.ExposureScore) || String(a.Category).localeCompare(String(b.Category), "zh-CN");
    });
    elements.exposureCount.textContent = String(exposures.length);
    exposures.forEach(function (exposure, index) {
      const score = Number(exposure.ExposureScore || 0);
      const card = node("details", "risk-exposure-card");
      if (index === 0) card.open = true;
      const summary = node("summary");
      summary.appendChild(node("span", "risk-score score-" + score, String(score)));
      const title = node("div", "risk-exposure-title");
      const titleText = exposure.RiskName;
      const category = language() === "en" ? (exposure.CategoryDisplay || categoryName(exposure.Category)) : exposure.Category;
      const subtitle = category + " · " + exposure.Subcategory + " · " + t("exposure", {score: score});
      title.appendChild(node("strong", "", titleText));
      title.appendChild(node("span", "", subtitle));
      summary.appendChild(title);
      summary.appendChild(node("span", "risk-expand-mark", "+"));
      card.appendChild(summary);
      const content = node("div", "risk-exposure-content");
      const reason = node("div");
      reason.appendChild(node("span", "risk-content-label", t("reason")));
      reason.appendChild(node("p", "risk-reason", exposure.Reason));
      content.appendChild(reason);
      content.appendChild(renderEvidence(exposure));
      content.appendChild(renderMitigants(exposure));
      card.appendChild(content);
      elements.exposureList.appendChild(card);
    });
  }

  function renderDetail() {
    if (!selectedDetail) {
      setPlaceholder(t("placeholderTitle"), t("placeholderCopy"));
      return;
    }
    const detail = selectedDetail;
    elements.detail.classList.remove("empty");
    elements.placeholder.classList.add("hidden");
    elements.content.classList.remove("hidden");
    elements.companyName.textContent = displayCompanyName(detail.company);
    elements.companyMeta.textContent = t("companyMeta", {id: detail.company.company_id, year: detail.company.report_year});
    const review = Boolean(detail.quality_flag.NeedHumanReview);
    elements.quality.className = "risk-quality" + (review ? " review" : "");
    elements.quality.textContent = review ? t("qualityReview") : t("qualitySufficient");
    elements.statCount.textContent = detail.summary.risk_count;
    elements.statMax.textContent = detail.summary.max_exposure_score + " / 3";
    elements.statEvidence.textContent = detail.evidence_count;
    renderCategories(detail);
    renderExposures(detail);
  }

  async function selectCompany(company) {
    selectedCompany = company;
    selectedDetail = null;
    renderCompanies();
    setPlaceholder(t("loadingDetail"), displayCompanyName(company) + " · " + company.report_year);
    const sequence = ++requestSequence;
    try {
      const detail = await getJson(API_BASE + "/companies/" + encodeURIComponent(company.company_id) + "?report_year=" + encodeURIComponent(company.report_year) + "&lang=" + encodeURIComponent(language()), 45000);
      if (sequence !== requestSequence) return;
      selectedDetail = detail;
      renderDetail();
    } catch (error) {
      if (sequence !== requestSequence) return;
      setPlaceholder(t("detailFailed"), error.message || t("unavailable"));
    }
  }

  async function loadHealth() {
    try {
      const payload = await getJson(API_BASE + "/health", 6000);
      coverage = payload.coverage || null;
    } catch (error) {
      coverage = coverage || null;
    }
    renderCoverage();
  }

  async function loadCompanies(preferredId) {
    if (loadingCatalog) return;
    loadingCatalog = true;
    const requestedLanguage = language();
    try {
      const payload = await getJson(API_BASE + "/companies?limit=100&lang=" + encodeURIComponent(requestedLanguage), 30000);
      companies = Array.isArray(payload.companies) ? payload.companies : [];
      coverage = payload.coverage || coverage;
      catalogLoaded = true;
      catalogFailed = false;
      loadedLanguage = requestedLanguage;
      const targetId = preferredId || (selectedCompany && selectedCompany.company_id);
      selectedCompany = companies.find(function (company) { return company.company_id === targetId; }) || null;
      renderCompanies();
      renderCoverage();
      if (companies.length) await selectCompany(selectedCompany || companies[0]);
    } catch (error) {
      catalogLoaded = false;
      catalogFailed = true;
      renderCompanies();
    } finally {
      loadingCatalog = false;
      if (loadedLanguage && loadedLanguage !== language()) loadCompanies(selectedCompany && selectedCompany.company_id);
    }
  }
  function renderLanguage() {
    elements.serviceTitle.textContent = t("serviceTitle");
    elements.catalogTitle.textContent = t("catalogTitle");
    elements.catalogDescription.textContent = t("catalogDescription");
    elements.search.placeholder = t("search");
    elements.statCountLabel.textContent = t("riskFactors");
    elements.statMaxLabel.textContent = t("maxExposure");
    elements.statEvidenceLabel.textContent = t("evidenceQuotes");
    elements.categoryTitle.textContent = t("categoryProfile");
    elements.legend.textContent = t("legend");
    elements.listTitle.textContent = t("conclusions");
    if (catalogLoaded && loadedLanguage !== language() && !loadingCatalog) loadCompanies(selectedCompany && selectedCompany.company_id);
    renderCoverage();
    renderCompanies();
    renderDetail();
  }

  elements.search.addEventListener("input", renderCompanies);
  window.renderRiskExposure = renderLanguage;
  renderLanguage();
  Promise.allSettled([loadHealth(), loadCompanies()]);
}());
