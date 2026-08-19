(function () {
  "use strict";

  const API_BASE = "/api/finglmqa";
  const MAX_SELECTED = 3;
  const el = {};
  let documents = [];
  let selection = [];
  let busy = false;
  let serviceReady = false;
  let catalogState = "loading";
  let lastAnswerData = null;
  let lastQuestion = "";
  let canonicalQuestionZh = "";
  let answerProjectionSequence = 0;

  const copy = {
    en: {
      brand: "Annual report QA", brandSub: "Choose up to {max} reports, then ask",
      search: "Search company or stock code", allYears: "All years",
      loadingCatalog: "Loading annual-report catalog…", catalogCount: "Showing {filtered} of {total} reports",
      noReports: "No matching annual report.", catalogFailed: "Could not load the annual-report catalog.",
      choose: "Choose one or more annual reports", chooseMeta: "The reports selected on the left define the scope of this answer.",
      reportMeta: "{code} · {year} Annual Report",
      welcome: "Ask the annual report", welcomeBody: "Answers are generated from retrieved annual-report evidence and an online language model. Supporting citations remain available.",
      answer: "Answer", completed: "Completed", incomplete: "Incomplete", generating: "Generating",
      send: "Send", generatingButton: "Generating…", selectFirst: "Choose a report from the left first",
      placeholder: "For example: What operating risks did {name} face in {year}?",
      revenuePrompt: "What was {name}'s revenue in {year}?",
      riskPrompt: "What operating risks did {name} face in {year}?",
      businessPrompt: "Summarize {name}'s main businesses and operating model in {year}.",
      sources: "View sources ({count})", source: "Source {index}",
      lines: "Lines {start}–{end}",
      tableChannel: "Table evidence", textChannel: "Text evidence", reportSource: "Annual-report source",
      generatingAnswer: "Retrieving annual-report evidence and generating an answer…",
      noAnswer: "No answer was produced. Review the status details below.",
      requestFailed: "Request failed", failedAnswer: "This answer could not be completed. Confirm that both the backend and model services are running.",
      timeout: "The request timed out.", disconnected: "Could not connect to the service.",
      capHint: "Up to {max} annual reports can be compared at once.",
      selectedCount: "{count} annual reports",
      multiPlaceholder: "Ask without naming a company — the selected reports define the scope.",
      progress: "Retrieving {index} of {total} · {name} {year}…",
      consolidating: "Organizing and comparing {total} annual reports…",
      reportFailed: "{name} {year}: retrieval failed",
      consolidateFailed: "The consolidated answer could not be produced."
    },
    zh: {
      brand: "年报问答", brandSub: "最多选择 {max} 份年报后提问",
      search: "搜索公司或股票代码", allYears: "全部年度",
      loadingCatalog: "正在加载年报目录…", catalogCount: "显示 {filtered} / 共 {total} 份年报",
      noReports: "没有匹配的年报。", catalogFailed: "无法加载年报目录。",
      choose: "选择一份或多份年报", chooseMeta: "左侧所选年报决定本次问答范围。",
      reportMeta: "{code} · {year} 年年度报告",
      welcome: "基于年报原文提问", welcomeBody: "答案由检索到的年报证据与在线大模型生成，并可查看引用依据。",
      answer: "回答", completed: "已完成", incomplete: "未完成", generating: "生成中",
      send: "发送", generatingButton: "生成中…", selectFirst: "请先从左侧选择年报",
      placeholder: "例如：{name}{year}年面临哪些经营风险？",
      revenuePrompt: "{name}{year}年营业收入是多少？",
      riskPrompt: "{name}{year}年面临哪些经营风险？",
      businessPrompt: "请简要分析{name}{year}年的主要业务和经营模式。",
      sources: "查看依据（{count}）", source: "依据 {index}",
      lines: "第 {start}–{end} 行",
      tableChannel: "表格证据", textChannel: "文本证据", reportSource: "年报原文",
      generatingAnswer: "正在检索年报原文并生成回答…",
      noAnswer: "未能生成回答，请查看下方状态信息。",
      requestFailed: "请求失败", failedAnswer: "无法完成本次问答，请确认后端服务与模型服务均已启动。",
      timeout: "请求超时。", disconnected: "无法连接服务。",
      capHint: "一次最多对比 {max} 份年报。",
      selectedCount: "已选 {count} 份年报",
      multiPlaceholder: "提问时不必写公司名称，所选年报即为本次问答范围。",
      progress: "正在检索第 {index} / {total} 份 · {name}{year}…",
      consolidating: "正在整合并对比 {total} 份年报…",
      reportFailed: "{name}{year}：检索失败",
      consolidateFailed: "未能生成整合回答。"
    }
  };

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

  function byId(id) { return document.getElementById(id); }
  function documentName(doc) {
    return language() === "en" ? (doc.display_name || doc.stock_name) : doc.stock_name;
  }
  function make(tag, className, value) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (value !== undefined) item.textContent = value;
    return item;
  }

  async function request(url, options, timeout) {
    const controller = new AbortController();
    const timer = window.setTimeout(function () { controller.abort(); }, timeout || 30000);
    try {
      const response = await fetch(url, Object.assign({}, options || {}, {
        signal: controller.signal,
        headers: Object.assign({Accept: "application/json"}, options && options.headers || {})
      }));
      return {response: response, data: await response.json()};
    } finally {
      window.clearTimeout(timer);
    }
  }

  function setReady(ready) {
    serviceReady = ready;
    updateComposer();
  }

  function isSelected(doc) {
    return selection.some(function (item) { return item.document_id === doc.document_id; });
  }

  function reportLabel(doc) {
    return documentName(doc) + " " + doc.report_year;
  }

  function updateComposer() {
    const canAsk = selection.length > 0 && !busy && serviceReady;
    el.question.disabled = !selection.length || busy;
    el.submit.disabled = !canAsk || !el.question.value.trim();
    el.submit.textContent = busy ? t("generatingButton") : t("send");
    el.counter.textContent = el.question.value.length + " / 1000";
  }


function renderCatalog() {
    if (catalogState !== "ready") {
      el.count.textContent = catalogState === "error" ? t("catalogFailed") : t("loadingCatalog");
      el.list.replaceChildren();
      if (catalogState === "error") el.list.appendChild(make("p", "qa-empty", t("catalogFailed")));
      return;
    }
    const query = el.search.value.trim().toLocaleLowerCase();
    const year = el.year.value;
    const filtered = documents.filter(function (doc) {
      const text = [doc.stock_name, doc.display_name, doc.stock_code, doc.company_full, doc.display_company_full, doc.report_year].join(" ").toLocaleLowerCase();
      return (!query || text.includes(query)) && (!year || String(doc.report_year) === year);
    });
    el.count.textContent = t("catalogCount", {filtered: filtered.length, total: documents.length});
    el.list.replaceChildren();
    if (!filtered.length) {
      el.list.appendChild(make("p", "qa-empty", t("noReports")));
      return;
    }
    const atCap = selection.length >= MAX_SELECTED;
    filtered.forEach(function (doc) {
      const active = isSelected(doc);
      const button = make("button", "qa-document" + (active ? " active" : ""));
      button.type = "button";
      button.disabled = busy || (atCap && !active);
      if (atCap && !active) button.title = t("capHint", {max: MAX_SELECTED});
      button.append(
        make("span", "qa-document-mark", active ? "✓" : ""),
        make("strong", "", documentName(doc)),
        make("span", "qa-document-meta", doc.stock_code + " · " + doc.report_year)
      );
      button.addEventListener("click", function () { toggleDocument(doc); });
      el.list.appendChild(button);
    });
  }

  function renderSelected() {
    el.prompts.replaceChildren();
    if (!selection.length) {
      el.selectedName.textContent = t("choose");
      el.selectedMeta.textContent = t("chooseMeta");
      el.question.placeholder = t("selectFirst");
      return;
    }
    if (selection.length > 1) {
      // Presets interpolate one company name, so they are hidden while several
      // reports share a single question.
      el.selectedName.textContent = t("selectedCount", {count: selection.length});
      el.selectedMeta.textContent = selection.map(reportLabel).join(" · ");
      el.question.placeholder = t("multiPlaceholder");
      return;
    }
    const selected = selection[0];
    const displayName = documentName(selected);
    const year = selected.report_year;
    el.selectedName.textContent = displayName;
    el.selectedMeta.textContent = t("reportMeta", {code: selected.stock_code, year: year});
    el.question.placeholder = t("placeholder", {name: displayName, year: year});
    const prompts = [
      {display: t("revenuePrompt", {name: displayName, year: year}), canonical: selected.stock_name + year + "年营业收入是多少？"},
      {display: t("riskPrompt", {name: displayName, year: year}), canonical: selected.stock_name + year + "年面临哪些经营风险？"},
      {display: t("businessPrompt", {name: displayName, year: year}), canonical: "请简要分析" + selected.stock_name + year + "年的主要业务和经营模式。"}
    ];
    prompts.forEach(function (prompt) {
      const button = make("button", "qa-prompt", prompt.display);
      button.type = "button";
      button.dataset.canonicalQuestionZh = prompt.canonical;
      button.addEventListener("click", function () {
        canonicalQuestionZh = prompt.canonical;
        el.question.value = prompt.display;
        updateComposer();
        el.question.focus({preventScroll: true});
      });
      el.prompts.appendChild(button);
    });
  }

  function toggleDocument(doc) {
    if (isSelected(doc)) {
      selection = selection.filter(function (item) { return item.document_id !== doc.document_id; });
    } else if (selection.length < MAX_SELECTED) {
      selection = selection.concat([doc]);
    } else {
      return;
    }
    canonicalQuestionZh = "";
    renderSelected();
    renderCatalog();
    updateComposer();
  }

  function renderCitations(citations) {
    el.sources.replaceChildren();
    if (!citations.length) return;
    const details = make("details", "qa-sources");
    details.appendChild(make("summary", "", t("sources", {count: citations.length})));
    citations.forEach(function (citation, index) {
      const card = make("article", "qa-citation");
      const provenance = citation.provenance || {};
      const path = citation.section_path || provenance.section_path;
      const section = language() === "zh" && Array.isArray(path) && path.length ? " · " + path[path.length - 1] : "";
      const evidenceId = String(provenance.evidence_chunk_id || "");
      const channel = evidenceId.startsWith("tabgr:") ? t("tableChannel") : t("textChannel");
      const reportName = language() === "en" ? t("reportSource") : (citation.document_id || t("reportSource"));
      const lineRange = Array.isArray(provenance.line_range) ? provenance.line_range : [];
      const location = lineRange.length === 2
        ? " · " + t("lines", {start: lineRange[0], end: lineRange[1]})
        : "";
      const excerpt = String(
        citation.excerpt || citation.source_excerpt || citation.supporting_text || citation.quote || ""
      ).trim();
      const scope = citation.report_scope || {};
      const scopeLabel = scope.company ? scope.company + " " + (scope.report_year || "") + " · " : "";
      card.append(
        make("strong", "", t("source", {index: index + 1}) + section),
        make("p", "qa-citation-meta", scopeLabel + channel + " · " + reportName + location)
      );
      if (excerpt) card.appendChild(make("blockquote", "qa-citation-excerpt", excerpt));
      details.appendChild(card);
    });
    el.sources.appendChild(details);
  }

  function localizedStatus(status) {
    if (status === "ok") return t("completed");
    if (status === "generating") return t("generating");
    if (status === "request_failed") return t("requestFailed");
    return status || t("incomplete");
  }


async function syncAnswerLanguage() {
    if (!lastAnswerData || busy || !lastAnswerData._translation) return;
    const desired = language();
    if (lastAnswerData._translation.language === desired) return;
    const sequence = ++answerProjectionSequence;
    try {
      const result = await request("/api/translation/qa", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({lang: desired, payload: lastAnswerData})
      }, 45000);
      if (sequence !== answerProjectionSequence || !result.response.ok) return;
      lastAnswerData = result.data;
      const shownQuestion = desired === "zh"
        ? (lastAnswerData.canonical_question_zh || lastQuestion)
        : (lastAnswerData.display_question || lastQuestion);
      showAnswer(lastAnswerData, shownQuestion, false);
    } catch (error) {
      console.error("QA answer translation failed", error);
    }
  }

  function showAnswer(data, question, remember) {
    if (remember !== false) {
      lastAnswerData = data;
      lastQuestion = question;
    }
    el.empty.classList.add("hidden");
    el.answerWrap.classList.remove("hidden");
    el.userQuestion.textContent = question;
    el.status.textContent = localizedStatus(data.status);
    el.answer.textContent = data.answer || t("noAnswer");
    const notes = [];
    (data.errors || []).forEach(function (x) { notes.push(x.message || String(x)); });
    if (data.status !== "ok") (data.warnings || []).forEach(function (x) { notes.push(x.message || String(x)); });
    el.note.textContent = notes.join(" ");
    el.note.classList.toggle("hidden", !notes.length);
    renderCitations(Array.isArray(data.citations) ? data.citations : []);
    window.requestAnimationFrame(function () { el.conversation.scrollTop = el.conversation.scrollHeight; });
  }


async function askOne(doc, question, canonical, scopePrefix) {
    const result = await request(API_BASE + "/qa", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        question: question,
        display_question: question,
        canonical_question_zh: canonical,
        question_language: language(),
        response_language: language(),
        company: doc.stock_name,
        report_year: doc.report_year,
        // One shared question covers several reports, so the gateway must name
        // each report rather than let a generic phrase like "the selected
        // companies" reach the resolver as a company mention.
        scope_company_prefix: Boolean(scopePrefix)
      })
    }, 190000);
    return result.data;
  }

  function reportResult(doc, data) {
    // The consolidation route restores the original Chinese from the "_original"
    // keys, so the projected payload is forwarded intact minus the bulky trace.
    const payload = Object.assign({}, data);
    delete payload.demo_trace;
    payload.company = doc.stock_name;
    payload.display_company = documentName(doc);
    payload.report_year = doc.report_year;
    payload.document_id = doc.document_id;
    return payload;
  }

  function progressAnswer(text, question) {
    showAnswer({status: "generating", answer: text, citations: []}, question, false);
  }

  async function askMany(reports, question) {
    const results = [];
    const notes = [];
    for (let index = 0; index < reports.length; index += 1) {
      const doc = reports[index];
      progressAnswer(t("progress", {
        index: index + 1, total: reports.length,
        name: documentName(doc), year: doc.report_year
      }), question);
      try {
        results.push(reportResult(doc, await askOne(doc, question, "", true)));
      } catch (error) {
        notes.push(t("reportFailed", {name: documentName(doc), year: doc.report_year}));
        results.push({
          company: doc.stock_name, report_year: doc.report_year, document_id: doc.document_id,
          answer: "", status: "request_failed", citations: []
        });
      }
    }
    progressAnswer(t("consolidating", {total: reports.length}), question);
    const canonical = results
      .map(function (item) { return item.canonical_question_zh; })
      .find(function (value) { return Boolean(value); });
    const consolidated = await request(API_BASE + "/consolidate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        question: canonical || question,
        display_question: question,
        response_language: language(),
        results: results
      })
    }, 90000);
    if (!consolidated.response.ok) throw new Error("consolidation failed");
    const data = consolidated.data;
    if (notes.length) {
      data.warnings = (data.warnings || []).concat(notes.map(function (message) {
        return {message: message};
      }));
    }
    return data;
  }

  async function ask(event) {
    event.preventDefault();
    const question = el.question.value.trim();
    const canonical = canonicalQuestionZh;
    const reports = selection.slice();
    if (!reports.length || !question || busy || el.submit.disabled) return;
    busy = true;
    canonicalQuestionZh = "";
    el.question.value = "";
    updateComposer();
    renderCatalog();
    showAnswer({status: "generating", answer: t("generatingAnswer"), citations: []}, question);
    try {
      const data = reports.length === 1
        ? await askOne(reports[0], question, canonical)
        : await askMany(reports, question);
      showAnswer(data, question);
    } catch (error) {
      showAnswer({
        status: "request_failed",
        answer: reports.length === 1 ? t("failedAnswer") : t("consolidateFailed"),
        errors: [{message: error.name === "AbortError" ? t("timeout") : t("disconnected")}], citations: []
      }, question);
    } finally {
      busy = false;
      updateComposer();
      renderCatalog();
    }
  }

  function renderLanguage() {
    byId("qa-brand-title").textContent = t("brand");
    byId("qa-brand-subtitle").textContent = t("brandSub", {max: MAX_SELECTED});
    el.search.placeholder = t("search");
    el.year.options[0].textContent = t("allYears");
    Array.from(el.year.options).slice(1).forEach(function (option) {
      option.textContent = language() === "zh" ? option.value + " 年" : option.value;
    });
    byId("qa-empty-title").textContent = t("welcome");
    byId("qa-empty-copy").textContent = t("welcomeBody");
    byId("qa-answer-label").textContent = t("answer");
    renderSelected();
    renderCatalog();
    if (lastAnswerData) {
      const shownQuestion = language() === "zh" ? (lastAnswerData.canonical_question_zh || lastQuestion) : (lastAnswerData.display_question || lastQuestion);
      showAnswer(lastAnswerData, shownQuestion, false);
      syncAnswerLanguage();
    }
    updateComposer();
  }


async function boot() {
    const view = byId("qa-view");
    view.innerHTML = [
      '<div class="qa-simple-shell">',
      '<aside class="qa-simple-catalog"><div class="qa-simple-brand"><div><b id="qa-brand-title"></b><small id="qa-brand-subtitle"></small></div></div><div class="qa-simple-filters"><input id="qa-document-search" type="search"><select id="qa-document-year"><option value=""></option></select></div><p id="qa-document-count" class="qa-simple-count"></p><div id="qa-document-list" class="qa-simple-list"></div></aside>',
      '<section class="qa-simple-chat"><header class="qa-simple-header"><div><strong id="qa-selected-name"></strong><small id="qa-selected-meta"></small></div></header>',
      '<main id="qa-conversation" class="qa-simple-conversation"><section id="qa-empty-state" class="qa-simple-empty"><h1 id="qa-empty-title"></h1><p id="qa-empty-copy"></p></section><section id="qa-answer-wrap" class="qa-simple-answer hidden"><p id="qa-user-question" class="qa-simple-user-question"></p><div class="qa-simple-answer-head"><span>DO</span><b id="qa-answer-label"></b><small id="qa-result-status"></small></div><p id="qa-answer-text" class="qa-simple-answer-text"></p><p id="qa-message" class="qa-simple-note hidden"></p><div id="qa-citations"></div></section></main>',
      '<footer class="qa-simple-composer"><div id="qa-prompts" class="qa-simple-prompts"></div><form id="qa-form" class="qa-simple-form"><textarea id="qa-question" rows="1" maxlength="1000" disabled></textarea><button id="qa-submit" type="submit" disabled></button></form><small id="qa-character-count">0 / 1000</small></footer></section>',
      '</div>'
    ].join("");
    Object.assign(el, {
      search: byId("qa-document-search"), year: byId("qa-document-year"),
      count: byId("qa-document-count"), list: byId("qa-document-list"), selectedName: byId("qa-selected-name"),
      selectedMeta: byId("qa-selected-meta"), prompts: byId("qa-prompts"), question: byId("qa-question"),
      counter: byId("qa-character-count"), submit: byId("qa-submit"), form: byId("qa-form"),
      conversation: byId("qa-conversation"), empty: byId("qa-empty-state"), answerWrap: byId("qa-answer-wrap"),
      userQuestion: byId("qa-user-question"), status: byId("qa-result-status"), answer: byId("qa-answer-text"),
      note: byId("qa-message"), sources: byId("qa-citations")
    });
    el.search.addEventListener("input", renderCatalog);
    el.year.addEventListener("change", renderCatalog);
    el.question.addEventListener("input", function (event) {
      if (event.isTrusted) canonicalQuestionZh = "";
      updateComposer();
    });
    el.question.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        el.form.requestSubmit();
      }
    });
    el.form.addEventListener("submit", ask);
    renderLanguage();
    try {
      const health = await request(API_BASE + "/health", {}, 7000);
      setReady(health.response.ok && health.data.ready === true);
    } catch (error) {
      setReady(false);
    }
    try {
      const catalog = await request(API_BASE + "/documents?lang=en", {}, 30000);
      documents = Array.isArray(catalog.data.documents) ? catalog.data.documents : [];
      const years = Array.from(new Set(documents.map(function (doc) { return String(doc.report_year); }))).sort().reverse();
      years.forEach(function (year) {
        const option = make("option", "", language() === "zh" ? year + " 年" : year);
        option.value = year;
        el.year.appendChild(option);
      });
      catalogState = "ready";
      renderCatalog();
    } catch (error) {
      catalogState = "error";
      renderCatalog();
    }
  }

  window.renderFinGLMQA = renderLanguage;


  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, {once: true});
  } else {
    boot();
  }
}());
