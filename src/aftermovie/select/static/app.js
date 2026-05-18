// Aftermovie clip-selection UI. Vanilla JS, no deps. Talks to /api/*.
(() => {
  "use strict";

  const KIND_BADGE = { video: "V", still: "S", live_photo: "L" };

  /** @type {Array<{path:string,name:string,kind:string,thumb_url:string,selected:boolean}>} */
  let sources = [];
  /** @type {Map<string,string>} path -> thumb_url, used to resolve thumbs for plan entries */
  const sourceThumbByPath = new Map();
  /** @type {Set<string>} */
  const excluded = new Set();
  const themes = new Map();
  let saveTimer = null;
  let pollTimer = null;
  let currentJobId = null;
  let pollFailures = 0;
  let currentIsPreview = false;

  const $ = (sel) => document.querySelector(sel);

  const els = {
    grid: $("#grid"),
    status: $("#status"),
    counter: $("#counter"),
    length: $("#length"),
    pace: $("#pace"),
    lut: $("#lut"),
    theme: $("#theme"),
    transitions: $("#transitions"),
    audioMix: $("#audio-mix"),
    aspect: $("#aspect"),
    sourceCap: $("#source-cap"),
    speedRamp: $("#speed-ramp"),
    reframe: $("#reframe"),
    keepBursts: $("#keep-bursts"),
    selectAll: $("#select-all"),
    deselectAll: $("#deselect-all"),
    render: $("#render"),
    renderPreviewBtn: $("#render-preview-btn"),
    previewStatus: $("#preview-status"),
    previewBadge: $("#preview-badge"),
    previewMessage: $("#preview-message"),
    cacheIndicator: $("#cache-indicator"),
    modal: $("#modal"),
    modalState: $("#modal-state"),
    modalLog: $("#modal-log"),
    modalResult: $("#modal-result"),
    modalPath: $("#modal-path"),
    modalClose: $("#modal-close"),
    copyPath: $("#copy-path"),
    revealLink: $("#reveal-link"),
    template: $("#cell-template"),
    planPanel: $("#plan-panel"),
    planTimeline: $("#plan-timeline"),
    planEmpty: $("#plan-empty"),
    planMeta: $("#plan-meta"),
    planTileTemplate: $("#plan-tile-template"),
  };

  // -------- API helpers --------
  async function api(method, url, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(url, opts);
    if (!res.ok) throw new Error(`${method} ${url} -> ${res.status}`);
    return res.json();
  }

  async function loadSources() {
    setStatus("Loading sources…");
    try {
      sources = await fetch("/api/sources").then((r) => {
        if (!r.ok) throw new Error(`GET /api/sources -> ${r.status}`);
        return r.json();
      });
      excluded.clear();
      sourceThumbByPath.clear();
      for (const s of sources) {
        if (s.selected === false) excluded.add(s.path);
        if (s.path && s.thumb_url) sourceThumbByPath.set(s.path, s.thumb_url);
      }
      renderGrid();
      updateCounter();
      if (!sources.length) setStatus("No sources found. Drop clips into the watched folder and reload.");
      else setStatus("", true);
    } catch (err) {
      setStatus(`Failed to load sources: ${err.message}`, false, true);
    }
  }

  async function loadOptions() {
    try {
      const opts = await fetch("/api/options").then((r) => {
        if (!r.ok) throw new Error(`GET /api/options -> ${r.status}`);
        return r.json();
      });
      fillSelect(els.lut, [
        { value: "", label: "default" },
        ...(opts.luts || []).map((l) => ({ value: l.name, label: l.name })),
      ]);
      themes.clear();
      for (const t of opts.themes || []) themes.set(t.name, t);
      fillSelect(els.theme, [
        { value: "", label: "custom" },
        ...(opts.themes || []).map((t) => ({ value: t.name, label: t.name })),
      ]);
    } catch (err) {
      console.warn("Failed to load render options:", err);
    }
  }

  function fillSelect(select, items) {
    const current = select.value;
    select.replaceChildren();
    for (const item of items) {
      const opt = document.createElement("option");
      opt.value = item.value;
      opt.textContent = item.label;
      select.appendChild(opt);
    }
    if (items.some((item) => item.value === current)) select.value = current;
  }

  function setStatus(text, hide = false, error = false) {
    els.status.textContent = text;
    els.status.classList.toggle("hidden", hide);
    els.status.classList.toggle("error", error);
  }

  // -------- Rendering --------
  function renderGrid() {
    els.grid.replaceChildren();
    const frag = document.createDocumentFragment();
    for (const s of sources) frag.appendChild(buildCell(s));
    els.grid.appendChild(frag);
  }

  function buildCell(src) {
    const node = els.template.content.firstElementChild.cloneNode(true);
    const img = node.querySelector(".thumb");
    const badge = node.querySelector(".badge");
    const name = node.querySelector(".name");

    img.src = src.thumb_url;
    img.alt = src.name;
    name.textContent = src.name;
    name.title = src.path;

    const code = KIND_BADGE[src.kind] || "?";
    badge.textContent = code;
    badge.classList.add(`kind-${code}`);

    const isSelected = !excluded.has(src.path);
    setCellSelected(node, isSelected);
    node.dataset.path = src.path;

    node.addEventListener("click", () => toggle(node, src.path));
    node.addEventListener("keydown", (e) => {
      if (e.key === " " || e.key === "Enter") {
        e.preventDefault();
        toggle(node, src.path);
      }
    });
    return node;
  }

  function setCellSelected(cell, isSelected) {
    cell.classList.toggle("selected", isSelected);
    cell.setAttribute("aria-pressed", String(isSelected));
  }

  function toggle(cell, path) {
    if (excluded.has(path)) excluded.delete(path);
    else excluded.add(path);
    setCellSelected(cell, !excluded.has(path));
    updateCounter();
    scheduleSave();
  }

  function updateCounter() {
    const total = sources.length;
    const sel = total - excluded.size;
    els.counter.textContent = `${sel} of ${total} selected`;
  }

  // -------- Selection persistence (debounced) --------
  function scheduleSave() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(saveSelection, 500);
  }

  async function saveSelection() {
    try {
      await api("POST", "/api/selection", { excluded: [...excluded] });
    } catch (err) {
      console.warn("Failed to persist selection:", err);
    }
  }

  // -------- Bulk actions --------
  function selectAll() {
    excluded.clear();
    for (const cell of els.grid.children) setCellSelected(cell, true);
    updateCounter();
    scheduleSave();
  }
  function deselectAll() {
    excluded.clear();
    for (const s of sources) excluded.add(s.path);
    for (const cell of els.grid.children) setCellSelected(cell, false);
    updateCounter();
    scheduleSave();
  }

  // -------- Render flow --------
  // Both buttons share this path; `isPreview=true` adds `preview: true` to the
  // POST body which the server forwards to AutoOpts.preview (quarter-res, no
  // LUT, no reframe). The preview status badge is surfaced while a preview
  // job is in flight; the final-render status uses the existing modal only.
  async function startRender(isPreview = false) {
    const body = compactPayload({
      excluded: [...excluded],
      max_length: clampInt(els.length.value, 10, 600, 90),
      pace: els.pace.value || "auto",
      lut: els.lut.value || null,
      theme: els.theme.value || null,
      transitions: els.transitions.value || null,
      audio_mix: els.audioMix.value || null,
      aspect: els.aspect.value || null,
      source_cap: clampInt(els.sourceCap.value, 1, 5, 1),
      no_speed_ramp: !els.speedRamp.checked,
      no_reframe: !els.reframe.checked,
      burst_window_s: els.keepBursts.checked ? 0 : 3,
    });
    if (isPreview) body.preview = true;
    openModal();
    setModalState("starting…");
    appendLog(renderSummary(body));
    els.render.disabled = true;
    els.renderPreviewBtn.disabled = true;
    currentIsPreview = isPreview;
    showPreviewStatus(isPreview);
    try {
      const { job_id } = await api("POST", "/api/render", body);
      currentJobId = job_id;
      pollFailures = 0;
      appendLog(`job_id = ${job_id}`);
      pollStatus();
    } catch (err) {
      setModalState(`error: ${err.message}`, "error");
      els.render.disabled = false;
      els.renderPreviewBtn.disabled = false;
      hidePreviewStatus();
    }
  }

  function renderPreview() {
    return startRender(true);
  }

  function showPreviewStatus(isPreview) {
    if (!isPreview) {
      hidePreviewStatus();
      return;
    }
    els.previewStatus.classList.remove("hidden");
    els.previewMessage.textContent =
      "Preview render: quarter-res, no LUT — fast iteration.";
    // The cache indicator stays hidden until /api/status/<job> reports it.
    els.cacheIndicator.classList.add("hidden");
  }

  function hidePreviewStatus() {
    els.previewStatus.classList.add("hidden");
    els.cacheIndicator.classList.add("hidden");
  }

  function clampInt(v, min, max, dflt) {
    const n = parseInt(v, 10);
    if (!Number.isFinite(n)) return dflt;
    return Math.max(min, Math.min(max, n));
  }

  function compactPayload(body) {
    return Object.fromEntries(Object.entries(body).filter(([, value]) => value !== null && value !== ""));
  }

  function renderSummary(body) {
    const bits = [
      `excluded=${body.excluded.length}`,
      `length=${body.max_length}s`,
      `pace=${body.pace}`,
      `lut=${body.lut || "default"}`,
      `theme=${body.theme || "custom"}`,
      `transitions=${body.transitions}`,
      `audio=${body.audio_mix}`,
      `aspect=${body.aspect}`,
      `reuse=${body.source_cap}`,
      `nearby=${body.burst_window_s === 0 ? "keep" : "filter"}`,
    ];
    if (body.no_speed_ramp) bits.push("no_speed_ramp");
    if (body.no_reframe) bits.push("no_reframe");
    if (body.preview) bits.push("preview");
    return `POST /api/render  ${bits.join("  ")}`;
  }

  function applyTheme() {
    const theme = themes.get(els.theme.value);
    if (!theme) return;
    if (theme.lut) els.lut.value = theme.lut;
    if (theme.pace) els.pace.value = theme.pace;
    if (theme.transitions) els.transitions.value = theme.transitions;
    if (theme.audio_mix) els.audioMix.value = theme.audio_mix;
    els.speedRamp.checked = theme.no_speed_ramp !== true;
  }

  async function pollStatus() {
    if (!currentJobId) return;
    try {
      const s = await fetch(`/api/status/${encodeURIComponent(currentJobId)}`).then((r) => {
        if (!r.ok) throw new Error(`GET /api/status -> ${r.status}`);
        return r.json();
      });
      pollFailures = 0;
      setModalState(s.state, s.state === "done" ? "done" : s.state === "error" ? "error" : "");
      if (s.log_tail) replaceLog(s.log_tail);
      // Cache-hit indicator is defensive: only lights up if the field is
      // present and truthy. Issue #1 may or may not add it — silently no-op
      // until it lands.
      if (currentIsPreview && s && s.cache_hit) {
        els.cacheIndicator.classList.remove("hidden");
      }
      if (s.state === "done") {
        showResult(s.output_path || "");
        els.render.disabled = false;
        els.renderPreviewBtn.disabled = false;
        loadPlan();
        return;
      }
      if (s.state === "error") {
        els.render.disabled = false;
        els.renderPreviewBtn.disabled = false;
        hidePreviewStatus();
        return;
      }
      pollTimer = setTimeout(pollStatus, 1500);
    } catch (err) {
      pollFailures += 1;
      if (pollFailures === 1 || pollFailures % 5 === 0) {
        appendLog(`status poll failed (${pollFailures}): ${err.message}`);
      }
      setModalState(`server disconnected; retrying (${pollFailures})`, "error");
      if (pollFailures >= 10) {
        appendLog("status polling stopped. Reopen the current GUI URL and start the render again.");
        els.render.disabled = false;
        els.renderPreviewBtn.disabled = false;
        hidePreviewStatus();
        currentJobId = null;
        return;
      }
      pollTimer = setTimeout(pollStatus, 2500);
    }
  }

  function appendLog(line) {
    els.modalLog.textContent += (els.modalLog.textContent ? "\n" : "") + line;
    els.modalLog.scrollTop = els.modalLog.scrollHeight;
  }
  function replaceLog(text) {
    els.modalLog.textContent = text;
    els.modalLog.scrollTop = els.modalLog.scrollHeight;
  }
  function setModalState(text, cls = "") {
    els.modalState.textContent = text;
    els.modalState.className = `modal-state${cls ? " " + cls : ""}`;
  }

  function openModal() {
    els.modal.classList.remove("hidden");
    els.modalResult.classList.add("hidden");
    els.modalLog.textContent = "";
  }
  function closeModal() {
    els.modal.classList.add("hidden");
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = null;
  }

  function showResult(outputPath) {
    if (!outputPath) return;
    els.modalPath.textContent = outputPath;
    // file:// link gives Finder a hint; browsers won't run shell open but it's the best portable handle.
    els.revealLink.href = `file://${outputPath}`;
    els.modalResult.classList.remove("hidden");
  }

  async function copyPath() {
    const path = els.modalPath.textContent;
    if (!path) return;
    try {
      await navigator.clipboard.writeText(path);
      els.copyPath.textContent = "Copied!";
      setTimeout(() => (els.copyPath.textContent = "Copy path"), 1200);
    } catch {
      els.copyPath.textContent = "Copy failed";
    }
  }

  // -------- Plan timeline (read-only) --------
  // Loads the most recent Plan from /api/plan and renders one tile per Entry.
  // The endpoint may not exist yet (404) or may return an empty plan; in either
  // case we hide the panel with a soft message instead of failing loudly.
  async function loadPlan() {
    try {
      const res = await fetch("/api/plan", { headers: { "Accept": "application/json" } });
      if (res.status === 404) {
        showPlanEmpty();
        return;
      }
      if (!res.ok) {
        showPlanEmpty();
        return;
      }
      const data = await res.json();
      const entries = extractPlanEntries(data);
      if (!entries.length) {
        showPlanEmpty();
        return;
      }
      renderPlanTimeline(entries, data);
    } catch (err) {
      console.warn("Failed to load plan:", err);
      showPlanEmpty();
    }
  }

  function extractPlanEntries(data) {
    if (!data) return [];
    if (Array.isArray(data)) return data;
    if (Array.isArray(data.entries)) return data.entries;
    return [];
  }

  function showPlanEmpty() {
    els.planPanel.classList.remove("hidden");
    els.planTimeline.replaceChildren();
    els.planEmpty.classList.remove("hidden");
    els.planMeta.textContent = "";
  }

  function renderPlanTimeline(entries, plan) {
    els.planPanel.classList.remove("hidden");
    els.planEmpty.classList.add("hidden");
    els.planTimeline.replaceChildren();
    const frag = document.createDocumentFragment();
    for (const entry of entries) frag.appendChild(buildPlanTile(entry));
    els.planTimeline.appendChild(frag);
    const totalDur = entries.reduce(
      (sum, e) => sum + (Number(e.out_duration_s) || (Number(e.end_s) - Number(e.start_s)) || 0),
      0,
    );
    const bits = [`${entries.length} entr${entries.length === 1 ? "y" : "ies"}`];
    if (totalDur > 0) bits.push(`${totalDur.toFixed(1)}s total`);
    if (plan && plan.target_length_s) bits.push(`target ${Number(plan.target_length_s).toFixed(0)}s`);
    els.planMeta.textContent = bits.join(" · ");
  }

  function buildPlanTile(entry) {
    const node = els.planTileTemplate.content.firstElementChild.cloneNode(true);
    const img = node.querySelector(".plan-thumb");
    const transitionBadge = node.querySelector(".plan-transition");
    const durationEl = node.querySelector(".plan-duration");
    const nameEl = node.querySelector(".plan-name");
    const reasonsEl = node.querySelector(".plan-reasons");
    const audioEl = node.querySelector(".plan-audio");
    const audioBar = node.querySelector(".plan-audio-bar");

    // Thumbnail — reuse the source's /thumbs/<key>.jpg when we know it.
    // The plan may include its own `thumb_url`; otherwise look up by source
    // path in the map we built from /api/sources. Fall back to a placeholder.
    const thumbUrl = entry.thumb_url || sourceThumbByPath.get(entry.source) || "";
    const baseName = basename(entry.source || "");
    if (thumbUrl) {
      img.src = thumbUrl;
      img.alt = baseName;
    } else {
      img.classList.add("placeholder");
      img.removeAttribute("src");
      img.alt = "no thumbnail";
    }

    // Filename + tooltip with full source path.
    nameEl.textContent = baseName;
    nameEl.title = entry.source || "";

    // Entry duration — prefer out_duration_s, fall back to end_s - start_s.
    const dur = Number(entry.out_duration_s);
    const computed = Number(entry.end_s) - Number(entry.start_s);
    const shown = Number.isFinite(dur) && dur > 0 ? dur : (Number.isFinite(computed) ? computed : 0);
    durationEl.textContent = `${shown.toFixed(1)}s`;

    // Transition kind badge (only if transition_in is present).
    const t = entry.transition_in;
    if (t && typeof t === "object" && t.kind) {
      const kind = String(t.kind);
      transitionBadge.textContent = kind;
      transitionBadge.classList.remove("hidden");
      transitionBadge.classList.add(`kind-${kind}`);
    }

    // Score reason pills.
    if (Array.isArray(entry.reasons)) {
      for (const reason of entry.reasons) {
        const pill = document.createElement("span");
        pill.className = "plan-reason";
        pill.textContent = String(reason);
        reasonsEl.appendChild(pill);
      }
    }

    // Audio interest bar (0..1).
    if (entry.audio_interest != null && Number.isFinite(Number(entry.audio_interest))) {
      const pct = Math.max(0, Math.min(1, Number(entry.audio_interest))) * 100;
      audioBar.style.width = `${pct.toFixed(0)}%`;
      audioEl.classList.remove("hidden");
      audioEl.setAttribute("title", `audio interest ${Number(entry.audio_interest).toFixed(2)}`);
    }

    return node;
  }

  function basename(p) {
    if (!p) return "";
    const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
    return i >= 0 ? p.slice(i + 1) : p;
  }

  // -------- Wire-up --------
  els.selectAll.addEventListener("click", selectAll);
  els.deselectAll.addEventListener("click", deselectAll);
  els.render.addEventListener("click", () => startRender(false));
  els.renderPreviewBtn.addEventListener("click", renderPreview);
  els.theme.addEventListener("change", applyTheme);
  els.modalClose.addEventListener("click", closeModal);
  els.copyPath.addEventListener("click", copyPath);
  els.modal.addEventListener("click", (e) => { if (e.target === els.modal) closeModal(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.modal.classList.contains("hidden")) closeModal();
  });

  loadOptions();
  // Load sources first so the plan can resolve thumb URLs via path → /thumbs/<key>.jpg.
  loadSources().finally(loadPlan);
})();
