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
    lengthMode: $("#length-mode"),
    lengthCustom: $("#length-custom"),
    lengthCustomSuffix: $("#length-custom-suffix"),
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
    importPanel: $("#import-panel"),
    importSince: $("#import-since"),
    importUntil: $("#import-until"),
    importSources: $("#import-sources"),
    importBtn: $("#import-btn"),
    dryRunBtn: $("#dry-run-btn"),
    importStatus: $("#import-status"),
    importStateBadge: $("#import-state-badge"),
    importProgress: $("#import-progress"),
    importError: $("#import-error"),
    importSuccess: $("#import-success"),
    importSuccessMsg: $("#import-success .import-success-msg"),
    importDest: $("#import-dest"),
    importUseFolder: $("#import-use-folder"),
    importLog: $("#import-log"),
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

    // Wire the favorite/ban buttons (defined in the preferences section
    // below). stopPropagation so clicking a pref button doesn't also flip
    // the cell's selection state via the cell-level click handler above.
    const favBtn = node.querySelector(".pref-fav");
    const banBtn = node.querySelector(".pref-ban");
    if (favBtn) {
      favBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        togglePreference(node, src.path, "favorited");
      });
    }
    if (banBtn) {
      banBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        togglePreference(node, src.path, "banned");
      });
    }
    applyPreferenceClasses(node, src.path);
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
      max_length: resolveMaxLength(),
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

  // Translate the length dropdown + custom input into the `max_length` payload
  // field that the planner consumes. `full` yields null so `compactPayload`
  // drops the key, and the backend's `cfg.max_length is None` path picks the
  // full Song duration (Phase C1).
  function resolveMaxLength() {
    const mode = (els.lengthMode && els.lengthMode.value) || "full";
    if (mode === "full") return null;
    if (mode === "custom") {
      return clampInt(els.lengthCustom && els.lengthCustom.value, 10, 600, 90);
    }
    // Preset second-counts ("90", "60", "30") come through as numeric strings.
    const n = parseInt(mode, 10);
    return Number.isFinite(n) ? n : null;
  }

  // Reveal the numeric input + suffix only when the user picks "Custom…".
  function syncLengthCustomVisibility() {
    const isCustom = els.lengthMode && els.lengthMode.value === "custom";
    if (els.lengthCustom) els.lengthCustom.classList.toggle("hidden", !isCustom);
    if (els.lengthCustomSuffix) {
      els.lengthCustomSuffix.classList.toggle("hidden", !isCustom);
    }
  }

  function compactPayload(body) {
    return Object.fromEntries(Object.entries(body).filter(([, value]) => value !== null && value !== ""));
  }

  function renderSummary(body) {
    const lengthLabel = body.max_length == null ? "full song" : `${body.max_length}s`;
    const bits = [
      `excluded=${body.excluded.length}`,
      `length=${lengthLabel}`,
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

  // -------- Preferences (favorite / ban) --------
  // Per-folder preferences sidecar (.aftermovie-preferences.json). Persisted
  // through POST /api/preferences (debounced) and hydrated on load through
  // GET /api/preferences. Favoriting a clip boosts every Candidate from it
  // by +2.0 in the scorer; banning a clip drops it from the candidate pool
  // entirely. Both controls are independent of the selection toggle.
  /** @type {Set<string>} */
  const favorited = new Set();
  /** @type {Set<string>} */
  const banned = new Set();
  /** @type {string[]} reserved — pinning isn't wired through the scorer yet */
  let pinnedEntries = [];
  let prefsSaveTimer = null;

  async function loadPreferences() {
    try {
      const data = await fetch("/api/preferences").then((r) => {
        if (!r.ok) throw new Error(`GET /api/preferences -> ${r.status}`);
        return r.json();
      });
      favorited.clear();
      banned.clear();
      for (const p of data.favorited || []) favorited.add(p);
      for (const p of data.banned || []) banned.add(p);
      pinnedEntries = Array.isArray(data.pinned_entries) ? data.pinned_entries : [];
      // Re-apply classes to whatever cells already exist (loadSources may
      // have raced ahead). Cheap: handful of attribute writes per cell.
      for (const cell of els.grid.children) {
        const p = cell.dataset.path;
        if (p) applyPreferenceClasses(cell, p);
      }
    } catch (err) {
      console.warn("Failed to load preferences:", err);
    }
  }

  function togglePreference(cell, path, kind) {
    const target = kind === "favorited" ? favorited : banned;
    const other = kind === "favorited" ? banned : favorited;
    if (target.has(path)) {
      target.delete(path);
    } else {
      target.add(path);
      // Favorite and ban are mutually exclusive — toggling one clears the
      // other so the visual state can't say "favorite AND banned".
      other.delete(path);
    }
    applyPreferenceClasses(cell, path);
    schedulePreferenceSave();
  }

  function applyPreferenceClasses(cell, path) {
    const isFav = favorited.has(path);
    const isBan = banned.has(path);
    cell.classList.toggle("is-favorited", isFav);
    cell.classList.toggle("is-banned", isBan);
    const favBtn = cell.querySelector(".pref-fav");
    const banBtn = cell.querySelector(".pref-ban");
    if (favBtn) favBtn.setAttribute("aria-pressed", String(isFav));
    if (banBtn) banBtn.setAttribute("aria-pressed", String(isBan));
  }

  function schedulePreferenceSave() {
    if (prefsSaveTimer) clearTimeout(prefsSaveTimer);
    prefsSaveTimer = setTimeout(savePreferences, 300);
  }

  async function savePreferences() {
    try {
      await api("POST", "/api/preferences", {
        favorited: [...favorited],
        banned: [...banned],
        pinned_entries: pinnedEntries,
      });
    } catch (err) {
      console.warn("Failed to persist preferences:", err);
    }
  }

  // -------- Import from devices --------
  // Talks to a parallel agent's backend (aftermovie.import_sources via
  // SelectionService). On boot, GET /api/import-sources populates the
  // checkbox list. POST /api/import kicks a job; we then poll
  // /api/import-status/<job_id> every ~750ms until state ∈ {done, error}.
  // Robustness rules:
  //   - /api/import-sources 404 → hide the panel silently (backend not deployed).
  //   - poll network failure → stop polling, show error message.
  /** @type {Array<{name:string,label:string,available:boolean}>} */
  let importSources = [];
  let importJobId = null;
  let importPollTimer = null;

  async function loadImportSources() {
    try {
      const res = await fetch("/api/import-sources");
      if (res.status === 404) {
        // Backend not deployed yet — panel stays hidden.
        els.importPanel.classList.add("hidden");
        return;
      }
      if (!res.ok) throw new Error(`GET /api/import-sources -> ${res.status}`);
      importSources = await res.json();
      els.importPanel.classList.remove("hidden");
      seedImportDates();
      renderImportSources();
      updateImportButtonState();
    } catch (err) {
      console.warn("Failed to load import sources:", err);
      els.importPanel.classList.add("hidden");
    }
  }

  function seedImportDates() {
    // Default window: today minus 7 days → today (YYYY-MM-DD, local).
    const today = new Date();
    const past = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
    if (!els.importSince.value) els.importSince.value = toDateInput(past);
    if (!els.importUntil.value) els.importUntil.value = toDateInput(today);
  }

  function toDateInput(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  }

  function renderImportSources() {
    els.importSources.replaceChildren();
    const frag = document.createDocumentFragment();
    for (const src of importSources) {
      const label = document.createElement("label");
      label.className = "import-source";
      if (src.available === false) label.classList.add("unavailable");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = src.name;
      cb.dataset.source = src.name;
      if (src.available === false) {
        cb.disabled = true;
        label.title = "source not detected — plug it in / install osxphotos";
      }
      cb.addEventListener("change", updateImportButtonState);
      label.appendChild(cb);
      const text = document.createElement("span");
      text.textContent = src.label || src.name;
      label.appendChild(text);
      if (src.available === false) {
        const badge = document.createElement("span");
        badge.className = "import-source-badge";
        badge.textContent = "unavailable";
        label.appendChild(badge);
      }
      frag.appendChild(label);
    }
    els.importSources.appendChild(frag);
  }

  function getCheckedImportSources() {
    return [...els.importSources.querySelectorAll("input[type=checkbox]:checked")].map(
      (cb) => cb.value,
    );
  }

  function updateImportButtonState() {
    const any = getCheckedImportSources().length > 0;
    const running = !!importJobId;
    els.importBtn.disabled = !any || running;
    els.dryRunBtn.disabled = !any || running;
  }

  async function startImport(dryRun = false) {
    const selected = getCheckedImportSources();
    if (!selected.length) return;
    const body = {
      since: els.importSince.value || null,
      until: els.importUntil.value || null,
      sources: selected,
      dest_parent: null,
      dry_run: !!dryRun,
    };
    resetImportStatus();
    els.importStatus.classList.remove("hidden");
    setImportState("starting", "running");
    try {
      const res = await api("POST", "/api/import", body);
      importJobId = res.job_id;
      if (res.dest_folder) {
        els.importDest.textContent = res.dest_folder;
      }
      updateImportButtonState();
      pollImportStatus();
    } catch (err) {
      setImportState("error", "error");
      showImportError(err.message);
      importJobId = null;
      updateImportButtonState();
    }
  }

  async function pollImportStatus() {
    if (!importJobId) return;
    try {
      const res = await fetch(`/api/import-status/${encodeURIComponent(importJobId)}`);
      if (!res.ok) throw new Error(`GET /api/import-status -> ${res.status}`);
      const s = await res.json();
      renderImportStatus(s);
      if (s.state === "done" || s.state === "error") {
        importJobId = null;
        updateImportButtonState();
        return;
      }
      importPollTimer = setTimeout(pollImportStatus, 750);
    } catch (err) {
      // Network failure → stop polling, show error.
      setImportState("error", "error");
      showImportError(`status poll failed: ${err.message}`);
      importJobId = null;
      if (importPollTimer) clearTimeout(importPollTimer);
      importPollTimer = null;
      updateImportButtonState();
    }
  }

  function renderImportStatus(s) {
    const state = s.state || "running";
    setImportState(state, state);
    const copied = Number(s.copied) || 0;
    const total = Number(s.total) || 0;
    const skipped = Number(s.skipped) || 0;
    const failed = Number(s.failed) || 0;
    els.importProgress.textContent =
      `(${copied} / ${total} copied, ${skipped} skipped, ${failed} failed)`;

    const tail = typeof s.log_tail === "string" ? s.log_tail : "";
    if (state === "error") {
      showImportError(s.error || "import failed");
      els.importLog.textContent = lastLines(tail, 6);
    } else {
      els.importError.classList.add("hidden");
      els.importLog.textContent = lastLines(tail, 3);
    }
    els.importLog.scrollTop = els.importLog.scrollHeight;

    if (state === "done") {
      const dest = s.dest_folder || els.importDest.textContent || "";
      showImportSuccess(dest);
    } else {
      els.importSuccess.classList.add("hidden");
    }
  }

  function lastLines(text, n) {
    if (!text) return "";
    const lines = text.split("\n").filter((l) => l.length > 0);
    return lines.slice(-n).join("\n");
  }

  function setImportState(label, cls) {
    els.importStateBadge.textContent = label;
    els.importStateBadge.className = `import-state-badge${cls ? " " + cls : ""}`;
  }

  function showImportError(msg) {
    els.importError.textContent = msg || "";
    els.importError.classList.toggle("hidden", !msg);
    els.importSuccess.classList.add("hidden");
  }

  function showImportSuccess(destFolder) {
    els.importSuccessMsg.textContent = "Imported to:";
    els.importDest.textContent = destFolder || "";
    // Hook for a future agent: navigates the GUI to the new folder via the
    // ?clips=<path> query param. The receiving handler isn't wired yet; this
    // just exposes the link.
    if (destFolder) {
      els.importUseFolder.href = `?clips=${encodeURIComponent(destFolder)}`;
      els.importUseFolder.addEventListener("click", onUseFolderClick, { once: true });
    }
    els.importSuccess.classList.remove("hidden");
    els.importError.classList.add("hidden");
  }

  function onUseFolderClick(e) {
    e.preventDefault();
    const dest = els.importDest.textContent || "";
    if (dest) {
      window.location = `?clips=${encodeURIComponent(dest)}`;
    }
  }

  function resetImportStatus() {
    els.importError.classList.add("hidden");
    els.importError.textContent = "";
    els.importSuccess.classList.add("hidden");
    els.importLog.textContent = "";
    els.importProgress.textContent = "";
    if (importPollTimer) clearTimeout(importPollTimer);
    importPollTimer = null;
  }

  // -------- Wire-up --------
  els.selectAll.addEventListener("click", selectAll);
  els.deselectAll.addEventListener("click", deselectAll);
  els.render.addEventListener("click", () => startRender(false));
  els.renderPreviewBtn.addEventListener("click", renderPreview);
  els.theme.addEventListener("change", applyTheme);
  if (els.lengthMode) {
    els.lengthMode.addEventListener("change", syncLengthCustomVisibility);
    // Sync once on load so a server-rendered initial value (e.g. when a
    // future Phase persists the user's choice) lands in the right state.
    syncLengthCustomVisibility();
  }
  els.modalClose.addEventListener("click", closeModal);
  els.copyPath.addEventListener("click", copyPath);
  els.modal.addEventListener("click", (e) => { if (e.target === els.modal) closeModal(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.modal.classList.contains("hidden")) closeModal();
  });

  els.importBtn.addEventListener("click", () => startImport(false));
  els.dryRunBtn.addEventListener("click", () => startImport(true));

  loadOptions();
  loadPreferences();
  loadImportSources();
  // Load sources first so the plan can resolve thumb URLs via path → /thumbs/<key>.jpg.
  loadSources().finally(loadPlan);
})();
