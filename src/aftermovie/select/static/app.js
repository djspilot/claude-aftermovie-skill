// Aftermovie clip-selection UI. Vanilla JS, no deps. Talks to /api/*.
(() => {
  "use strict";

  const KIND_BADGE = { video: "V", still: "S", live_photo: "L" };

  /** @type {Array<{path:string,name:string,kind:string,thumb_url:string,selected:boolean}>} */
  let sources = [];
  /** @type {Set<string>} */
  const excluded = new Set();
  let saveTimer = null;
  let pollTimer = null;
  let currentJobId = null;

  const $ = (sel) => document.querySelector(sel);

  const els = {
    grid: $("#grid"),
    status: $("#status"),
    counter: $("#counter"),
    length: $("#length"),
    pace: $("#pace"),
    selectAll: $("#select-all"),
    deselectAll: $("#deselect-all"),
    render: $("#render"),
    modal: $("#modal"),
    modalState: $("#modal-state"),
    modalLog: $("#modal-log"),
    modalResult: $("#modal-result"),
    modalPath: $("#modal-path"),
    modalClose: $("#modal-close"),
    copyPath: $("#copy-path"),
    revealLink: $("#reveal-link"),
    template: $("#cell-template"),
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
      for (const s of sources) if (s.selected === false) excluded.add(s.path);
      renderGrid();
      updateCounter();
      if (!sources.length) setStatus("No sources found. Drop clips into the watched folder and reload.");
      else setStatus("", true);
    } catch (err) {
      setStatus(`Failed to load sources: ${err.message}`, false, true);
    }
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
  async function startRender() {
    const body = {
      excluded: [...excluded],
      max_length: clampInt(els.length.value, 10, 600, 90),
      pace: els.pace.value || "auto",
    };
    openModal();
    setModalState("starting…");
    appendLog(`POST /api/render  excluded=${body.excluded.length}  max_length=${body.max_length}s  pace=${body.pace}`);
    els.render.disabled = true;
    try {
      const { job_id } = await api("POST", "/api/render", body);
      currentJobId = job_id;
      appendLog(`job_id = ${job_id}`);
      pollStatus();
    } catch (err) {
      setModalState(`error: ${err.message}`, "error");
      els.render.disabled = false;
    }
  }

  function clampInt(v, min, max, dflt) {
    const n = parseInt(v, 10);
    if (!Number.isFinite(n)) return dflt;
    return Math.max(min, Math.min(max, n));
  }

  async function pollStatus() {
    if (!currentJobId) return;
    try {
      const s = await fetch(`/api/status/${encodeURIComponent(currentJobId)}`).then((r) => r.json());
      setModalState(s.state, s.state === "done" ? "done" : s.state === "error" ? "error" : "");
      if (s.log_tail) replaceLog(s.log_tail);
      if (s.state === "done") {
        showResult(s.output_path || "");
        els.render.disabled = false;
        return;
      }
      if (s.state === "error") {
        els.render.disabled = false;
        return;
      }
      pollTimer = setTimeout(pollStatus, 1500);
    } catch (err) {
      appendLog(`status poll failed: ${err.message}`);
      pollTimer = setTimeout(pollStatus, 1500);
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

  // -------- Wire-up --------
  els.selectAll.addEventListener("click", selectAll);
  els.deselectAll.addEventListener("click", deselectAll);
  els.render.addEventListener("click", startRender);
  els.modalClose.addEventListener("click", closeModal);
  els.copyPath.addEventListener("click", copyPath);
  els.modal.addEventListener("click", (e) => { if (e.target === els.modal) closeModal(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.modal.classList.contains("hidden")) closeModal();
  });

  loadSources();
})();
