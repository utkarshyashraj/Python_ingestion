(() => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-btn");
  const runBtn = document.getElementById("run-btn");
  const fileName = document.getElementById("file-name");
  const maxPages = document.getElementById("max-pages");
  const backend = document.getElementById("backend");
  const status = document.getElementById("status");
  const statusText = document.getElementById("status-text");
  const results = document.getElementById("results");
  const meta = document.getElementById("meta");
  const summary = document.getElementById("summary");
  const logEl = document.getElementById("log");
  const copyBtn = document.getElementById("copy-btn");
  const clearBtn = document.getElementById("clear-btn");

  let selectedFile = null;

  function setFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      fileName.textContent = "Please choose a PDF file.";
      fileName.classList.add("error");
      selectedFile = null;
      runBtn.disabled = true;
      return;
    }
    selectedFile = file;
    fileName.classList.remove("error");
    fileName.textContent = `${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    runBtn.disabled = false;
  }

  browseBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) setFile(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  function setBusy(busy, message) {
    status.hidden = !busy;
    runBtn.disabled = busy || !selectedFile;
    browseBtn.disabled = busy;
    if (message) statusText.textContent = message;
  }

  function renderSummary(sections) {
    summary.innerHTML = "";
    if (!sections || !sections.length) {
      summary.textContent = "No section groups returned.";
      return;
    }
    sections.forEach((s) => {
      const row = document.createElement("div");
      row.className = "row";
      const pad = "  ".repeat(Number(s.depth || 0));
      const marker = s.depth === 0 ? "•" : s.depth === 1 ? "◦" : "▪";
      row.textContent = `${pad}${marker} ${s.heading || "(untitled)"}  [${s.items} items · p${s.pages}]`;
      summary.appendChild(row);
    });
  }

  async function runIngest() {
    if (!selectedFile) return;
    setBusy(true, "Running discovery… large PDFs can take a minute.");
    results.hidden = true;

    const body = new FormData();
    body.append("file", selectedFile, selectedFile.name);
    body.append("backend", backend.value || "structured");
    if (maxPages.value) body.append("max_pages", maxPages.value);

    try {
      const res = await fetch("/api/ingest", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Ingest failed (${res.status})`);
      }

      meta.textContent = [
        data.filename,
        data.document_id ? `id ${data.document_id}` : null,
        data.page_count != null ? `${data.page_count} pages` : null,
        `${data.section_count} sections`,
        `${data.logical_block_count} logical blocks`,
        data.max_pages ? `max-pages ${data.max_pages}` : "all pages",
        data.backend,
      ]
        .filter(Boolean)
        .join(" · ");

      renderSummary(data.sections || []);
      logEl.textContent = data.human_readable_log || "";
      results.hidden = false;
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      fileName.classList.add("error");
      fileName.textContent = err.message || String(err);
    } finally {
      setBusy(false);
    }
  }

  runBtn.addEventListener("click", runIngest);

  copyBtn.addEventListener("click", async () => {
    const text = logEl.textContent || "";
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      copyBtn.textContent = "Copied";
      setTimeout(() => {
        copyBtn.textContent = "Copy log";
      }, 1200);
    } catch (_) {
      copyBtn.textContent = "Copy failed";
    }
  });

  clearBtn.addEventListener("click", () => {
    results.hidden = true;
    logEl.textContent = "";
    summary.innerHTML = "";
    meta.textContent = "";
  });
})();
