// ===== DOM =====
const $ = (id) => document.getElementById(id);
const input       = $("inputText");
const output      = $("outputText");
const charCount   = $("charCount");
const clearBtn    = $("clearBtn");
const copyBtn     = $("copyBtn");
const swapBtn     = $("swapBtn");
const reloadBtn   = $("reloadBtn");
const srcLangBtn  = $("srcLangBtn");
const tgtLangBtn  = $("tgtLangBtn");
const modelSelect = $("modelSelect");
const statusPill  = $("statusPill");
const statusText  = $("statusText");
const banner      = $("banner");
const toast       = $("toast");

let LANGUAGES = [];
let currentAbort = null;

// ===== Status & banner =====
const setStatus = (state, label) => {
  statusPill.dataset.state = state;
  statusText.textContent = label;
};
const showBanner = (html) => { banner.innerHTML = html; banner.classList.add("show"); };
const hideBanner = () => banner.classList.remove("show");

// ===== Server-driven config =====
async function loadConfig() {
  const r = await fetch("/api/config", { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const cfg = await r.json();
  LANGUAGES = cfg.languages || [];
  if (cfg.defaultSourceLanguage) {
    srcLangBtn.dataset.lang = cfg.defaultSourceLanguage;
    srcLangBtn.textContent = cfg.defaultSourceLanguage;
  }
  if (cfg.defaultTargetLanguage) {
    tgtLangBtn.dataset.lang = cfg.defaultTargetLanguage;
    tgtLangBtn.textContent = cfg.defaultTargetLanguage;
  }
}

// ===== Model list =====
async function loadModels() {
  setStatus("loading", "connecting");
  modelSelect.disabled = true;
  modelSelect.innerHTML = '<option>Loading models...</option>';

  try {
    const r = await fetch("/api/models", { cache: "no-store" });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    const models = data.models || [];

    if (models.length === 0) {
      setStatus("err", "no models");
      modelSelect.innerHTML = '<option>No models available</option>';
      if (data.allow_listed) {
        showBanner(
          `No installed Ollama models match the allow-list. ` +
          `Pull an allowed model or edit <code>allowed_models</code> in <code>config.json</code>.`
        );
      } else {
        showBanner(
          `Ollama is running but no models are installed. Pull one, e.g. ` +
          `<code>ollama pull llama3.2</code>.`
        );
      }
      return;
    }

    modelSelect.innerHTML = models
      .map(name => `<option value="${name}">${name}</option>`).join("");
    modelSelect.disabled = false;
    setStatus("ok", `${models.length} model${models.length > 1 ? "s" : ""}`);
    hideBanner();
  } catch (err) {
    setStatus("err", "offline");
    modelSelect.innerHTML = '<option>Ollama unreachable</option>';
    showBanner(
      `Backend could not reach Ollama. Start it with <code>ollama serve</code>, ` +
      `then click the reload button. <span style="opacity:0.7">(${err.message})</span>`
    );
  }
}

// ===== Translation =====
async function translate() {
  const text = input.value.trim();
  if (currentAbort) { currentAbort.abort(); currentAbort = null; }
  if (!text) { renderEmpty(); return; }
  if (modelSelect.disabled || !modelSelect.value) {
    renderError("No model available. Check Ollama connection.");
    return;
  }

  const model = modelSelect.value;
  const src = srcLangBtn.dataset.lang;
  const tgt = tgtLangBtn.dataset.lang;

  const body = {
    model,
    stream: true,
    options: { temperature: 0.2 },
    // Used by the local RNN model to pick its direction; ignored by Ollama.
    source_language: src,
    target_language: tgt,
    messages: [
      {
        role: "system",
        content:
          `You are a precise translator. Translate the user's text from ${src} to ${tgt}. ` +
          `Output ONLY the translation. No quotes, no commentary, no romanization, ` +
          `no source text, no explanations.`,
      },
      { role: "user", content: text },
    ],
  };

  const ac = new AbortController();
  currentAbort = ac;

  output.classList.remove("is-empty", "is-error");
  output.textContent = "";
  output.classList.add("typing-caret");

  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
    if (!r.ok || !r.body) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let acc = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() || "";
      for (const line of lines) {
        const s = line.trim();
        if (!s) continue;
        try {
          const chunk = JSON.parse(s);
          if (chunk.message?.content) {
            acc += chunk.message.content;
            output.textContent = acc;
          }
        } catch { /* ignore parse blip */ }
      }
    }
    output.classList.remove("typing-caret");
    if (!acc) renderEmpty();
  } catch (err) {
    if (err.name === "AbortError") return;
    output.classList.remove("typing-caret");
    renderError(`Translation failed: ${err.message}`);
  } finally {
    if (currentAbort === ac) currentAbort = null;
  }
}

function renderEmpty() {
  output.classList.remove("is-error", "typing-caret");
  output.classList.add("is-empty");
  output.textContent = "Translation will appear here";
}
function renderError(msg) {
  output.classList.remove("is-empty", "typing-caret");
  output.classList.add("is-error");
  output.textContent = msg;
}

// ===== Debounced auto-translate =====
let debounceTimer;
const scheduleTranslate = () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(translate, 650);
};

// ===== Char count =====
const updateCount = () => { charCount.textContent = input.value.length; };

// ===== Language picker menu =====
let openMenu = null;
function closeMenu() {
  if (openMenu) { openMenu.remove(); openMenu = null; }
}
document.addEventListener("click", (e) => {
  if (openMenu && !openMenu.contains(e.target) &&
      e.target !== srcLangBtn && e.target !== tgtLangBtn) closeMenu();
});

function openLangMenu(btn) {
  closeMenu();
  const menu = document.createElement("div");
  menu.className = "menu";
  const rect = btn.getBoundingClientRect();
  menu.style.top = `${rect.bottom + window.scrollY + 4}px`;
  menu.style.left = `${rect.left + window.scrollX}px`;
  for (const lang of LANGUAGES) {
    const item = document.createElement("button");
    item.textContent = lang;
    if (lang === btn.dataset.lang) item.classList.add("is-selected");
    item.addEventListener("click", () => {
      btn.dataset.lang = lang;
      btn.textContent = lang;
      closeMenu();
      if (input.value.trim()) translate();
    });
    menu.appendChild(item);
  }
  document.body.appendChild(menu);
  openMenu = menu;
}
srcLangBtn.addEventListener("click", () => openLangMenu(srcLangBtn));
tgtLangBtn.addEventListener("click", () => openLangMenu(tgtLangBtn));

// ===== Wire up =====
input.addEventListener("input", () => { updateCount(); scheduleTranslate(); });
input.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    clearTimeout(debounceTimer);
    translate();
  }
});

clearBtn.addEventListener("click", () => {
  input.value = "";
  updateCount();
  if (currentAbort) currentAbort.abort();
  renderEmpty();
  input.focus();
});

const showToast = (msg) => {
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => toast.classList.remove("show"), 1400);
};

copyBtn.addEventListener("click", async () => {
  if (output.classList.contains("is-empty") || output.classList.contains("is-error")) return;
  const text = output.textContent;
  if (!text) return;
  try { await navigator.clipboard.writeText(text); showToast("Copied"); }
  catch { showToast("Copy failed"); }
});

swapBtn.addEventListener("click", () => {
  const s = srcLangBtn.dataset.lang;
  srcLangBtn.dataset.lang = tgtLangBtn.dataset.lang;
  srcLangBtn.textContent = tgtLangBtn.dataset.lang;
  tgtLangBtn.dataset.lang = s;
  tgtLangBtn.textContent = s;

  const outText =
    output.classList.contains("is-empty") || output.classList.contains("is-error")
      ? "" : output.textContent;
  input.value = outText;
  updateCount();
  if (input.value.trim()) translate(); else renderEmpty();
});

reloadBtn.addEventListener("click", () => {
  reloadBtn.style.transition = "transform 500ms ease";
  reloadBtn.style.transform = "rotate(360deg)";
  setTimeout(() => {
    reloadBtn.style.transition = "none";
    reloadBtn.style.transform = "rotate(0)";
  }, 550);
  loadModels();
});

document.querySelectorAll(".nav-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".nav-tab").forEach((t) => t.classList.remove("is-active"));
    tab.classList.add("is-active");
  });
});

// ===== Boot =====
(async () => {
  updateCount();
  try {
    await loadConfig();
  } catch (err) {
    showBanner(`Could not load config from backend: ${err.message}. Is <code>server.py</code> running?`);
  }
  await loadModels();
})();
