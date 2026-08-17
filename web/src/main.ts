import "./style.css";
import {
  checkHealth,
  fetchDependencies,
  fetchProviders,
  fetchSettings,
  installDependencies,
  saveApiKey,
  transcribeAudio,
  type DependencyReport,
  type ProviderInfo,
} from "./api.ts";
import { LANGUAGES, MODELS, type TranscriptResult } from "./models.ts";

type View = "transcribe" | "settings" | "setup";

const AUDIO_TYPES = ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac", "audio/mp4", "audio/ogg", "audio/webm", "audio/aac", "audio/x-m4a"];
const AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm", ".aac", ".mp4"];

let activeView: View = "transcribe";
let selectedProvider = "local";
let selectedModel = "small";
let selectedDevice: "cuda" | "cpu" = "cuda";
let selectedFile: File | null = null;
let lastResult: TranscriptResult | null = null;
let gpuVramGb: number | null = null;
let gpuName: string | null = null;
let cudaAvailable = false;
let providers: ProviderInfo[] = [];
let dependencyReport: DependencyReport | null = null;
let installingDeps = false;
let isTranscribing = false;
let statusMessage = "Starting…";
let statusKind: "info" | "ok" | "error" = "info";
let serverOk = false;

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function escapeHtml(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function isAudioFile(file: File): boolean {
  if (AUDIO_TYPES.includes(file.type)) return true;
  return AUDIO_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext));
}

function fileLabel(file: File | null): string {
  if (!file) return "No file selected";
  return `${file.name} · ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
}

function currentProvider() {
  return providers.find((p) => p.id === selectedProvider);
}

function providerModels() {
  const provider = currentProvider();
  if (provider?.models?.length) return provider.models;
  return MODELS.map((m) => ({ id: m.id, label: m.label, recommended: m.recommended }));
}

function currentModelMeta() {
  return MODELS.find((m) => m.id === selectedModel) ?? MODELS[4];
}

function modelNeedsRam(model: { vramGb?: number; id?: string }): boolean {
  if (selectedProvider !== "local") return false;
  const meta = MODELS.find((m) => m.id === model.id) ?? currentModelMeta();
  return cudaAvailable && gpuVramGb != null && meta.vramGb > gpuVramGb;
}

function largestGpuFriendlyModel(): string {
  const localModels = MODELS.filter((model) => !model.englishOnly);
  const fits = gpuVramGb != null
    ? localModels.filter((model) => model.vramGb <= gpuVramGb!)
    : localModels.filter((model) => model.vramGb <= 2);
  return fits.at(-1)?.id ?? "small";
}

function syncModelForProvider(): void {
  const models = providerModels();
  if (!models.some((m) => m.id === selectedModel)) {
    selectedModel = models.find((m) => m.recommended)?.id ?? models[0]?.id ?? "small";
  }
}

function applyGpuSelection(): void {
  if (!cudaAvailable) {
    selectedDevice = "cpu";
    setStatus("No NVIDIA GPU detected. Using RAM.", "info");
    return;
  }
  selectedDevice = "cuda";
  if (modelNeedsRam(currentModelMeta())) {
    const next = largestGpuFriendlyModel();
    selectedModel = next;
    setStatus(`Switched to GPU. ${currentModelMeta().label} fits this card.`, "ok");
    return;
  }
  setStatus(`Using GPU · ${gpuVramGb ?? "?"} GB VRAM`, "ok");
}

function setStatus(message: string, kind: "info" | "ok" | "error" = "info"): void {
  statusMessage = message;
  statusKind = kind;
  const bar = document.querySelector("#status-text");
  const dot = document.querySelector("#status-dot");
  if (bar) bar.textContent = message;
  if (dot) dot.className = `dot ${kind === "ok" ? "ok" : kind === "error" ? "error" : "info"}`;
}

function renderTranscribeView(): string {
  const isLocal = selectedProvider === "local";
  return `
    <div class="view-header">
      <h1>Transcribe</h1>
      <p>Upload audio and convert speech to text.</p>
    </div>
    <div class="grid-2">
      <div class="panel">
        <div class="panel-head"><div><h2>Audio file</h2><p>MP3, WAV, FLAC, M4A, OGG, and more</p></div></div>
        <label class="dropzone" id="dropzone">
          <input id="file-input" type="file" accept="audio/*,.mp3,.wav,.flac,.m4a,.ogg,.webm,.aac" hidden />
          <div class="dropzone-icon">🎙</div>
          <strong>Drop audio here</strong>
          <span>or click to browse</span>
          <em id="file-name">${fileLabel(selectedFile)}</em>
        </label>
        <div class="field-row">
          <label>Task<select id="task"><option value="transcribe">Transcribe</option><option value="translate">Translate to English</option></select></label>
          <label>Language<select id="language">${LANGUAGES.map((l) => `<option value="${l.id}">${l.label}</option>`).join("")}</select></label>
        </div>
        <button id="run" class="btn btn-primary" type="button" ${isTranscribing ? "disabled" : ""}>
          ${isTranscribing ? "Transcribing…" : "Start transcription"}
        </button>
      </div>
      <div class="panel">
        <div class="panel-head">
          <div><h2>Result</h2><p>Full transcript and timestamps</p></div>
          <button id="copy" class="btn" type="button" ${lastResult?.text ? "" : "disabled"}>Copy</button>
        </div>
        <div id="result">${lastResult ? resultHtml(lastResult) : '<p class="empty-state">Your transcript will appear here.</p>'}</div>
      </div>
    </div>
    ${isLocal ? `
    <div class="panel">
      <div class="panel-head"><div><h2>Run on</h2><p>GPU is faster. RAM can run larger models.</p></div></div>
      <div class="chip-row">
        <button class="chip ${selectedDevice === "cuda" ? "active" : ""}" data-device="cuda" type="button" ${cudaAvailable ? "" : "disabled"}>GPU ${cudaAvailable ? `(${gpuVramGb ?? "?"} GB)` : "(unavailable)"}</button>
        <button class="chip ${selectedDevice === "cpu" ? "active" : ""}" data-device="cpu" type="button">RAM</button>
      </div>
    </div>` : ""}
    <div class="panel">
      <div class="panel-head"><div><h2>API keys</h2><p>Add OpenAI, Deepgram, or AssemblyAI keys in Settings.</p></div></div>
      <p style="margin:0;color:var(--muted);font-size:13px">Open <button class="chip active" type="button" data-goto="settings" style="cursor:pointer">Settings</button> → API keys, paste a key, then Save.</p>
    </div>
  `;
}

function renderSettingsView(): string {
  const isLocal = selectedProvider === "local";
  const models = providerModels();
  const cloudProviders = providers.filter((p) => p.requiresApiKey);
  const selectedCloud = cloudProviders.find((p) => p.id === selectedProvider);

  return `
    <div class="view-header">
      <h1>Settings</h1>
      <p>Choose engine, GPU or RAM, model, and cloud API keys.</p>
    </div>

    <div class="panel" id="api-keys-panel">
      <div class="panel-head"><div><h2>API keys</h2><p>Paste a key here to use OpenAI, Deepgram, or AssemblyAI. Keys stay on this PC.</p></div></div>
      ${cloudProviders.length ? cloudProviders.map((provider) => `
        <form class="settings-card ${provider.id === selectedProvider ? "active-key" : ""}" data-key-provider="${provider.id}">
          <div><strong>${escapeHtml(provider.name)}</strong><span class="pill ${provider.configured ? "" : "muted"}">${provider.configured ? "Configured" : "Not set"}</span><p>${escapeHtml(provider.description)}</p></div>
          <label>API key<input type="password" name="apiKey" placeholder="${provider.configured ? "Enter a new key to replace the saved one" : "Paste your API key"}" autocomplete="off" /></label>
          <button class="btn" type="submit">Save ${escapeHtml(provider.name)} key</button>
        </form>
      `).join("") : `<p class="empty-state">Cloud providers are not available.</p>`}
      ${selectedCloud && !selectedCloud.configured ? `<div class="banner-inline info">Add a ${escapeHtml(selectedCloud.name)} key above, then you can transcribe with that engine.</div>` : ""}
    </div>

    <div class="panel">
      <div class="panel-head"><div><h2>Engine</h2><p>Local Whisper runs on this PC. Cloud engines use the API keys above.</p></div></div>
      <div class="grid-3" id="provider-grid">
        ${providers.map((item) => `
            <button class="card ${item.id === selectedProvider ? "active" : ""}" data-provider="${item.id}" type="button">
              <strong>${escapeHtml(item.name)}</strong>
              <span>${item.requiresApiKey ? (item.configured ? "Key saved" : "Add key above") : "This device"}</span>
            </button>`).join("")}
      </div>
    </div>

    ${isLocal ? `
    <div class="panel">
      <div class="panel-head"><div><h2>Compute device</h2><p>You can switch GPU and RAM anytime. GPU is faster; RAM supports larger models.</p></div></div>
      <div class="chip-row" id="device-toggle">
        <button class="chip ${selectedDevice === "cuda" ? "active" : ""}" data-device="cuda" type="button" ${cudaAvailable ? "" : "disabled"}>GPU ${cudaAvailable ? `(${gpuVramGb ?? "?"} GB)` : "(unavailable)"}</button>
        <button class="chip ${selectedDevice === "cpu" ? "active" : ""}" data-device="cpu" type="button">RAM</button>
      </div>
      ${!cudaAvailable ? `<p class="hint-text">GPU is unavailable because CUDA was not detected.</p>` : ""}
    </div>` : ""}

    <div class="panel">
      <div class="panel-head"><div><h2>Model</h2><p>${isLocal ? "Larger models need more memory. Models that do not fit GPU will run on RAM." : "Cloud model for the selected provider."}</p></div></div>
      <div class="grid-3" id="model-grid">
        ${models.map((model) => {
          const meta = MODELS.find((m) => m.id === model.id);
          const needsRam = isLocal && meta ? modelNeedsRam(meta) : false;
          return `
            <button class="card ${model.id === selectedModel ? "active" : ""}" data-model="${model.id}" type="button">
              <strong>${escapeHtml(model.label)}${model.recommended ? '<span class="pill">Best</span>' : ""}${needsRam ? '<span class="pill muted">RAM</span>' : ""}</strong>
              <span>${meta ? meta.vram + " · " + meta.speed : "Cloud"}</span>
            </button>`;
        }).join("")}
      </div>
    </div>
  `;
}

function renderSetupView(): string {
  const deps = dependencyReport;
  return `
    <div class="view-header">
      <h1>Setup</h1>
      <p>System dependencies required to run Whisper UI on this PC.</p>
    </div>

    ${deps && !deps.ready ? `<div class="banner-inline error">${escapeHtml(dependencyMessage())}</div>` : ""}
    ${deps?.ready ? `<div class="banner-inline ok">All dependencies are installed and ready.</div>` : ""}

    <div class="panel">
      <div class="panel-head"><div><h2>Dependency status</h2><p>Checked on this machine</p></div></div>
      <div class="setup-grid">
        <div class="setup-item"><span>Python environment</span><span class="${deps?.packages.ok ? "ok" : "bad"}">${deps?.packages.ok ? "Ready" : "Missing: " + (deps?.packages.missing.join(", ") ?? "?")}</span></div>
        <div class="setup-item"><span>ffmpeg (audio decode)</span><span class="${deps?.ffmpeg.ok ? "ok" : "bad"}">${deps?.ffmpeg.ok ? "Installed" : "Not found"}</span></div>
        <div class="setup-item"><span>NVIDIA GPU (optional)</span><span class="${deps?.cuda.available ? "ok" : ""}">${deps?.cuda.available ? `${deps.cuda.gpu} · ${deps.cuda.vramGb} GB` : "CPU/RAM mode"}</span></div>
        <div class="setup-item"><span>Disk space</span><span class="${deps?.disk.ok ? "ok" : "bad"}">${deps?.disk.freeGb ?? "?"} GB free</span></div>
        <div class="setup-item"><span>Application UI</span><span class="${deps?.frontend.ok ? "ok" : "bad"}">${deps?.frontend.ok ? "Built" : "Not built"}</span></div>
      </div>
      ${deps && !deps.ready ? `
        <button id="install-deps" class="btn btn-primary" type="button" style="margin-top:16px;max-width:320px" ${installingDeps ? "disabled" : ""}>
          ${installingDeps ? "Installing…" : "Install missing dependencies"}
        </button>` : ""}
    </div>

    <div class="panel">
      <div class="panel-head"><div><h2>About</h2><p>Whisper UI v1.0 · Windows desktop app</p></div></div>
      <p style="margin:0;color:var(--muted);font-size:13px;line-height:1.6">
        Installed apps are located in your Programs folder. Use the Start Menu shortcut to launch.
        Model files download automatically on first use and are cached locally.
      </p>
    </div>
  `;
}

function resultHtml(result: TranscriptResult): string {
  const deviceLabel = result.device === "cloud" ? "Cloud" : result.device === "gpu" ? "GPU" : "RAM";
  const segments = result.segments.map((s) => `
    <li><time>${formatTime(s.start)} – ${formatTime(s.end)}</time><p>${escapeHtml(s.text)}</p></li>
  `).join("");

  return `
    <div class="meta-row">
      <span class="meta-tag">${escapeHtml(result.provider ?? selectedProvider)}</span>
      <span class="meta-tag">${escapeHtml(result.model)}</span>
      <span class="meta-tag">${escapeHtml(result.task)}</span>
      <span class="meta-tag">${escapeHtml(deviceLabel)}</span>
      <span class="meta-tag">${escapeHtml(result.language ?? "auto")}</span>
    </div>
    <article class="transcript-box">${escapeHtml(result.text) || "No speech detected."}</article>
    ${segments ? `<ol class="segments">${segments}</ol>` : ""}
  `;
}

function dependencyMessage(): string {
  if (!dependencyReport) return "Checking dependencies…";
  if (dependencyReport.ready) return "All dependencies ready.";
  const parts: string[] = [];
  if (!dependencyReport.ffmpeg.ok) parts.push("ffmpeg");
  if (!dependencyReport.packages.ok) parts.push("Python packages");
  if (!dependencyReport.disk.ok) parts.push("disk space");
  return `Missing: ${parts.join(", ")}. Click Install to set up automatically.`;
}

function renderMainView(): string {
  if (activeView === "settings") return renderSettingsView();
  if (activeView === "setup") return renderSetupView();
  return renderTranscribeView();
}

function render(options: { resetScroll?: boolean } = {}): void {
  const app = document.querySelector<HTMLDivElement>("#app");
  if (!app) return;
  const previousScroll = options.resetScroll ? 0 : (document.querySelector(".main-content")?.scrollTop ?? 0);

  app.innerHTML = `
    <div class="app-window">
      <header class="titlebar">
        <div class="titlebar-brand">
          <img src="/favicon.svg" alt="" class="app-icon" />
          <strong>Whisper UI</strong>
          <span>Speech to Text</span>
        </div>
        <div class="titlebar-actions">
          <button class="icon-btn primary-action" type="button" data-goto="transcribe">Transcribe</button>
        </div>
      </header>

      <div class="app-body">
        <nav class="sidebar">
          <button class="nav-item ${activeView === "transcribe" ? "active" : ""}" data-view="transcribe" type="button">
            <span class="nav-icon">🎙</span> Transcribe
          </button>
          <button class="nav-item ${activeView === "settings" ? "active" : ""}" data-view="settings" type="button">
            <span class="nav-icon">⚙</span> Settings
          </button>
          <button class="nav-item ${activeView === "setup" ? "active" : ""}" data-view="setup" type="button">
            <span class="nav-icon">📦</span> Setup
            ${dependencyReport && !dependencyReport.ready ? '<span class="pill">!</span>' : ""}
          </button>
          <div class="sidebar-footer">
            ${serverOk ? (cudaAvailable ? `GPU: ${gpuName ?? "CUDA"}` : "CPU mode") : "Server offline"}
          </div>
        </nav>
        <main class="main-content">${renderMainView()}</main>
      </div>

      <footer class="statusbar">
        <span><span id="status-dot" class="dot ${statusKind === "ok" ? "ok" : statusKind === "error" ? "error" : "info"}"></span><span id="status-text">${escapeHtml(statusMessage)}</span></span>
        <span>Whisper UI · Local &amp; Cloud</span>
      </footer>
    </div>
    ${isTranscribing ? `
      <div class="overlay">
        <div class="overlay-card">
          <div class="spinner"></div>
          <strong>Transcribing…</strong>
          <p style="color:var(--muted);font-size:13px;margin:8px 0 0">This may take a few minutes for large files.</p>
        </div>
      </div>` : ""}
  `;

  bindEvents();
  const main = document.querySelector(".main-content");
  if (main) main.scrollTop = previousScroll;
}

function bindEvents(): void {
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeView = (btn.dataset.view as View) ?? "transcribe";
      render({ resetScroll: true });
    });
  });

  document.querySelectorAll<HTMLButtonElement>("[data-goto]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeView = (btn.dataset.goto as View) ?? "transcribe";
      render({ resetScroll: true });
    });
  });

  document.querySelectorAll<HTMLFormElement>(".settings-card").forEach((form) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const provider = form.dataset.keyProvider ?? "";
      const apiKey = form.querySelector<HTMLInputElement>('input[name="apiKey"]')?.value ?? "";
      if (!apiKey.trim()) {
        setStatus("Paste an API key first.", "error");
        return;
      }
      try {
        await saveApiKey(provider, apiKey);
        await refreshProviders(false);
        selectedProvider = provider;
        syncModelForProvider();
        setStatus(`${currentProvider()?.name ?? provider} API key saved.`, "ok");
        render();
      } catch (error) {
        setStatus(error instanceof Error ? error.message : "Could not save key", "error");
      }
    });
  });

  document.querySelector("#install-deps")?.addEventListener("click", async () => {
    installingDeps = true;
    render();
    setStatus("Installing dependencies…", "info");
    try {
      await installDependencies();
      await refreshDependencies();
      setStatus(dependencyReport?.ready ? "Setup complete." : "Some items still need attention.", dependencyReport?.ready ? "ok" : "error");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Install failed", "error");
    } finally {
      installingDeps = false;
      render();
    }
  });

  document.querySelectorAll<HTMLButtonElement>("[data-provider]").forEach((btn) => {
    btn.addEventListener("click", () => {
      selectedProvider = btn.dataset.provider ?? "local";
      syncModelForProvider();
      const provider = currentProvider();
      const needsKey = Boolean(provider?.requiresApiKey && !provider.configured);
      if (needsKey) {
        setStatus(`Paste a ${provider?.name} API key in the API keys section, then Save.`, "info");
      } else {
        setStatus(`Engine: ${provider?.name ?? selectedProvider}`, "ok");
      }
      render();
    });
  });

  document.querySelectorAll<HTMLButtonElement>("[data-device]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      if (btn.dataset.device === "cpu") {
        selectedDevice = "cpu";
        setStatus("Using system RAM.", "ok");
      } else {
        applyGpuSelection();
      }
      render();
    });
  });

  document.querySelectorAll<HTMLButtonElement>("[data-model]").forEach((btn) => {
    btn.addEventListener("click", () => {
      selectedModel = btn.dataset.model ?? "small";
      if (selectedProvider === "local" && modelNeedsRam(currentModelMeta()) && selectedDevice === "cuda") {
        selectedDevice = "cpu";
        setStatus(`${currentModelMeta().label} is too large for this GPU. Switched to RAM. Click GPU anytime to go back.`, "info");
      }
      render();
    });
  });

  const dropzone = document.querySelector<HTMLLabelElement>("#dropzone");
  const fileInput = document.querySelector<HTMLInputElement>("#file-input");
  const fileName = document.querySelector("#file-name");

  const assignFile = (file: File | null) => {
    if (file && !isAudioFile(file)) { setStatus("Please choose an audio file.", "error"); return; }
    selectedFile = file;
    if (fileName) fileName.textContent = fileLabel(file);
  };

  fileInput?.addEventListener("change", () => assignFile(fileInput.files?.[0] ?? null));
  dropzone?.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("over"); });
  dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("over"));
  dropzone?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("over");
    assignFile(e.dataTransfer?.files[0] ?? null);
  });

  document.querySelector("#copy")?.addEventListener("click", async () => {
    if (!lastResult?.text) return;
    await navigator.clipboard.writeText(lastResult.text);
    setStatus("Copied to clipboard.", "ok");
  });

  document.querySelector("#run")?.addEventListener("click", async () => {
    if (!selectedFile) { setStatus("Choose an audio file first.", "error"); return; }
    const provider = currentProvider();
    if (provider?.requiresApiKey && !provider.configured) {
      setStatus("Add API key in Settings.", "error");
      activeView = "settings";
      render();
      return;
    }

    const task = document.querySelector<HTMLSelectElement>("#task")?.value ?? "transcribe";
    const language = document.querySelector<HTMLSelectElement>("#language")?.value ?? "";

    isTranscribing = true;
    render();
    setStatus(selectedProvider === "local" ? "Processing on this device…" : `Sending to ${provider?.name}…`, "info");

    try {
      lastResult = await transcribeAudio({
        file: selectedFile,
        provider: selectedProvider,
        model: selectedModel,
        task,
        language,
        device: selectedDevice,
      });
      setStatus("Transcription complete.", "ok");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Transcription failed", "error");
    } finally {
      isTranscribing = false;
      render();
    }
  });
}

async function refreshProviders(applyDefault = true): Promise<void> {
  providers = await fetchProviders();
  if (applyDefault) {
    const settings = await fetchSettings();
    selectedProvider = settings.defaultProvider || selectedProvider;
  }
  syncModelForProvider();
}

async function refreshDependencies(): Promise<void> {
  dependencyReport = await fetchDependencies();
}

async function boot(): Promise<void> {
  render();
  try {
    const health = await checkHealth();
    serverOk = true;
    cudaAvailable = health.cuda;
    gpuVramGb = health.vramGb;
    gpuName = health.gpu;
    if (!cudaAvailable) selectedDevice = "cpu";

    await Promise.all([
      refreshProviders().catch(() => {
        providers = [{ id: "local", name: "Local Whisper", requiresApiKey: false, configured: true, description: "", models: MODELS.map((m) => ({ id: m.id, label: m.label })) }];
      }),
      refreshDependencies().catch(() => undefined),
    ]);

    syncModelForProvider();
    if (selectedDevice === "cuda" && modelNeedsRam(currentModelMeta())) selectedDevice = "cpu";

    if (dependencyReport && !dependencyReport.ready) {
      setStatus("Setup required — open Setup to install dependencies.", "error");
    } else {
      const mode = selectedProvider !== "local" ? "Cloud ready" : cudaAvailable ? `GPU ready · ${gpuVramGb} GB VRAM` : "CPU/RAM mode";
      setStatus(`Connected · ${mode}`, "ok");
    }
  } catch {
    serverOk = false;
    setStatus("Server not running. Launch Whisper UI from the Start Menu.", "error");
  }
  render();
}

void boot();
