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
import { LANGUAGES, MODELS, TARGET_LANGUAGES, type TranscriptResult } from "./models.ts";

type View = "transcribe" | "live" | "settings" | "setup";

type StreamEvent = {
  type: string;
  text?: string;
  scope?: string;
  seconds?: number;
};

type LiveAudio = {
  stream: MediaStream;
  context: AudioContext;
  node: AudioWorkletNode;
};

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
let liveActive = false;
let liveSourceLanguage = "";
let liveTargetLanguage = "en";
let liveTranscript = "";
let livePartial = "";
let liveTranslation = "";
let liveTranslationNote = "";
let liveSocket: WebSocket | null = null;
let liveAudio: LiveAudio | null = null;
let liveAnalyser: AnalyserNode | null = null;
let liveMeterFrame = 0;
let liveHeardSeconds = 0;
let liveServerSeconds = 0;
let liveStreamStatus = "";
let liveAudioWatch = 0;
let liveStopping = false;
let microphoneCount: number | null = null;
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

function appendChunk(current: string, text: string): string {
  const next = text.trim();
  if (!next) return current;
  return current ? `${current} ${next}` : next;
}

function liveModelId(): string {
  return MODELS.some((model) => model.id === selectedModel) ? selectedModel : "small";
}

function liveModelLabel(): string {
  return MODELS.find((model) => model.id === liveModelId())?.label ?? liveModelId();
}

function liveDeviceLabel(): string {
  return selectedDevice === "cuda" ? "GPU" : "RAM";
}

function liveMeterLabel(): string {
  if (!liveActive) return "Start listening and this meter shows the live microphone stream.";
  if (liveHeardSeconds <= 0) return "Opening the microphone stream…";
  const sent = liveHeardSeconds.toFixed(1);
  if (liveServerSeconds > 0) return `Streaming · ${sent}s sent · ${liveServerSeconds.toFixed(1)}s received`;
  return `Streaming · ${sent}s of microphone audio`;
}

function liveTranscriptPlaceholder(): string {
  if (liveStreamStatus) return liveStreamStatus;
  if (liveActive) return "Streaming… words appear here after a few seconds of speech.";
  return "Your words will appear here as you speak.";
}

function paintStreamLabels(): void {
  const label = document.querySelector("#live-meter-label");
  if (label) label.textContent = liveMeterLabel();
  const empty = document.querySelector<HTMLElement>("#live-transcript-empty");
  if (empty && !liveTranscript && !livePartial) {
    empty.hidden = false;
    empty.textContent = liveTranscriptPlaceholder();
  }
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

function renderLiveView(): string {
  const deviceLabel = liveDeviceLabel();
  const modelLabel = liveModelLabel();
  const transcriptEmpty = !liveTranscript && !livePartial;
  const translationEmpty = !liveTranslation && !liveTranslationNote;
  return `
    <div class="view-header">
      <h1>Live ${liveActive ? '<span class="live-dot" title="Listening"></span>' : ""}</h1>
      <p>Speak into the microphone. Transcript and translation appear as you talk.</p>
    </div>
    ${microphoneCount === 0 ? `<div class="banner-inline error">No microphone is connected. Plug one in, or enable it under Windows Settings, System, Sound, Input.</div>` : ""}
    ${selectedProvider !== "local" ? `<div class="banner-inline info">Live speech uses Local Whisper (${escapeHtml(modelLabel)}) on ${deviceLabel}. Cloud engines still apply to file transcription.</div>` : ""}
    <div class="panel">
      <div class="panel-head">
        <div>
          <h2>Microphone</h2>
          <p>Local Whisper · ${escapeHtml(modelLabel)} · ${deviceLabel}</p>
        </div>
      </div>
      <div class="field-row">
        <label>Source language
          <select id="live-source" ${liveActive ? "disabled" : ""}>
            ${LANGUAGES.map((language) => `<option value="${language.id}" ${language.id === liveSourceLanguage ? "selected" : ""}>${language.label}</option>`).join("")}
          </select>
        </label>
        <label>Translate into
          <select id="live-target" ${liveActive ? "disabled" : ""}>
            ${TARGET_LANGUAGES.map((language) => `<option value="${language.id}" ${language.id === liveTargetLanguage ? "selected" : ""}>${language.label}</option>`).join("")}
          </select>
        </label>
      </div>
      <p class="hint-text">The browser uses this PC's microphone. English translates on this PC. Any other language needs an OpenAI key in Settings.</p>
      <div class="live-stream ${liveActive ? "on" : ""}">
        <div id="live-meter" class="live-meter" aria-hidden="true">${Array.from({ length: 28 }, () => "<span></span>").join("")}</div>
        <p id="live-meter-label">${liveMeterLabel()}</p>
      </div>
      <button id="live-toggle" class="btn btn-primary" type="button" ${liveStopping ? "disabled" : ""}>${liveActive ? "Stop listening" : liveStopping ? "Stopping…" : "Start listening"}</button>
    </div>
    <div class="grid-2">
      <div class="panel">
        <div class="panel-head"><div><h2>Transcript</h2><p>Committed text, with the latest phrase in italics</p></div></div>
        <div class="transcript-box">
          <p id="live-transcript-empty" class="empty-state" ${transcriptEmpty ? "" : "hidden"}>${escapeHtml(liveTranscriptPlaceholder())}</p>
          <span id="live-transcript">${escapeHtml(liveTranscript)}</span>
          <p id="live-partial" class="live-partial">${escapeHtml(livePartial)}</p>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head"><div><h2>Translation</h2><p>${liveTargetLanguage ? "Translated phrases" : "Translation is off"}</p></div></div>
        <div class="transcript-box">
          <p id="live-translation-empty" class="empty-state" ${translationEmpty ? "" : "hidden"}>${liveTargetLanguage ? "Translation appears here after each phrase." : "Transcript only is selected."}</p>
          <span id="live-translation">${escapeHtml(liveTranslation)}</span>
          <p id="live-translation-note" class="live-note">${escapeHtml(liveTranslationNote)}</p>
        </div>
      </div>
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
  if (activeView === "live") return renderLiveView();
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
          <button class="nav-item ${activeView === "live" ? "active" : ""}" data-view="live" type="button">
            <span class="nav-icon">📡</span> Live
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

function paintLive(): void {
  const transcript = document.querySelector<HTMLElement>("#live-transcript");
  const partial = document.querySelector<HTMLElement>("#live-partial");
  const translation = document.querySelector<HTMLElement>("#live-translation");
  const note = document.querySelector<HTMLElement>("#live-translation-note");
  const transcriptEmpty = document.querySelector<HTMLElement>("#live-transcript-empty");
  const translationEmpty = document.querySelector<HTMLElement>("#live-translation-empty");
  if (transcript) transcript.textContent = liveTranscript;
  if (partial) partial.textContent = livePartial;
  if (translation) translation.textContent = liveTranslation;
  if (note) note.textContent = liveTranslationNote;
  if (transcriptEmpty) {
    transcriptEmpty.hidden = Boolean(liveTranscript || livePartial);
    if (!transcriptEmpty.hidden) transcriptEmpty.textContent = liveTranscriptPlaceholder();
  }
  if (translationEmpty) translationEmpty.hidden = Boolean(liveTranslation || liveTranslationNote);
  paintStreamLabels();
}

async function refreshMicrophones(): Promise<void> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    microphoneCount = 0;
    return;
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  microphoneCount = devices.filter((device) => device.kind === "audioinput").length;
}

async function openMicrophone(): Promise<MediaStream> {
  const attempts: MediaStreamConstraints[] = [
    { audio: true },
    {
      audio: {
        channelCount: { ideal: 1 },
        echoCancellation: true,
        noiseSuppression: true,
      },
    },
  ];
  let lastError: unknown;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      lastError = error;
      if (error instanceof DOMException && error.name !== "OverconstrainedError") throw error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Could not start the microphone");
}

function microphoneError(error: unknown): string {
  if (!navigator.mediaDevices?.getUserMedia) {
    return "This page cannot use the microphone. Open Whisper UI in the browser at 127.0.0.1.";
  }
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
      return "Microphone blocked. Allow microphone access for this page in the browser, then try again.";
    }
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      return "No microphone is connected. Plug one in, or enable it under Windows Settings, System, Sound, Input.";
    }
    if (error.name === "NotReadableError") {
      return "The microphone is in use by another app.";
    }
  }
  return error instanceof Error ? error.message : "Could not start the microphone";
}

function startLiveMeter(analyser: AnalyserNode): void {
  stopLiveMeter();
  liveAnalyser = analyser;
  const bins = new Uint8Array(analyser.frequencyBinCount);
  const draw = () => {
    liveMeterFrame = requestAnimationFrame(draw);
    const bars = document.querySelectorAll<HTMLElement>("#live-meter span");
    if (!bars.length || !liveAnalyser) return;
    liveAnalyser.getByteFrequencyData(bins);
    bars.forEach((bar, index) => {
      const bin = bins[Math.floor((index / bars.length) * bins.length * 0.65)] ?? 0;
      const level = 0.08 + (bin / 255) * 0.92;
      bar.style.transform = `scaleY(${level.toFixed(3)})`;
    });
  };
  draw();
}

function stopLiveMeter(): void {
  if (liveMeterFrame) cancelAnimationFrame(liveMeterFrame);
  liveMeterFrame = 0;
  liveAnalyser = null;
  window.clearTimeout(liveAudioWatch);
  liveAudioWatch = 0;
}

function teardownLiveAudio(): void {
  stopLiveMeter();
  liveAudio?.stream.getTracks().forEach((track) => track.stop());
  void liveAudio?.context.close().catch(() => undefined);
  liveAudio = null;
}

function stopLive(sendStop = true): void {
  const socket = liveSocket;
  const running = liveActive || Boolean(socket) || Boolean(liveAudio);
  if (!running) return;
  liveActive = false;
  teardownLiveAudio();
  if (sendStop && socket && socket.readyState === WebSocket.OPEN) {
    liveStopping = true;
    socket.send(JSON.stringify({ type: "stop" }));
    return;
  }
  liveStopping = false;
  if (socket) {
    socket.close();
    liveSocket = null;
  }
}

function handleStreamEvent(event: StreamEvent): void {
  if (event.type === "ready") {
    liveStreamStatus = "Connected. Speak and the words will stream here.";
    paintStreamLabels();
    setStatus("Listening…", "ok");
    return;
  }
  if (event.type === "level") {
    liveServerSeconds = Number(event.seconds ?? 0);
    paintStreamLabels();
    return;
  }
  if (event.type === "status") {
    liveStreamStatus = event.text ?? "";
    paintStreamLabels();
    setStatus(liveStreamStatus, "info");
    return;
  }
  if (event.type === "partial") {
    livePartial = event.text ?? "";
    paintLive();
    return;
  }
  if (event.type === "transcript") {
    liveTranscript = appendChunk(liveTranscript, event.text ?? "");
    paintLive();
    return;
  }
  if (event.type === "translation") {
    liveTranslation = appendChunk(liveTranslation, event.text ?? "");
    paintLive();
    return;
  }
  if (event.type === "error") {
    const message = event.text ?? "Live transcription failed";
    if (event.scope === "translation") {
      liveTranslationNote = message;
      paintLive();
      setStatus(message, "info");
      return;
    }
    setStatus(message, "error");
    stopLive(false);
    if (activeView === "live") render();
    return;
  }
  if (event.type === "stopped") {
    livePartial = "";
    liveActive = false;
    liveStopping = false;
    paintLive();
    const socket = liveSocket;
    liveSocket = null;
    socket?.close();
    if (activeView === "live") render();
    setStatus("Live session ended.", "ok");
  }
}

async function startLive(context: AudioContext): Promise<void> {
  if (liveActive || liveSocket || liveStopping) {
    void context.close();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    void context.close();
    setStatus("This page cannot use the microphone. Open Whisper UI in the browser at 127.0.0.1.", "error");
    return;
  }

  setStatus("Allow the microphone in the browser prompt…", "info");
  let stream: MediaStream;
  try {
    stream = await openMicrophone();
  } catch (error) {
    void context.close();
    microphoneCount = 0;
    setStatus(microphoneError(error), "error");
    if (activeView === "live") render();
    return;
  }

  liveTranscript = "";
  livePartial = "";
  liveTranslation = "";
  liveTranslationNote = "";
  liveHeardSeconds = 0;
  liveServerSeconds = 0;
  liveStreamStatus = "Opening the microphone stream…";
  liveActive = true;
  render();
  setStatus("Connecting microphone…", "info");

  try {
    if (context.state === "suspended") await context.resume();
    if (context.state !== "running") {
      throw new Error("The browser blocked the microphone stream. Click Start listening again.");
    }
    if (!liveActive) {
      stream.getTracks().forEach((track) => track.stop());
      await context.close();
      return;
    }
    await context.audioWorklet.addModule("/pcm-worklet.js");
    if (!liveActive) {
      stream.getTracks().forEach((track) => track.stop());
      await context.close();
      return;
    }
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, "pcm-capture");
    const analyser = context.createAnalyser();
    analyser.fftSize = 128;
    analyser.smoothingTimeConstant = 0.75;
    const silent = context.createGain();
    silent.gain.value = 0;
    source.connect(node);
    source.connect(analyser);
    node.connect(silent);
    silent.connect(context.destination);
    startLiveMeter(analyser);
    liveAudio = { stream, context, node };

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${proto}://${location.host}/api/stream`);
    liveSocket = socket;
    socket.addEventListener("open", () => {
      socket.send(JSON.stringify({
        type: "start",
        model: liveModelId(),
        language: liveSourceLanguage,
        targetLanguage: liveTargetLanguage,
        device: selectedDevice,
      }));
      node.port.onmessage = (message: MessageEvent<ArrayBuffer>) => {
        const bytes = message.data.byteLength;
        liveHeardSeconds += bytes / 2 / 16000;
        paintStreamLabels();
        if (socket.readyState === WebSocket.OPEN) socket.send(message.data);
      };
      window.clearTimeout(liveAudioWatch);
      liveAudioWatch = window.setTimeout(() => {
        if (liveActive && liveHeardSeconds < 0.2) {
          liveStreamStatus = "No audio is streaming from the microphone.";
          paintStreamLabels();
          setStatus(liveStreamStatus, "error");
        }
      }, 2500);
    });
    socket.addEventListener("message", (message: MessageEvent<string>) => {
      if (typeof message.data !== "string") return;
      try {
        handleStreamEvent(JSON.parse(message.data) as StreamEvent);
      } catch {
        setStatus("Could not read the live stream.", "error");
      }
    });
    socket.addEventListener("close", () => {
      if (liveSocket !== socket) return;
      const unexpected = liveActive;
      const wasStopping = liveStopping;
      liveSocket = null;
      liveActive = false;
      liveStopping = false;
      teardownLiveAudio();
      if ((unexpected || wasStopping) && activeView === "live") render();
      if (unexpected && statusKind !== "error") setStatus("Live session ended.", "info");
      else if (wasStopping && statusKind !== "error") setStatus("Live session ended.", "ok");
    });
    socket.addEventListener("error", () => {
      setStatus("Could not connect to the live stream.", "error");
    });
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    teardownLiveAudio();
    void context.close().catch(() => undefined);
    liveSocket?.close();
    liveSocket = null;
    liveActive = false;
    render();
    setStatus(microphoneError(error), "error");
  }
}

function bindEvents(): void {
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const next = (btn.dataset.view as View) ?? "transcribe";
      if (next !== "live") stopLive(true);
      activeView = next;
      render({ resetScroll: true });
    });
  });

  document.querySelectorAll<HTMLButtonElement>("[data-goto]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const next = (btn.dataset.goto as View) ?? "transcribe";
      if (next !== "live") stopLive(true);
      activeView = next;
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

  document.querySelector<HTMLSelectElement>("#live-source")?.addEventListener("change", (event) => {
    liveSourceLanguage = (event.target as HTMLSelectElement).value;
  });

  document.querySelector<HTMLSelectElement>("#live-target")?.addEventListener("change", (event) => {
    liveTargetLanguage = (event.target as HTMLSelectElement).value;
    if (activeView === "live" && !liveActive) render();
  });

  document.querySelector("#live-toggle")?.addEventListener("click", () => {
    if (liveActive) {
      stopLive(true);
      render();
      setStatus("Stopping…", "info");
      return;
    }
    const context = new AudioContext();
    void context.resume();
    void startLive(context);
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
    await refreshMicrophones().catch(() => undefined);
    navigator.mediaDevices?.addEventListener("devicechange", () => {
      void refreshMicrophones().then(() => {
        if (activeView === "live" && !liveActive) render();
      }).catch(() => undefined);
    });

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
