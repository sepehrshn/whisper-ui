import "./style.css";
import { checkHealth, transcribeAudio } from "./api.ts";
import { LANGUAGES, MODELS, type TranscriptResult } from "./models.ts";

const AUDIO_TYPES = [
  "audio/mpeg",
  "audio/wav",
  "audio/x-wav",
  "audio/flac",
  "audio/mp4",
  "audio/ogg",
  "audio/webm",
  "audio/aac",
  "audio/x-m4a",
];

const AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".webm", ".aac", ".mp4"];

let selectedModel = "small";
let selectedDevice: "cuda" | "cpu" = "cuda";
let selectedFile: File | null = null;
let lastResult: TranscriptResult | null = null;
let gpuVramGb: number | null = null;
let cudaAvailable = false;

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function isAudioFile(file: File): boolean {
  if (AUDIO_TYPES.includes(file.type)) return true;
  const name = file.name.toLowerCase();
  return AUDIO_EXTENSIONS.some((ext) => name.endsWith(ext));
}

function fileLabel(file: File | null): string {
  if (!file) return "No file selected";
  const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
  return `${file.name} · ${sizeMb} MB`;
}

function modelNeedsRam(model: { vramGb: number }): boolean {
  return cudaAvailable && gpuVramGb != null && model.vramGb > gpuVramGb;
}

function currentModel() {
  return MODELS.find((model) => model.id === selectedModel) ?? MODELS[0];
}

function syncDeviceForModel(): void {
  if (modelNeedsRam(currentModel())) {
    selectedDevice = "cpu";
  }
}

function render(): void {
  const app = document.querySelector<HTMLDivElement>("#app");
  if (!app) return;

  app.innerHTML = `
    <div class="shell">
      <header class="topbar">
        <div class="brand">
          <div class="mark" aria-hidden="true">
            <span></span><span></span><span></span><span></span>
          </div>
          <div>
            <h1>Whisper Transcribe</h1>
            <p>Upload audio and get speech-to-text with OpenAI Whisper</p>
          </div>
        </div>
        <a class="source" href="https://github.com/openai/whisper" target="_blank" rel="noreferrer">
          openai/whisper
        </a>
      </header>

      <p id="status" class="banner">Checking Whisper server…</p>

      <section class="panel">
        <div class="panel-head">
          <h2>Run on</h2>
          <p>GPU is faster for small models. RAM can run Medium, Large, and Turbo.</p>
        </div>
        <div class="device-toggle" id="device-toggle">
          <button class="device-btn ${selectedDevice === "cuda" ? "active" : ""}" data-device="cuda" type="button" ${cudaAvailable ? "" : "disabled"}>
            <strong>GPU</strong>
            <span>${cudaAvailable ? `Faster · ${gpuVramGb ?? "?"} GB VRAM` : "Not available"}</span>
          </button>
          <button class="device-btn ${selectedDevice === "cpu" ? "active" : ""}" data-device="cpu" type="button">
            <strong>RAM</strong>
            <span>Slower · advanced models</span>
          </button>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head">
          <h2>Select model</h2>
          <p>${selectedDevice === "cpu" ? "Running on system RAM. First load of a large model can take several minutes." : "On GPU, tiny/base/small fit best. Pick Medium or larger to switch to RAM."}</p>
        </div>
        <div class="model-grid" id="model-grid">
          ${MODELS.map((model) => {
            const needsRam = modelNeedsRam(model);
            const classes = ["model-card", model.id === selectedModel ? "active" : ""].filter(Boolean).join(" ");
            return `
              <button class="${classes}" data-model="${model.id}" type="button">
                <div class="model-top">
                  <strong>${model.label}</strong>
                  ${model.recommended ? '<span class="pill">Recommended</span>' : ""}
                  ${model.englishOnly ? '<span class="pill muted">English</span>' : ""}
                  ${needsRam ? '<span class="pill muted">Uses RAM</span>' : ""}
                </div>
                <dl>
                  <div><dt>Params</dt><dd>${model.params}</dd></div>
                  <div><dt>Memory</dt><dd>${model.vram}</dd></div>
                  <div><dt>Speed</dt><dd>${model.speed}</dd></div>
                </dl>
              </button>
            `;
          }).join("")}
        </div>
      </section>

      <section class="layout">
        <div class="panel">
          <div class="panel-head">
            <h2>Audio file</h2>
            <p>MP3, WAV, FLAC, M4A, OGG, and similar formats.</p>
          </div>
          <label class="dropzone" id="dropzone">
            <input id="file-input" type="file" accept="audio/*,.mp3,.wav,.flac,.m4a,.ogg,.webm,.aac" hidden />
            <strong>Drop an audio file here</strong>
            <span>or click to browse</span>
            <em id="file-name">${fileLabel(selectedFile)}</em>
          </label>
          <div class="controls">
            <label>
              Task
              <select id="task">
                <option value="transcribe">Transcribe</option>
                <option value="translate">Translate to English</option>
              </select>
            </label>
            <label>
              Language
              <select id="language">
                ${LANGUAGES.map((lang) => `<option value="${lang.id}">${lang.label}</option>`).join("")}
              </select>
            </label>
          </div>
          <button id="run" class="primary" type="button">Transcribe</button>
        </div>

        <div class="panel result-panel">
          <div class="panel-head">
            <h2>Result</h2>
            <div class="result-actions">
              <button id="copy" type="button" ${lastResult ? "" : "disabled"}>Copy text</button>
            </div>
          </div>
          <div id="result">${emptyResult()}</div>
        </div>
      </section>
    </div>
  `;

  bindEvents();
}

function emptyResult(): string {
  return `<p class="empty">Transcript will appear here after you upload a file.</p>`;
}

function resultHtml(result: TranscriptResult): string {
  const segments = result.segments
    .map(
      (segment) => `
        <li>
          <time>${formatTime(segment.start)} – ${formatTime(segment.end)}</time>
          <p>${escapeHtml(segment.text)}</p>
        </li>
      `,
    )
    .join("");

  return `
    <div class="meta">
      <span>Model: ${escapeHtml(result.model)}</span>
      <span>Task: ${escapeHtml(result.task)}</span>
      <span>Device: ${escapeHtml(result.device === "ram" ? "RAM" : result.device === "gpu" ? "GPU" : selectedDevice === "cpu" ? "RAM" : "GPU")}</span>
      <span>Language: ${escapeHtml(result.language ?? "unknown")}</span>
    </div>
    <article class="transcript">${escapeHtml(result.text) || "No speech detected."}</article>
    ${segments ? `<ol class="segments">${segments}</ol>` : ""}
  `;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setStatus(message: string, kind: "info" | "ok" | "error" = "info"): void {
  const status = document.querySelector("#status");
  if (!status) return;
  status.textContent = message;
  status.className = `banner ${kind}`;
}

function bindEvents(): void {
  document.querySelectorAll<HTMLButtonElement>(".device-btn").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.disabled) return;
      const next = button.dataset.device === "cpu" ? "cpu" : "cuda";
      if (next === "cuda" && modelNeedsRam(currentModel())) {
        setStatus(
          `${currentModel().label} needs more than ${gpuVramGb ?? 4} GB VRAM. Staying on RAM, or pick tiny/base/small for GPU.`,
          "info",
        );
        return;
      }
      selectedDevice = next;
      document.querySelectorAll(".device-btn").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      setStatus(
        selectedDevice === "cpu"
          ? "Using system RAM. Advanced models are available."
          : `Using GPU · ${gpuVramGb ?? "?"} GB VRAM`,
        "ok",
      );
    });
  });

  document.querySelectorAll<HTMLButtonElement>(".model-card").forEach((button) => {
    button.addEventListener("click", () => {
      selectedModel = button.dataset.model ?? "small";
      document.querySelectorAll(".model-card").forEach((card) => card.classList.remove("active"));
      button.classList.add("active");
      if (modelNeedsRam(currentModel()) && selectedDevice !== "cpu") {
        selectedDevice = "cpu";
        document.querySelectorAll<HTMLButtonElement>(".device-btn").forEach((item) => {
          item.classList.toggle("active", item.dataset.device === "cpu");
        });
        setStatus(`${currentModel().label} is too large for the GPU. Switched to RAM.`, "info");
      }
    });
  });

  const dropzone = document.querySelector<HTMLLabelElement>("#dropzone");
  const fileInput = document.querySelector<HTMLInputElement>("#file-input");
  const fileName = document.querySelector("#file-name");
  const runButton = document.querySelector<HTMLButtonElement>("#run");
  const copyButton = document.querySelector<HTMLButtonElement>("#copy");
  const resultBox = document.querySelector("#result");

  const assignFile = (file: File | null) => {
    if (file && !isAudioFile(file)) {
      setStatus("Please choose an audio file.", "error");
      return;
    }
    selectedFile = file;
    if (fileName) fileName.textContent = fileLabel(file);
  };

  fileInput?.addEventListener("change", () => {
    assignFile(fileInput.files?.[0] ?? null);
  });

  dropzone?.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("over");
  });

  dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("over"));

  dropzone?.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("over");
    const file = event.dataTransfer?.files[0];
    assignFile(file ?? null);
  });

  copyButton?.addEventListener("click", async () => {
    if (!lastResult?.text) return;
    await navigator.clipboard.writeText(lastResult.text);
    setStatus("Copied transcript to clipboard.", "ok");
  });

  runButton?.addEventListener("click", async () => {
    if (!selectedFile) {
      setStatus("Choose an audio file first.", "error");
      return;
    }

    const task = document.querySelector<HTMLSelectElement>("#task")?.value ?? "transcribe";
    const language = document.querySelector<HTMLSelectElement>("#language")?.value ?? "";

    runButton.disabled = true;
    runButton.textContent = "Working…";
    setStatus(
      selectedDevice === "cpu"
        ? "Loading model into RAM and transcribing. Large models can take several minutes."
        : "Loading model on GPU and transcribing. First run may take a while.",
      "info",
    );

    try {
      lastResult = await transcribeAudio({
        file: selectedFile,
        model: selectedModel,
        task,
        language,
        device: selectedDevice,
      });
      if (resultBox) resultBox.innerHTML = resultHtml(lastResult);
      if (copyButton) copyButton.disabled = !lastResult.text;
      setStatus("Transcription complete.", "ok");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Transcription failed";
      setStatus(message, "error");
    } finally {
      runButton.disabled = false;
      runButton.textContent = "Transcribe";
    }
  });
}

async function boot(): Promise<void> {
  render();
  try {
    const health = await checkHealth();
    cudaAvailable = health.cuda;
    gpuVramGb = health.cuda ? health.vramGb : null;
    if (!cudaAvailable) selectedDevice = "cpu";
    syncDeviceForModel();
    render();
    const where = selectedDevice === "cpu" ? "RAM" : "GPU";
    const gpu = health.cuda
      ? `GPU ${health.gpu ?? "CUDA"} · ${health.vramGb ?? "?"} GB VRAM · or use RAM for larger models`
      : "CPU/RAM mode";
    setStatus(`Whisper server connected · ${gpu} · ready on ${where}`, health.cuda ? "ok" : "info");
  } catch {
    setStatus("Start the Python server on port 8000, then refresh.", "error");
  }
}

void boot();
