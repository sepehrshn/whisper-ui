import type { TranscriptResult } from "./models.ts";

export type HealthStatus = {
  ok: boolean;
  cuda: boolean;
  device: string;
  gpu: string | null;
  vramGb: number | null;
  loadedModel: string | null;
};

export type ProviderInfo = {
  id: string;
  name: string;
  requiresApiKey: boolean;
  configured: boolean;
  description: string;
  models: Array<{ id: string; label: string; recommended?: boolean }>;
};

export type DependencyReport = {
  ready: boolean;
  issues: string[];
  ffmpeg: { ok: boolean; path: string | null; version?: string };
  python: { ok: boolean; executable: string; version: string };
  packages: { ok: boolean; missing: string[]; venv: boolean };
  cuda: { ok: boolean; available: boolean; gpu: string | null; vramGb?: number | null };
  disk: { ok: boolean; freeGb: number };
  frontend: { ok: boolean; path: string };
};

export type AppSettings = {
  defaultProvider: string;
  configuredKeys: Record<string, boolean>;
  configPath: string;
  secureStorage: boolean;
};

export async function checkHealth(): Promise<HealthStatus> {
  const response = await fetch("/api/health");
  if (!response.ok) throw new Error("Whisper server is not running");
  return response.json();
}

export async function fetchDependencies(): Promise<DependencyReport> {
  const response = await fetch("/api/dependencies");
  if (!response.ok) throw new Error("Could not check dependencies");
  return response.json();
}

export async function installDependencies(options: { python?: boolean; ffmpeg?: boolean } = {}): Promise<unknown> {
  const response = await fetch("/api/setup/install", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ python: options.python ?? true, ffmpeg: options.ffmpeg ?? true }),
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(payload.detail ?? "Dependency install failed");
  }
  return response.json();
}

export async function fetchProviders(): Promise<ProviderInfo[]> {
  const response = await fetch("/api/providers");
  if (!response.ok) throw new Error("Could not load providers");
  const payload = (await response.json()) as { providers: ProviderInfo[] };
  return payload.providers;
}

export async function fetchSettings(): Promise<AppSettings> {
  const response = await fetch("/api/settings");
  if (!response.ok) throw new Error("Could not load settings");
  return response.json();
}

export async function saveDefaultProvider(provider: string): Promise<void> {
  const response = await fetch("/api/settings/default-provider", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider }),
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(payload.detail ?? "Could not save engine");
  }
}

export async function saveApiKey(provider: string, apiKey: string): Promise<AppSettings> {
  const response = await fetch("/api/settings/keys", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, apiKey }),
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(payload.detail ?? "Could not save API key");
  }
  return response.json();
}

export async function transcribeAudio(options: {
  file: File;
  provider: string;
  model: string;
  task: string;
  language: string;
  device: "cuda" | "cpu";
}): Promise<TranscriptResult> {
  const body = new FormData();
  body.append("file", options.file);
  body.append("provider", options.provider);
  body.append("model", options.model);
  body.append("task", options.task);
  body.append("language", options.language);
  body.append("device", options.device);

  const response = await fetch("/api/transcribe", { method: "POST", body });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }

  return response.json();
}
