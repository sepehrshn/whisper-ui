import type { TranscriptResult } from "./models.ts";

export type HealthStatus = {
  ok: boolean;
  cuda: boolean;
  device: string;
  gpu: string | null;
  vramGb: number | null;
  loadedModel: string | null;
};

export async function checkHealth(): Promise<HealthStatus> {
  const response = await fetch("/api/health");
  if (!response.ok) {
    throw new Error("Whisper server is not running");
  }
  return response.json();
}

export async function transcribeAudio(options: {
  file: File;
  model: string;
  task: string;
  language: string;
  device: "cuda" | "cpu";
}): Promise<TranscriptResult> {
  const body = new FormData();
  body.append("file", options.file);
  body.append("model", options.model);
  body.append("task", options.task);
  body.append("language", options.language);
  body.append("device", options.device);

  const response = await fetch("/api/transcribe", {
    method: "POST",
    body,
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      // ignore JSON parse errors
    }
    throw new Error(detail);
  }

  return response.json();
}
