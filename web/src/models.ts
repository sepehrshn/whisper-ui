export type WhisperModel = {
  id: string;
  label: string;
  params: string;
  vram: string;
  vramGb: number;
  speed: string;
  englishOnly: boolean;
  recommended?: boolean;
};

export type TranscriptSegment = {
  id: number;
  start: number;
  end: number;
  text: string;
};

export type TranscriptResult = {
  text: string;
  language: string | null;
  model: string;
  task: string;
  provider?: string;
  device?: string;
  segments: TranscriptSegment[];
};

export const MODELS: WhisperModel[] = [
  { id: "tiny", label: "Tiny", params: "39 M", vram: "~1 GB", vramGb: 1, speed: "~10x", englishOnly: false },
  { id: "tiny.en", label: "Tiny EN", params: "39 M", vram: "~1 GB", vramGb: 1, speed: "~10x", englishOnly: true },
  { id: "base", label: "Base", params: "74 M", vram: "~1 GB", vramGb: 1, speed: "~7x", englishOnly: false },
  { id: "base.en", label: "Base EN", params: "74 M", vram: "~1 GB", vramGb: 1, speed: "~7x", englishOnly: true },
  { id: "small", label: "Small", params: "244 M", vram: "~2 GB", vramGb: 2, speed: "~4x", englishOnly: false, recommended: true },
  { id: "small.en", label: "Small EN", params: "244 M", vram: "~2 GB", vramGb: 2, speed: "~4x", englishOnly: true },
  { id: "medium", label: "Medium", params: "769 M", vram: "~5 GB RAM", vramGb: 5, speed: "~2x", englishOnly: false },
  { id: "medium.en", label: "Medium EN", params: "769 M", vram: "~5 GB RAM", vramGb: 5, speed: "~2x", englishOnly: true },
  { id: "large", label: "Large", params: "1550 M", vram: "~10 GB RAM", vramGb: 10, speed: "1x", englishOnly: false },
  { id: "turbo", label: "Turbo", params: "809 M", vram: "~6 GB RAM", vramGb: 6, speed: "~8x", englishOnly: false },
];

export const LANGUAGES = [
  { id: "", label: "Auto detect" },
  { id: "en", label: "English" },
  { id: "fa", label: "Persian" },
  { id: "ar", label: "Arabic" },
  { id: "zh", label: "Chinese" },
  { id: "es", label: "Spanish" },
  { id: "fr", label: "French" },
  { id: "de", label: "German" },
  { id: "hi", label: "Hindi" },
  { id: "ja", label: "Japanese" },
  { id: "ko", label: "Korean" },
  { id: "pt", label: "Portuguese" },
  { id: "ru", label: "Russian" },
  { id: "tr", label: "Turkish" },
];

export const TARGET_LANGUAGES = [
  { id: "", label: "Transcript only" },
  ...LANGUAGES.filter((language) => language.id),
];
