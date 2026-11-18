/**
 * TypeScript API type definitions strictly aligned with FastAPI backend Pydantic schemas.
 */

export interface GenerationConfig {
  num_beams?: number;
  max_new_tokens?: number;
  min_new_tokens?: number;
  no_repeat_ngram_size?: number;
  length_penalty?: number;
  do_sample?: boolean;
  temperature?: number;
  top_p?: number;
}

export interface ProductDescriptionRequest {
  product_name: string;
  category?: string | null;
  brand?: string | null;
  features?: string[] | null;
  specifications?: Record<string, string> | null;
  audience?: string | null;
  generation_config?: GenerationConfig | null;
}

export interface GenerationData {
  generated_description: string;
  model: string;
  generation_strategy: string;
  latency_ms: number;
  output_tokens: number;
  input_prompt?: string | null;
  metadata?: Record<string, unknown>;
}

export interface ProductDescriptionResponse {
  success: boolean;
  data: GenerationData;
}

export interface ModelInfoData {
  model_id: string;
  architecture: string;
  checkpoint: string;
  device: string;
  total_parameters: number;
  precision: string;
  supported_strategies: string[];
  default_config: Record<string, unknown>;
  is_ready: boolean;
}

export interface ModelInfoResponse {
  success: boolean;
  data: ModelInfoData;
}

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  timestamp: string;
}

export interface ReadinessResponse {
  status: "ready" | "not_ready";
  model_loaded: boolean;
  model_path: string;
  device: string;
  half_precision: boolean;
  startup_error?: string | null;
  timestamp: string;
}

export interface ApiErrorDetail {
  loc?: (string | number)[];
  msg?: string;
  type?: string;
}

export interface ApiErrorResponse {
  error: string;
  message: string;
  details?: ApiErrorDetail[] | Record<string, unknown>;
}

export type BackendConnectionStatus =
  | "checking"
  | "online_ready"
  | "online_not_ready"
  | "offline";
