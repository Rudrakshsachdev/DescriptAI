import { apiClient, ApiError } from "@/lib/api/client";
import {
  BackendConnectionStatus,
  HealthResponse,
  ModelInfoResponse,
  ProductDescriptionRequest,
  ProductDescriptionResponse,
  ReadinessResponse,
} from "@/lib/types/api";

/**
 * Checks API service liveness.
 */
export async function checkHealth(): Promise<HealthResponse> {
  return apiClient<HealthResponse>("/health", { timeoutMs: 5000 });
}

/**
 * Checks whether the underlying Transformer model is mounted and ready for inference.
 */
export async function checkReadiness(): Promise<ReadinessResponse> {
  try {
    return await apiClient<ReadinessResponse>("/health/ready", { timeoutMs: 5000 });
  } catch (err) {
    if (
      err instanceof ApiError &&
      err.statusCode === 503 &&
      err.rawBody &&
      typeof err.rawBody === "object" &&
      "status" in err.rawBody
    ) {
      return err.rawBody as ReadinessResponse;
    }
    throw err;
  }
}

/**
 * Submits structured product information to generate a description.
 */
export async function generateDescription(
  payload: ProductDescriptionRequest
): Promise<ProductDescriptionResponse> {
  return apiClient<ProductDescriptionResponse>("/api/v1/generate", {
    method: "POST",
    body: JSON.stringify(payload),
    timeoutMs: 45000,
  });
}

/**
 * Retrieves safe operational metadata for the active deep learning model.
 */
export async function getModelInfo(): Promise<ModelInfoResponse> {
  return apiClient<ModelInfoResponse>("/api/v1/model/info", { timeoutMs: 10000 });
}

/**
 * Probes the backend to determine consolidated operational status.
 */
export async function getBackendStatus(): Promise<{
  status: BackendConnectionStatus;
  health?: HealthResponse;
  readiness?: ReadinessResponse;
  error?: string;
}> {
  try {
    const health = await checkHealth();
    try {
      const readiness = await checkReadiness();
      if (readiness.model_loaded && readiness.status === "ready") {
        return { status: "online_ready", health, readiness };
      }
      return { status: "online_not_ready", health, readiness };
    } catch {
      return {
        status: "online_not_ready",
        health,
        error: "Backend is running, but the AI model is not ready.",
      };
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Connection failed";
    return { status: "offline", error: message };
  }
}
