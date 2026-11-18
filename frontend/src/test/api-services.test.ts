import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  checkHealth,
  checkReadiness,
  generateDescription,
  getModelInfo,
  getBackendStatus,
} from "@/lib/api/services";
import { ApiError } from "@/lib/api/client";
import type {
  HealthResponse,
  ReadinessResponse,
  ProductDescriptionResponse,
  ModelInfoResponse,
} from "@/lib/types/api";

describe("API Client & Services", () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it("checkHealth returns health payload on success", async () => {
    const mockHealth: HealthResponse = {
      status: "healthy",
      version: "0.1.0",
      environment: "production",
      timestamp: "2026-11-18T12:00:00Z",
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockHealth,
    } as Response);

    const result = await checkHealth();
    expect(result).toEqual(mockHealth);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/health"),
      expect.any(Object)
    );
  });

  it("checkReadiness returns readiness status on 200", async () => {
    const mockReadiness: ReadinessResponse = {
      status: "ready",
      model_loaded: true,
      model_path: "checkpoints/sft/best_checkpoint",
      device: "cuda:0",
      half_precision: true,
      timestamp: "2026-11-18T12:00:00Z",
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockReadiness,
    } as Response);

    const result = await checkReadiness();
    expect(result.status).toBe("ready");
    expect(result.model_loaded).toBe(true);
  });

  it("checkReadiness handles 503 not_ready gracefully", async () => {
    const mockNotReady: ReadinessResponse = {
      status: "not_ready",
      model_loaded: false,
      model_path: "checkpoints/sft/best_checkpoint",
      device: "cpu",
      half_precision: false,
      startup_error: "Loading weights",
      timestamp: "2026-11-18T12:00:00Z",
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => mockNotReady,
    } as Response);

    const result = await checkReadiness();
    expect(result.status).toBe("not_ready");
    expect(result.model_loaded).toBe(false);
  });

  it("generateDescription successfully returns generated copy", async () => {
    const mockResponse: ProductDescriptionResponse = {
      success: true,
      data: {
        generated_description:
          "Experience radiant glow with our hydrating organic serum.",
        model: "google/flan-t5-small",
        generation_strategy: "beam_search",
        latency_ms: 184.2,
        output_tokens: 28,
        input_prompt:
          "Generate product description:\nProduct: Radiant Facial Serum",
        metadata: {
          num_beams: 2,
        },
      },
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockResponse,
    } as Response);

    const result = await generateDescription({
      product_name: "Radiant Facial Serum",
      features: ["Hydrating", "Organic"],
    });

    expect(result.success).toBe(true);
    expect(result.data.generated_description).toContain("radiant glow");
    expect(result.data.latency_ms).toBe(184.2);
  });

  it("generateDescription throws ApiError on validation failure (422)", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({
        error: "Unprocessable Entity",
        message: "Validation failed",
        details: [
          {
            loc: ["body", "product_name"],
            msg: "Field required",
            type: "missing",
          },
        ],
      }),
    } as Response);

    await expect(
      generateDescription({ product_name: "" })
    ).rejects.toThrow(ApiError);
  });

  it("getModelInfo retrieves operational metadata", async () => {
    const mockModelInfo: ModelInfoResponse = {
      success: true,
      data: {
        model_id: "google/flan-t5-small",
        architecture: "Encoder-Decoder Transformer (Seq2Seq)",
        checkpoint: "checkpoints/sft/best_checkpoint",
        device: "cuda:0",
        total_parameters: 76961152,
        precision: "bfloat16",
        supported_strategies: ["beam_search", "greedy", "sampling"],
        default_config: { num_beams: 2, max_new_tokens: 128 },
        is_ready: true,
      },
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockModelInfo,
    } as Response);

    const result = await getModelInfo();
    expect(result.success).toBe(true);
    expect(result.data.total_parameters).toBe(76961152);
    expect(result.data.device).toBe("cuda:0");
  });

  it("getBackendStatus returns correct status enum", async () => {
    // 1. Online Ready
    global.fetch = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ status: "healthy" }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ status: "ready", model_loaded: true }),
      } as Response);

    const statusReady = await getBackendStatus();
    expect(statusReady.status).toBe("online_ready");

    // 2. Online Not Ready
    global.fetch = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ status: "healthy" }),
      } as Response)
      .mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: async () => ({ status: "not_ready", model_loaded: false }),
      } as Response);

    const statusNotReady = await getBackendStatus();
    expect(statusNotReady.status).toBe("online_not_ready");

    // 3. Offline
    global.fetch = vi.fn().mockRejectedValue(new Error("Connection refused"));

    const statusOffline = await getBackendStatus();
    expect(statusOffline.status).toBe("offline");
  });
});
