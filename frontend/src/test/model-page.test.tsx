import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ModelInfoPage from "@/app/model/page";
import * as services from "@/lib/api/services";
import type { ModelInfoResponse } from "@/lib/types/api";

vi.mock("@/lib/api/services", () => ({
  getModelInfo: vi.fn(),
}));

describe("ModelInfoPage Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders model metadata when API call succeeds", async () => {
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
        default_config: {
          num_beams: 2,
          max_new_tokens: 128,
          min_new_tokens: 10,
          no_repeat_ngram_size: 3,
        },
        is_ready: true,
      },
    };

    vi.mocked(services.getModelInfo).mockResolvedValueOnce(mockModelInfo);

    render(<ModelInfoPage />);

    // Check loading state first or wait for content
    await waitFor(() => {
      expect(screen.getByText("google/flan-t5-small")).toBeInTheDocument();
      expect(
        screen.getByText("Encoder-Decoder Transformer (Seq2Seq)")
      ).toBeInTheDocument();
      expect(screen.getByText("76,961,152")).toBeInTheDocument();
      expect(screen.getByText("cuda:0")).toBeInTheDocument();
      expect(screen.getByText("bfloat16")).toBeInTheDocument();
      expect(screen.getByText("Ready for Inference")).toBeInTheDocument();
    });
  });

  it("renders error state with retry button when backend fails", async () => {
    vi.mocked(services.getModelInfo).mockRejectedValueOnce(
      new Error("Failed to connect to backend service")
    );

    render(<ModelInfoPage />);

    await waitFor(() => {
      expect(
        screen.getByText(/Unable to connect to model serving endpoint/i)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Failed to connect to backend service/i)
      ).toBeInTheDocument();
    });

    // Test retry
    const retryBtn = screen.getByRole("button", { name: /try again/i });
    expect(retryBtn).toBeInTheDocument();

    const mockModelInfo: ModelInfoResponse = {
      success: true,
      data: {
        model_id: "google/flan-t5-small",
        architecture: "Encoder-Decoder Transformer (Seq2Seq)",
        checkpoint: "checkpoints/sft/best_checkpoint",
        device: "cpu",
        total_parameters: 76961152,
        precision: "float32",
        supported_strategies: ["beam_search"],
        default_config: { num_beams: 2 },
        is_ready: true,
      },
    };

    vi.mocked(services.getModelInfo).mockResolvedValueOnce(mockModelInfo);
    fireEvent.click(retryBtn);

    await waitFor(() => {
      expect(screen.getByText("google/flan-t5-small")).toBeInTheDocument();
    });
  });
});
