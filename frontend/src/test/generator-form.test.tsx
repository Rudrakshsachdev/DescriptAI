import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { GeneratorForm } from "@/components/generator/GeneratorForm";
import * as services from "@/lib/api/services";
import type { ProductDescriptionResponse } from "@/lib/types/api";

vi.mock("@/lib/api/services", () => ({
  generateDescription: vi.fn(),
}));

describe("GeneratorForm Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the initial form elements", () => {
    render(<GeneratorForm />);

    expect(screen.getByText("Product Information")).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText(/e\.g\., Organic Hydrating Face Cream/i)
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /generate description/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /sample product/i })
    ).toBeInTheDocument();
  });

  it("displays client validation error when submitting with empty product name", async () => {
    render(<GeneratorForm />);

    const submitBtn = screen.getByRole("button", {
      name: /generate description/i,
    });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(
        screen.getByText("Product name is required.")
      ).toBeInTheDocument();
    });

    expect(services.generateDescription).not.toHaveBeenCalled();
  });

  it("loads sample product attributes when clicking Sample Product", async () => {
    render(<GeneratorForm />);

    const sampleBtn = screen.getByRole("button", { name: /sample product/i });
    fireEvent.click(sampleBtn);

    const nameInput = screen.getByPlaceholderText(
      /e\.g\., Organic Hydrating Face Cream/i
    ) as HTMLInputElement;

    expect(nameInput.value).toBe(
      "Hydrating Daily Face Moisturizer with Hyaluronic Acid"
    );
  });

  it("submits generation request and displays output on success", async () => {
    const mockOutput: ProductDescriptionResponse = {
      success: true,
      data: {
        generated_description:
          "Transform your skincare routine with this potent hydrating daily face moisturizer formulated with ceramides.",
        model: "google/flan-t5-small",
        generation_strategy: "beam_search",
        latency_ms: 215.4,
        output_tokens: 32,
        input_prompt:
          "Generate product description:\nProduct: Hydrating Daily Face Moisturizer with Hyaluronic Acid",
        metadata: {
          num_beams: 2,
        },
      },
    };

    vi.mocked(services.generateDescription).mockResolvedValueOnce(mockOutput);

    render(<GeneratorForm />);

    // Load sample for quick valid data
    const sampleBtn = screen.getByRole("button", { name: /sample product/i });
    fireEvent.click(sampleBtn);

    // Click submit
    const submitBtn = screen.getByRole("button", {
      name: /generate description/i,
    });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(services.generateDescription).toHaveBeenCalledTimes(1);
    });

    // Check output rendered
    await waitFor(() => {
      expect(
        screen.getByText(/Transform your skincare routine with this potent/i)
      ).toBeInTheDocument();
      expect(screen.getByText(/215 ms/i)).toBeInTheDocument();
      expect(screen.getByText(/32 tokens/i)).toBeInTheDocument();
    });
  });

  it("displays backend error message when generation fails", async () => {
    vi.mocked(services.generateDescription).mockRejectedValueOnce(
      new Error("Inference execution timed out on GPU.")
    );

    render(<GeneratorForm />);

    const sampleBtn = screen.getByRole("button", { name: /sample product/i });
    fireEvent.click(sampleBtn);

    const submitBtn = screen.getByRole("button", {
      name: /generate description/i,
    });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(
        screen.getByText(/Inference execution timed out on GPU\./i)
      ).toBeInTheDocument();
    });
  });

  it("copies generated output to clipboard", async () => {
    const mockOutput: ProductDescriptionResponse = {
      success: true,
      data: {
        generated_description: "Copyable description text.",
        model: "google/flan-t5-small",
        generation_strategy: "beam_search",
        latency_ms: 150,
        output_tokens: 10,
      },
    };

    vi.mocked(services.generateDescription).mockResolvedValueOnce(mockOutput);

    render(<GeneratorForm />);

    const sampleBtn = screen.getByRole("button", { name: /sample product/i });
    fireEvent.click(sampleBtn);

    const submitBtn = screen.getByRole("button", {
      name: /generate description/i,
    });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText("Copyable description text.")).toBeInTheDocument();
    });

    const copyBtn = screen.getByRole("button", { name: /copy description/i });
    fireEvent.click(copyBtn);

    await waitFor(() => {
      expect(screen.getByText("Copied!")).toBeInTheDocument();
    });

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      "Copyable description text."
    );
  });

  it("clears form inputs when clicking reset button", async () => {
    render(<GeneratorForm />);

    const sampleBtn = screen.getByRole("button", { name: /sample product/i });
    fireEvent.click(sampleBtn);

    const nameInput = screen.getByPlaceholderText(
      /e\.g\., Organic Hydrating Face Cream/i
    ) as HTMLInputElement;
    expect(nameInput.value).not.toBe("");

    const clearBtn = screen.getByTitle("Clear all fields");
    fireEvent.click(clearBtn);

    expect(nameInput.value).toBe("");
  });

  it("triggers regeneration when clicking Regenerate button in output panel", async () => {
    const mockOutput: ProductDescriptionResponse = {
      success: true,
      data: {
        generated_description: "Initial description.",
        model: "google/flan-t5-small",
        generation_strategy: "beam_search",
        latency_ms: 150,
        output_tokens: 10,
      },
    };

    vi.mocked(services.generateDescription).mockResolvedValue(mockOutput);

    render(<GeneratorForm />);

    const sampleBtn = screen.getByRole("button", { name: /sample product/i });
    fireEvent.click(sampleBtn);

    const submitBtn = screen.getByRole("button", {
      name: /generate description/i,
    });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(screen.getByText("Initial description.")).toBeInTheDocument();
    });

    const regenBtn = screen.getByRole("button", { name: /regenerate/i });
    fireEvent.click(regenBtn);

    await waitFor(() => {
      expect(services.generateDescription).toHaveBeenCalledTimes(2);
    });
  });
});
