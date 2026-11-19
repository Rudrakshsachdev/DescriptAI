import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Sidebar } from "@/components/layout/Sidebar";
import * as navigation from "next/navigation";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  usePathname: vi.fn(),
}));

describe("Sidebar Navigation", () => {
  it("renders all operational navigation links", () => {
    vi.mocked(navigation.usePathname).mockReturnValue("/");
    const onClose = vi.fn();

    render(<Sidebar isOpen={true} onClose={onClose} />);

    expect(screen.getByRole("link", { name: /overview/i })).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /generator/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /model diagnostics/i })
    ).toBeInTheDocument();
  });

  it("marks the active link based on current pathname", () => {
    vi.mocked(navigation.usePathname).mockReturnValue("/generate");
    const onClose = vi.fn();

    render(<Sidebar isOpen={true} onClose={onClose} />);

    const generateLink = screen.getByRole("link", {
      name: /generator/i,
    });
    expect(generateLink).toHaveAttribute("aria-current", "page");

    const overviewLink = screen.getByRole("link", { name: /overview/i });
    expect(overviewLink).not.toHaveAttribute("aria-current");
  });

  it("does not render dead placeholder or unfinished navigation items", () => {
    vi.mocked(navigation.usePathname).mockReturnValue("/");
    const onClose = vi.fn();

    render(<Sidebar isOpen={true} onClose={onClose} />);

    expect(screen.queryByText(/history & logs/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/step 2/i)).not.toBeInTheDocument();
  });

  it("calls onClose when mobile close button or backdrop is clicked", () => {
    vi.mocked(navigation.usePathname).mockReturnValue("/");
    const onClose = vi.fn();

    render(<Sidebar isOpen={true} onClose={onClose} />);

    const closeBtn = screen.getByLabelText(/close navigation/i);
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalled();
  });
});
