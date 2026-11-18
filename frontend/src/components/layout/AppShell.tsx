"use client";

import * as React from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { usePathname } from "next/navigation";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = React.useState(false);
  const pathname = usePathname();

  const getPageMeta = () => {
    switch (pathname) {
      case "/":
        return {
          title: "System Overview",
          description: "Descript AI Product Copy Generation Platform",
        };
      case "/generate":
        return {
          title: "Product Description Generator",
          description: "Generate e-commerce descriptions with fine-tuned FLAN-T5",
        };
      case "/model":
        return {
          title: "Model & Architecture Diagnostics",
          description: "Inspect operational model checkpoint and inference configuration",
        };
      default:
        return {
          title: "Descript AI",
          description: "AI-Powered Product Description Generator",
        };
    }
  };

  const meta = getPageMeta();

  return (
    <div className="min-h-screen flex bg-slate-50 text-slate-900">
      {/* Sidebar navigation */}
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Main column */}
      <div className="flex-1 flex flex-col min-w-0">
        <Header
          title={meta.title}
          description={meta.description}
          onOpenSidebar={() => setSidebarOpen(true)}
        />
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
