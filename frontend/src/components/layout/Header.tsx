"use client";

import * as React from "react";
import Link from "next/link";
import { HealthIndicator } from "@/components/ui/HealthIndicator";
import { Menu, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface HeaderProps {
  title?: string;
  description?: string;
  onOpenSidebar: () => void;
}

export function Header({ onOpenSidebar }: HeaderProps) {
  return (
    <header className="h-14 px-4 sm:px-6 lg:px-8 border-b border-slate-200/80 bg-white flex items-center justify-between sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onOpenSidebar}
          className="p-1.5 -ml-1 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg lg:hidden"
          aria-label="Open navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Clean breadcrumbs from reference */}
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Link href="/" className="hover:text-slate-800 transition-colors">
            Descript AI
          </Link>
          <span className="text-slate-300">/</span>
          <span className="text-slate-700 font-medium">Workspace</span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <HealthIndicator />

        <Link href="/generate">
          <Button
            size="sm"
            className="bg-[#0066e6] hover:bg-[#0052cc] text-white text-xs font-semibold px-3 py-1.5 rounded-lg shadow-xs flex items-center gap-1.5"
          >
            <Sparkles className="w-3.5 h-3.5 text-white/90" />
            <span>New description</span>
          </Button>
        </Link>
      </div>
    </header>
  );
}
