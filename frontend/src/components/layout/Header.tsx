"use client";

import * as React from "react";
import { HealthIndicator } from "@/components/ui/HealthIndicator";
import { Menu } from "lucide-react";

interface HeaderProps {
  title: string;
  description?: string;
  onOpenSidebar: () => void;
}

export function Header({ title, description, onOpenSidebar }: HeaderProps) {
  return (
    <header className="h-16 px-4 sm:px-6 lg:px-8 border-b border-slate-200/90 bg-white/95 backdrop-blur-xs flex items-center justify-between sticky top-0 z-30">
      <div className="flex items-center gap-3 min-w-0">
        <button
          type="button"
          onClick={onOpenSidebar}
          className="p-2 -ml-2 text-slate-500 hover:text-slate-700 hover:bg-slate-100 rounded-lg lg:hidden"
          aria-label="Open navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="min-w-0">
          <h1 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight truncate">
            {title}
          </h1>
          {description && (
            <p className="text-xs text-slate-500 truncate hidden sm:block">
              {description}
            </p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <HealthIndicator />
      </div>
    </header>
  );
}
