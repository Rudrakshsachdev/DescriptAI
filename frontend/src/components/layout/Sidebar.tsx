"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  Sparkles,
  LayoutDashboard,
  PenTool,
  Cpu,
  History,
  X,
  Bot,
} from "lucide-react";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export function Sidebar({ isOpen, onClose }: SidebarProps) {
  const pathname = usePathname();

  const navigation = [
    {
      name: "Overview",
      href: "/",
      icon: LayoutDashboard,
      active: pathname === "/",
    },
    {
      name: "Generate Description",
      href: "/generate",
      icon: PenTool,
      active: pathname === "/generate",
    },
    {
      name: "Model Information",
      href: "/model",
      icon: Cpu,
      active: pathname === "/model",
    },
  ];

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/40 backdrop-blur-sm lg:hidden transition-opacity"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar container */}
      <aside
        className={cn(
          "fixed top-0 bottom-0 left-0 z-50 w-64 bg-white border-r border-slate-200/90 flex flex-col transition-transform duration-200 ease-in-out lg:translate-x-0 lg:static",
          isOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {/* Brand header */}
        <div className="h-16 px-6 flex items-center justify-between border-b border-slate-100">
          <Link
            href="/"
            onClick={onClose}
            className="flex items-center gap-2.5 font-bold text-slate-900 group"
          >
            <div className="w-8 h-8 rounded-lg bg-brand-600 text-white flex items-center justify-center shadow-sm shadow-brand-500/20 group-hover:bg-brand-700 transition-colors">
              <Sparkles className="w-4 h-4" />
            </div>
            <div className="flex flex-col">
              <span className="text-base tracking-tight font-extrabold text-slate-900">
                Descript <span className="text-brand-600">AI</span>
              </span>
              <span className="text-[10px] font-semibold text-slate-400 tracking-wider uppercase -mt-0.5">
                Product Copy Engine
              </span>
            </div>
          </Link>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-600 rounded-lg lg:hidden"
            aria-label="Close navigation"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation list */}
        <div className="flex-1 px-3 py-6 space-y-6 overflow-y-auto">
          <div>
            <div className="px-3 mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Workspace
            </div>
            <nav className="space-y-1">
              {navigation.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    onClick={onClose}
                    className={cn(
                      "flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded-lg transition-colors group",
                      item.active
                        ? "bg-brand-50 text-brand-700 font-semibold"
                        : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                    )}
                  >
                    <Icon
                      className={cn(
                        "w-4 h-4 transition-colors",
                        item.active
                          ? "text-brand-600"
                          : "text-slate-400 group-hover:text-slate-600"
                      )}
                    />
                    <span>{item.name}</span>
                  </Link>
                );
              })}
            </nav>
          </div>

          <div>
            <div className="px-3 mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-400">
              Persistence
            </div>
            <div className="space-y-1">
              <div
                className="flex items-center justify-between px-3 py-2.5 text-sm font-medium rounded-lg text-slate-400 bg-slate-50/50 cursor-not-allowed select-none"
                title="Generation history requires Phase 3 Step 2 database persistence."
              >
                <div className="flex items-center gap-3">
                  <History className="w-4 h-4 text-slate-300" />
                  <span>History & Logs</span>
                </div>
                <span className="text-[10px] font-semibold tracking-wider uppercase px-1.5 py-0.5 rounded bg-slate-100 text-slate-500">
                  Step 2
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* System identity footer */}
        <div className="p-4 border-t border-slate-100 bg-slate-50/50">
          <div className="p-3 bg-white rounded-lg border border-slate-200/80 shadow-xs flex items-center gap-2.5">
            <div className="p-1.5 rounded-md bg-brand-50 text-brand-700">
              <Bot className="w-4 h-4" />
            </div>
            <div className="flex flex-col min-w-0">
              <span className="text-xs font-semibold text-slate-800 truncate">
                FLAN-T5 Small (60M)
              </span>
              <span className="text-[10px] text-slate-500 truncate">
                Beam Search (2 Beams)
              </span>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
