"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { X } from "lucide-react";

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
      icon: (active: boolean) => (
        <svg
          className={cn("w-5 h-5 shrink-0 transition-colors", active ? "text-[#0066e6]" : "text-[#64748b]")}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z" />
          <path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65" />
          <path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65" />
        </svg>
      ),
      active: pathname === "/",
    },
    {
      name: "Generator",
      href: "/generate",
      icon: (active: boolean) => (
        <svg
          className={cn("w-5 h-5 shrink-0 transition-colors", active ? "text-[#0066e6]" : "text-[#64748b]")}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z" />
          <circle cx="4" cy="4" r="1" fill="currentColor" />
        </svg>
      ),
      active: pathname === "/generate",
    },
    {
      name: "Model diagnostics",
      href: "/model",
      icon: (active: boolean) => (
        <svg
          className={cn("w-5 h-5 shrink-0 transition-colors", active ? "text-[#0066e6]" : "text-[#64748b]")}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
        </svg>
      ),
      active: pathname === "/model",
    },
  ];

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/30 backdrop-blur-xs lg:hidden transition-opacity"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar container matching screenshot */}
      <aside
        className={cn(
          "fixed top-0 bottom-0 left-0 z-50 w-64 bg-[#f8fafd] border-r border-[#eaecf0] flex flex-col transition-transform duration-200 ease-in-out lg:translate-x-0 lg:static shrink-0 select-none",
          isOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {/* Top Logo and Branding */}
        <div className="h-20 px-5 flex items-center justify-between border-b border-[#eaecf0]">
          <Link
            href="/"
            onClick={onClose}
            className="flex items-center gap-3.5 group"
          >
            {/* Blue squircle logo icon */}
            <div className="w-10 h-10 rounded-[14px] bg-[#0066e6] text-white flex items-center justify-center shrink-0 shadow-2xs">
              <svg
                className="w-5 h-5 text-white"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
                <line x1="10" y1="9" x2="8" y2="9" />
              </svg>
            </div>

            {/* Typography */}
            <div className="flex flex-col justify-center">
              <span className="text-[17px] font-bold text-[#0f172a] tracking-tight leading-none">
                descript
              </span>
              <span className="text-[10px] font-bold text-[#8492a6] tracking-[0.16em] uppercase mt-1 leading-none">
                AI COPY STUDIO
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

        {/* Workspace Navigation Section */}
        <div className="flex-1 px-4 pt-7 space-y-1">
          <div className="px-3 mb-3 text-[11px] font-bold tracking-[0.14em] text-[#64748b] uppercase">
            WORKSPACE
          </div>

          <nav className="space-y-1.5" aria-label="Main navigation">
            {navigation.map((item) => {
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  onClick={onClose}
                  className={cn(
                    "flex items-center gap-3.5 px-3.5 py-2.5 rounded-xl text-[14px] transition-all",
                    item.active
                      ? "bg-[#e8f1fc] text-[#0f2b48] font-semibold shadow-2xs"
                      : "text-[#475569] hover:bg-slate-100/70 hover:text-slate-900 font-medium"
                  )}
                  aria-current={item.active ? "page" : undefined}
                >
                  {item.icon(item.active)}
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>
        </div>
      </aside>
    </>
  );
}
