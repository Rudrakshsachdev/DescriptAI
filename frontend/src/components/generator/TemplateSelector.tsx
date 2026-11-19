"use client";

import * as React from "react";
import {
  Watch,
  Droplets,
  Headphones,
  Briefcase,
  Coffee,
  Lamp,
  RotateCcw,
  Sparkles,
  Check,
} from "lucide-react";
import { ProductTemplate } from "@/lib/constants/templates";
import { cn } from "@/lib/utils";

interface TemplateSelectorProps {
  templates: ProductTemplate[];
  selectedTemplateId: string | null;
  onSelectTemplate: (template: ProductTemplate) => void;
  onClear: () => void;
  disabled?: boolean;
}

const TEMPLATE_ICONS: Record<
  ProductTemplate["iconName"],
  React.ComponentType<{ className?: string }>
> = {
  Watch,
  Droplets,
  Headphones,
  Briefcase,
  Coffee,
  Lamp,
};

export function TemplateSelector({
  templates,
  selectedTemplateId,
  onSelectTemplate,
  onClear,
  disabled = false,
}: TemplateSelectorProps) {
  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Sparkles className="w-3.5 h-3.5 text-[#0066e6]" />
          <span className="text-[11px] font-bold text-slate-700 uppercase tracking-wider">
            Pre-configured product templates
          </span>
        </div>

        <button
          type="button"
          onClick={onClear}
          disabled={disabled}
          className="text-xs text-slate-500 hover:text-slate-800 flex items-center gap-1.5 transition-colors font-medium disabled:opacity-50"
          title="Start with a blank form"
        >
          <RotateCcw className="w-3 h-3 text-slate-400" />
          <span>Blank form</span>
        </button>
      </div>

      {/* Templates Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        {templates.map((tpl) => {
          const Icon = TEMPLATE_ICONS[tpl.iconName] || Sparkles;
          const isSelected = selectedTemplateId === tpl.id;

          return (
            <button
              key={tpl.id}
              type="button"
              disabled={disabled}
              onClick={() => onSelectTemplate(tpl)}
              aria-pressed={isSelected}
              aria-label={`Load ${tpl.name} template`}
              title={`Load template: ${tpl.name} (${tpl.category})`}
              className={cn(
                "p-2.5 rounded-xl border text-left transition-all flex flex-col justify-between gap-2 select-none group relative",
                isSelected
                  ? "border-[#0066e6] bg-[#eef4ff] text-[#0066e6] ring-1 ring-[#0066e6] shadow-2xs"
                  : "border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50"
              )}
            >
              <div className="flex items-center justify-between w-full">
                <div
                  className={cn(
                    "w-7 h-7 rounded-lg flex items-center justify-center shrink-0 transition-colors",
                    isSelected
                      ? "bg-[#0066e6] text-white"
                      : "bg-slate-100 text-slate-600 group-hover:bg-slate-200/80"
                  )}
                >
                  <Icon className="w-3.5 h-3.5" />
                </div>

                {isSelected && (
                  <span className="w-4 h-4 rounded-full bg-[#0066e6] text-white flex items-center justify-center shrink-0">
                    <Check className="w-2.5 h-2.5 stroke-[3]" />
                  </span>
                )}
              </div>

              <div className="min-w-0">
                <span
                  className={cn(
                    "text-xs font-semibold block truncate leading-tight",
                    isSelected ? "text-[#0066e6]" : "text-slate-900"
                  )}
                >
                  {tpl.shortLabel}
                </span>
                <span className="text-[10px] text-slate-400 block truncate mt-0.5">
                  {tpl.category}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
