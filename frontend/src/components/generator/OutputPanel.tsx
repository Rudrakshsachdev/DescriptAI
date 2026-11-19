"use client";

import * as React from "react";
import { Copy, Check, Sparkles, Clock, Hash, Cpu, Terminal, ArrowRight, RefreshCw, FileText } from "lucide-react";
import { GenerationData } from "@/lib/types/api";
import { formatLatency } from "@/lib/utils";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";

interface OutputPanelProps {
  data: GenerationData | null;
  isLoading: boolean;
  onLoadExample?: () => void;
  onRegenerate?: () => void;
}

export function OutputPanel({ data, isLoading, onLoadExample, onRegenerate }: OutputPanelProps) {
  const [copied, setCopied] = React.useState(false);
  const [showPrompt, setShowPrompt] = React.useState(false);

  const wordCount = React.useMemo(() => {
    if (!data?.generated_description) return 0;
    return data.generated_description.trim().split(/\s+/).filter(Boolean).length;
  }, [data?.generated_description]);

  const handleCopy = async () => {
    if (!data?.generated_description) return;
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(data.generated_description);
      } else {
        // Fallback for non-secure contexts
        const textarea = document.createElement("textarea");
        textarea.value = data.generated_description;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback failure handling
      setCopied(false);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 flex flex-col items-center justify-center min-h-[400px] text-center shadow-xs">
        <div className="relative mb-5">
          <div className="w-14 h-14 rounded-2xl bg-brand-50 border border-brand-200 flex items-center justify-center text-brand-600 animate-pulse">
            <Sparkles className="w-7 h-7" />
          </div>
          <span className="absolute -top-1 -right-1 flex h-3.5 w-3.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-brand-600"></span>
          </span>
        </div>
        <h4 className="text-base font-semibold text-slate-900 mb-1.5">
          Generating Product Copy...
        </h4>
        <p className="text-xs text-slate-500 max-w-sm leading-relaxed mb-6">
          The fine-tuned FLAN-T5 Seq2Seq model is executing beam search decoding on the GPU. This typically takes 500–1,500 ms.
        </p>

        {/* Pulse skeleton placeholder */}
        <div className="w-full max-w-md space-y-2.5">
          <div className="h-3.5 bg-slate-100 rounded-full w-full animate-pulse" />
          <div className="h-3.5 bg-slate-100 rounded-full w-5/6 animate-pulse" />
          <div className="h-3.5 bg-slate-100 rounded-full w-4/6 animate-pulse" />
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-white border border-dashed border-slate-300 rounded-xl p-8 flex flex-col items-center justify-center min-h-[400px] text-center">
        <div className="w-12 h-12 rounded-xl bg-slate-100 text-slate-400 flex items-center justify-center mb-4">
          <Sparkles className="w-6 h-6" />
        </div>
        <h4 className="text-sm font-semibold text-slate-800 mb-1">
          No Description Generated Yet
        </h4>
        <p className="text-xs text-slate-500 max-w-xs leading-relaxed mb-5">
          Fill in the product attributes on the left and click Generate Description to trigger the Transformer model.
        </p>
        {onLoadExample && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onLoadExample}
            rightIcon={<ArrowRight className="w-3.5 h-3.5" />}
          >
            Load Example Product
          </Button>
        )}
      </div>
    );
  }

  return (
    <div className="bg-white border border-slate-200/90 rounded-xl shadow-xs overflow-hidden flex flex-col h-full animate-in fade-in duration-200">
      {/* Header bar */}
      <div className="px-5 py-3.5 bg-slate-50/70 border-b border-slate-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-slate-700 tracking-tight uppercase">
            Generated Description
          </span>
          <Badge variant="brand" size="sm">
            {data.generation_strategy || "beam_search"}
          </Badge>
        </div>

        <div className="flex items-center gap-2">
          {onRegenerate && (
            <button
              type="button"
              onClick={onRegenerate}
              disabled={isLoading}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold rounded-md border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 active:bg-slate-100 shadow-2xs transition-all disabled:opacity-40"
              title="Rerun generation with current settings"
            >
              <RefreshCw className="w-3.5 h-3.5 text-slate-500" />
              <span>Regenerate</span>
            </button>
          )}

          <button
            type="button"
            onClick={handleCopy}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 active:bg-slate-100 shadow-2xs transition-all focus:outline-none focus:ring-2 focus:ring-brand-500"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-emerald-700">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-500" />
                <span>Copy Description</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main text output */}
      <div className="p-6 flex-1 flex flex-col justify-between">
        <div className="prose prose-sm max-w-none text-slate-800 leading-relaxed text-sm font-normal selection:bg-brand-100 selection:text-brand-900 whitespace-pre-wrap">
          {data.generated_description}
        </div>

        {/* Telemetry metadata footer */}
        <div className="mt-8 pt-4 border-t border-slate-100">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-[11px] text-slate-500 font-mono">
            <div className="flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              <span>Latency:</span>
              <span className="font-semibold text-slate-800">
                {formatLatency(data.latency_ms)}
              </span>
            </div>

            <div className="flex items-center gap-1.5">
              <Hash className="w-3.5 h-3.5 text-slate-400" />
              <span>Output:</span>
              <span className="font-semibold text-slate-800">
                {wordCount} words ({data.output_tokens} tokens)
              </span>
            </div>

            <div className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-slate-400" />
              <span>Model:</span>
              <span className="font-semibold text-slate-800 truncate" title={data.model}>
                {data.model.replace("google/", "")}
              </span>
            </div>
          </div>

          {/* Canonical prompt inspector */}
          {data.input_prompt && (
            <div className="mt-3 pt-2">
              <button
                type="button"
                onClick={() => setShowPrompt(!showPrompt)}
                className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-600 transition-colors"
              >
                <Terminal className="w-3 h-3" />
                <span>{showPrompt ? "Hide Formatted Input Prompt" : "Inspect Raw Input Prompt"}</span>
              </button>
              {showPrompt && (
                <div className="mt-2 p-3 bg-slate-900 text-slate-200 rounded-lg text-[11px] font-mono leading-relaxed overflow-x-auto whitespace-pre-wrap border border-slate-800">
                  {data.input_prompt}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
