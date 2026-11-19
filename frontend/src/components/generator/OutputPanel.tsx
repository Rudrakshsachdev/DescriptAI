"use client";

import * as React from "react";
import Link from "next/link";
import {
  Copy,
  Check,
  Sparkles,
  Clock,
  Hash,
  Cpu,
  Terminal,
  RefreshCw,
  FileText,
  Wand2,
  Zap,
} from "lucide-react";
import { GenerationData } from "@/lib/types/api";
import { formatLatency } from "@/lib/utils";
import { Badge } from "@/components/ui/Badge";

interface OutputPanelProps {
  data: GenerationData | null;
  isLoading: boolean;
  onLoadExample?: () => void;
  onRegenerate?: () => void;
}

export function OutputPanel({
  data,
  isLoading,
  onRegenerate,
}: OutputPanelProps) {
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
      setCopied(false);
    }
  };

  return (
    <div className="space-y-3">
      {/* Header matching reference */}
      <div className="flex items-center gap-2.5 px-1">
        <div className="w-7 h-7 rounded-lg bg-[#eef4ff] text-[#0066e6] flex items-center justify-center shrink-0">
          <Wand2 className="w-4 h-4 stroke-[2]" />
        </div>
        <div>
          <h2 className="text-sm font-bold text-slate-900">Your description</h2>
          <p className="text-[11px] text-slate-400">Your draft will appear here.</p>
        </div>
      </div>

      {/* Main Container */}
      <div className="bg-white border border-slate-200/90 rounded-2xl min-h-[520px] flex flex-col p-6 shadow-xs transition-all relative">
        {/* Loading State */}
        {isLoading && (
          <div className="flex-1 flex flex-col items-center justify-center text-center py-12">
            <div className="w-12 h-12 rounded-2xl bg-[#eef4ff] text-[#0066e6] flex items-center justify-center animate-pulse mb-4">
              <Sparkles className="w-6 h-6 animate-spin" />
            </div>
            <h4 className="text-sm font-semibold text-slate-900">
              Generating Product Copy...
            </h4>
            <p className="text-xs text-slate-500 max-w-xs mt-1 leading-relaxed">
              Running beam search decoding through your local FLAN-T5 Seq2Seq model.
            </p>
            <div className="w-48 space-y-2 mt-6">
              <div className="h-2.5 bg-slate-100 rounded-full animate-pulse" />
              <div className="h-2.5 bg-slate-100 rounded-full w-5/6 animate-pulse" />
              <div className="h-2.5 bg-slate-100 rounded-full w-4/6 animate-pulse" />
            </div>
          </div>
        )}

        {/* Empty State matching screenshot exactly */}
        {!isLoading && !data && (
          <div className="flex-1 flex flex-col items-center justify-center text-center py-12">
            <div className="w-12 h-12 rounded-full border border-slate-200/80 bg-slate-50/50 flex items-center justify-center text-slate-400 mx-auto">
              <FileText className="w-5 h-5 stroke-[1.8]" />
            </div>

            <h3 className="font-serif text-lg font-normal text-slate-900 mt-4">
              Ready when you are
            </h3>

            <p className="text-xs text-slate-500 max-w-xs mx-auto mt-1 leading-relaxed">
              Your product description will show up here once you create a draft.
            </p>

            <div className="mt-8 pt-8 border-t border-slate-100 w-full flex justify-center">
              <Link
                href="/model"
                className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-700 transition-colors"
              >
                <Zap className="w-3.5 h-3.5 text-slate-400" />
                <span>Local template mode</span>
                <span className="text-[#0066e6] font-medium ml-1">Details →</span>
              </Link>
            </div>
          </div>
        )}

        {/* Generated Output State */}
        {!isLoading && data && (
          <div className="flex-1 flex flex-col justify-between space-y-6 animate-in fade-in duration-200">
            {/* Action Bar */}
            <div className="flex items-center justify-between pb-4 border-b border-slate-100">
              <div className="flex items-center gap-2">
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
                    title="Rerun generation"
                  >
                    <RefreshCw className="w-3.5 h-3.5 text-slate-500" />
                    <span>Regenerate</span>
                  </button>
                )}

                <button
                  type="button"
                  onClick={handleCopy}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-md border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 active:bg-slate-100 shadow-2xs transition-all focus:outline-none focus:ring-2 focus:ring-[#0066e6]"
                >
                  {copied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                      <span className="text-emerald-700">Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5 text-slate-500" />
                      <span>Copy description</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Generated Copy text */}
            <div className="text-slate-800 text-sm leading-relaxed whitespace-pre-wrap font-normal selection:bg-blue-100 selection:text-blue-900 flex-1">
              {data.generated_description}
            </div>

            {/* Telemetry footer */}
            <div className="pt-4 border-t border-slate-100 space-y-3">
              <div className="grid grid-cols-3 gap-2 text-[11px] text-slate-500 font-mono">
                <div className="flex items-center gap-1">
                  <Clock className="w-3.5 h-3.5 text-slate-400" />
                  <span className="font-semibold text-slate-800">
                    {formatLatency(data.latency_ms)}
                  </span>
                </div>

                <div className="flex items-center gap-1">
                  <Hash className="w-3.5 h-3.5 text-slate-400" />
                  <span className="font-semibold text-slate-800">
                    {wordCount} words ({data.output_tokens} tokens)
                  </span>
                </div>

                <div className="flex items-center gap-1 truncate">
                  <Cpu className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                  <span className="font-semibold text-slate-800 truncate" title={data.model}>
                    {data.model.replace("google/", "")}
                  </span>
                </div>
              </div>

              {/* Raw Prompt Inspector */}
              {data.input_prompt && (
                <div className="pt-2">
                  <button
                    type="button"
                    onClick={() => setShowPrompt(!showPrompt)}
                    className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-600 transition-colors"
                  >
                    <Terminal className="w-3 h-3" />
                    <span>
                      {showPrompt
                        ? "Hide Formatted Input Prompt"
                        : "Inspect Raw Input Prompt"}
                    </span>
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
        )}
      </div>
    </div>
  );
}
