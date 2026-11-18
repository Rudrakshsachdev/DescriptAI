"use client";

import * as React from "react";
import { ChevronDown, ChevronUp, Sliders, RotateCcw } from "lucide-react";
import { GenerationConfig } from "@/lib/types/api";

interface ConfigPanelProps {
  config: GenerationConfig;
  onChange: (config: GenerationConfig) => void;
  disabled?: boolean;
}

export const DEFAULT_CONFIG: Required<GenerationConfig> = {
  num_beams: 2,
  max_new_tokens: 128,
  min_new_tokens: 10,
  no_repeat_ngram_size: 3,
  length_penalty: 1.0,
  do_sample: false,
  temperature: 1.0,
  top_p: 1.0,
};

export function ConfigPanel({ config, onChange, disabled }: ConfigPanelProps) {
  const [isOpen, setIsOpen] = React.useState(false);

  const update = <K extends keyof GenerationConfig>(key: K, value: GenerationConfig[K]) => {
    onChange({ ...config, [key]: value });
  };

  const handleReset = () => {
    onChange({ ...DEFAULT_CONFIG });
  };

  const isCustomized =
    (config.num_beams ?? 2) !== DEFAULT_CONFIG.num_beams ||
    (config.max_new_tokens ?? 128) !== DEFAULT_CONFIG.max_new_tokens ||
    (config.do_sample ?? false) !== DEFAULT_CONFIG.do_sample ||
    (config.length_penalty ?? 1.0) !== DEFAULT_CONFIG.length_penalty;

  return (
    <div className="border border-slate-200 rounded-xl bg-slate-50/70 overflow-hidden transition-all">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between text-left text-xs font-semibold text-slate-700 hover:bg-slate-100/60 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-brand-600" />
          <span>Decoding & Generation Settings</span>
          {isCustomized ? (
            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-brand-100 text-brand-800">
              Customized
            </span>
          ) : (
            <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-slate-200/80 text-slate-600">
              Beam-2 (Recommended)
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isOpen ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </div>
      </button>

      {isOpen && (
        <div className="p-4 border-t border-slate-200/80 space-y-4 bg-white text-xs text-slate-700 animate-in fade-in duration-150">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Number of Beams */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="font-semibold text-slate-700">Beam Search Width (`num_beams`)</label>
                <span className="font-mono text-slate-500 font-semibold">{config.num_beams ?? 2}</span>
              </div>
              <input
                type="range"
                min={1}
                max={5}
                step={1}
                disabled={disabled}
                value={config.num_beams ?? 2}
                onChange={(e) => update("num_beams", parseInt(e.target.value, 10))}
                className="w-full accent-brand-600"
              />
              <p className="text-[11px] text-slate-400">1 = Greedy search; 2–5 = Beam search (2 is optimal).</p>
            </div>

            {/* Max Output Tokens */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="font-semibold text-slate-700">Max Length (`max_new_tokens`)</label>
                <span className="font-mono text-slate-500 font-semibold">{config.max_new_tokens ?? 128}</span>
              </div>
              <input
                type="range"
                min={20}
                max={256}
                step={8}
                disabled={disabled}
                value={config.max_new_tokens ?? 128}
                onChange={(e) => update("max_new_tokens", parseInt(e.target.value, 10))}
                className="w-full accent-brand-600"
              />
              <p className="text-[11px] text-slate-400">Upper token limit (128 tokens ≈ 100 words).</p>
            </div>

            {/* Repetition Penalty / No Repeat N-Gram */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="font-semibold text-slate-700">No-Repeat N-Gram (`no_repeat_ngram_size`)</label>
                <span className="font-mono text-slate-500 font-semibold">{config.no_repeat_ngram_size ?? 3}</span>
              </div>
              <input
                type="range"
                min={0}
                max={5}
                step={1}
                disabled={disabled}
                value={config.no_repeat_ngram_size ?? 3}
                onChange={(e) => update("no_repeat_ngram_size", parseInt(e.target.value, 10))}
                className="w-full accent-brand-600"
              />
              <p className="text-[11px] text-slate-400">Suppresses repetitive phrases of length N.</p>
            </div>

            {/* Length Penalty */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="font-semibold text-slate-700">Length Penalty (`length_penalty`)</label>
                <span className="font-mono text-slate-500 font-semibold">{(config.length_penalty ?? 1.0).toFixed(1)}</span>
              </div>
              <input
                type="range"
                min={0.5}
                max={2.0}
                step={0.1}
                disabled={disabled}
                value={config.length_penalty ?? 1.0}
                onChange={(e) => update("length_penalty", parseFloat(e.target.value))}
                className="w-full accent-brand-600"
              />
              <p className="text-[11px] text-slate-400">&gt; 1.0 encourages longer output; &lt; 1.0 encourages concise output.</p>
            </div>
          </div>

          <div className="pt-2 border-t border-slate-100 flex items-center justify-between">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                disabled={disabled}
                checked={config.do_sample ?? false}
                onChange={(e) => update("do_sample", e.target.checked)}
                className="rounded text-brand-600 focus:ring-brand-500"
              />
              <span className="text-xs text-slate-700 font-medium">Enable Stochastic Sampling (`do_sample`)</span>
            </label>

            <button
              type="button"
              disabled={disabled || !isCustomized}
              onClick={handleReset}
              className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800 disabled:opacity-40 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Defaults</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
