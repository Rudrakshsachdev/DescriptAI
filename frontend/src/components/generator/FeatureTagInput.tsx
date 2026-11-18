"use client";

import * as React from "react";
import { Plus, X } from "lucide-react";

interface FeatureTagInputProps {
  features: string[];
  onChange: (features: string[]) => void;
  maxFeatures?: number;
  disabled?: boolean;
}

export function FeatureTagInput({
  features,
  onChange,
  maxFeatures = 20,
  disabled = false,
}: FeatureTagInputProps) {
  const [inputValue, setInputValue] = React.useState("");

  const handleAdd = () => {
    const trimmed = inputValue.trim();
    if (!trimmed) return;
    if (features.length >= maxFeatures) return;
    if (features.includes(trimmed)) {
      setInputValue("");
      return;
    }
    onChange([...features, trimmed]);
    setInputValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAdd();
    } else if (e.key === "Backspace" && !inputValue && features.length > 0) {
      onChange(features.slice(0, -1));
    }
  };

  const handleRemove = (index: number) => {
    onChange(features.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700">
          Key Features & Bullet Points
        </label>
        <span className="text-[11px] text-slate-400 font-mono">
          {features.length}/{maxFeatures}
        </span>
      </div>

      <div className="flex flex-wrap gap-1.5 p-2 bg-white border border-slate-300 rounded-lg shadow-sm min-h-[44px] focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-500/20 transition-all">
        {features.map((feature, index) => (
          <span
            key={index}
            className="inline-flex items-center gap-1 px-2.5 py-1 bg-slate-100 text-slate-800 text-xs rounded-md border border-slate-200"
          >
            <span>{feature}</span>
            <button
              type="button"
              disabled={disabled}
              onClick={() => handleRemove(index)}
              className="text-slate-400 hover:text-slate-600 focus:outline-none"
              aria-label={`Remove feature ${feature}`}
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </span>
        ))}

        {features.length < maxFeatures && (
          <div className="flex items-center flex-1 min-w-[140px]">
            <input
              type="text"
              disabled={disabled}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={features.length === 0 ? "e.g., Infused with 2% pure hyaluronic acid" : "Add another feature..."}
              className="w-full text-xs text-slate-900 placeholder:text-slate-400 bg-transparent outline-none py-1"
            />
            {inputValue.trim() && (
              <button
                type="button"
                onClick={handleAdd}
                className="p-1 text-brand-600 hover:bg-brand-50 rounded"
                title="Add feature"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        )}
      </div>
      <p className="text-[11px] text-slate-400">
        Type a bullet point feature and press Enter to add.
      </p>
    </div>
  );
}
