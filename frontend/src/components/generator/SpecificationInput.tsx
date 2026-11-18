"use client";

import * as React from "react";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface SpecificationInputProps {
  specifications: Record<string, string>;
  onChange: (specifications: Record<string, string>) => void;
  maxEntries?: number;
  disabled?: boolean;
}

export function SpecificationInput({
  specifications,
  onChange,
  maxEntries = 20,
  disabled = false,
}: SpecificationInputProps) {
  const entries = Object.entries(specifications);
  const [newKey, setNewKey] = React.useState("");
  const [newValue, setNewValue] = React.useState("");

  const handleAdd = () => {
    const k = newKey.trim();
    const v = newValue.trim();
    if (!k || !v) return;
    if (entries.length >= maxEntries) return;

    onChange({
      ...specifications,
      [k]: v,
    });
    setNewKey("");
    setNewValue("");
  };

  const handleRemove = (keyToRemove: string) => {
    const next = { ...specifications };
    delete next[keyToRemove];
    onChange(next);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-700">
          Technical Specifications
        </label>
        <span className="text-[11px] text-slate-400 font-mono">
          {entries.length}/{maxEntries}
        </span>
      </div>

      {entries.length > 0 && (
        <div className="space-y-1.5">
          {entries.map(([key, val]) => (
            <div
              key={key}
              className="flex items-center gap-2 p-2 bg-slate-50 border border-slate-200 rounded-lg text-xs"
            >
              <span className="font-semibold text-slate-700 min-w-[100px] truncate">
                {key}:
              </span>
              <span className="text-slate-600 flex-1 truncate">{val}</span>
              <button
                type="button"
                disabled={disabled}
                onClick={() => handleRemove(key)}
                className="text-slate-400 hover:text-rose-600 p-1 rounded"
                title={`Delete specification ${key}`}
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}

      {entries.length < maxEntries && (
        <div className="flex items-center gap-2">
          <input
            type="text"
            disabled={disabled}
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            placeholder="Spec Name (e.g., Volume)"
            className="w-1/3 px-3 py-1.5 text-xs bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-1 focus:ring-brand-500"
          />
          <input
            type="text"
            disabled={disabled}
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            placeholder="Value (e.g., 50ml / 1.7 fl oz)"
            className="flex-1 px-3 py-1.5 text-xs bg-white border border-slate-300 rounded-md focus:outline-none focus:ring-1 focus:ring-brand-500"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleAdd();
              }
            }}
          />
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={disabled || !newKey.trim() || !newValue.trim()}
            onClick={handleAdd}
            leftIcon={<Plus className="w-3.5 h-3.5" />}
          >
            Add
          </Button>
        </div>
      )}
      <p className="text-[11px] text-slate-400">
        Optional structured key-value attributes (e.g. Skin Type, Capacity, Finish).
      </p>
    </div>
  );
}
