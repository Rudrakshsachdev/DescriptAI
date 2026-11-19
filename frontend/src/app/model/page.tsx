"use client";

import * as React from "react";
import Link from "next/link";
import {
  Cpu,
  Layers,
  Zap,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Sliders,
  Binary,
  ArrowLeft,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { getModelInfo } from "@/lib/api/services";
import type { ModelInfoData } from "@/lib/types/api";
import { formatNumber } from "@/lib/utils";

export default function ModelInfoPage() {
  const [modelInfo, setModelInfo] = React.useState<ModelInfoData | null>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  const fetchInfo = React.useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await getModelInfo();
      if (res.success && res.data) {
        setModelInfo(res.data);
      } else {
        setError("Failed to parse model operational metadata.");
      }
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Backend service unreachable.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  React.useEffect(() => {
    fetchInfo();
  }, [fetchInfo]);

  return (
    <div className="space-y-6 pb-12 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div>
          <span className="text-[11px] font-bold text-[#0066e6] tracking-wider uppercase block">
            SYSTEM DIAGNOSTICS
          </span>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight mt-1">
            Model diagnostics
          </h1>
          <p className="text-sm text-slate-500 mt-1 max-w-2xl">
            Real-time metadata retrieved directly from the FastAPI inference engine.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <Link href="/generate">
            <Button variant="outline" size="sm" className="text-xs">
              <ArrowLeft className="w-3.5 h-3.5 mr-1.5" />
              Generator
            </Button>
          </Link>
          <Button
            variant="outline"
            size="sm"
            onClick={fetchInfo}
            disabled={isLoading}
            className="text-xs"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 mr-1.5 ${
                isLoading ? "animate-spin text-[#0066e6]" : ""
              }`}
            />
            {isLoading ? "Refreshing..." : "Refresh Status"}
          </Button>
        </div>
      </div>

      {/* Loading Skeleton */}
      {isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-pulse">
          <div className="h-64 rounded-2xl bg-slate-100" />
          <div className="h-64 rounded-2xl bg-slate-100" />
          <div className="h-48 rounded-2xl bg-slate-100 md:col-span-2" />
        </div>
      )}

      {/* Error State */}
      {!isLoading && error && (
        <Card className="border-red-200 bg-red-50/50 rounded-2xl">
          <CardContent className="pt-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 text-red-600 mt-0.5 shrink-0" />
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-red-900">
                  Unable to connect to model serving endpoint
                </h3>
                <p className="text-xs text-red-700">
                  {error}. Make sure the FastAPI application is running locally on port 8000.
                </p>
                <div className="pt-2">
                  <Button size="sm" variant="outline" onClick={fetchInfo}>
                    Try Again
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Diagnostics Content */}
      {!isLoading && modelInfo && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Identity & Checkpoint */}
          <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-[#eef4ff] text-[#0066e6] flex items-center justify-center">
                  <Layers className="w-4 h-4" />
                </div>
                <h3 className="text-sm font-bold text-slate-900">Model Identity</h3>
              </div>
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium border ${
                  modelInfo.is_ready
                    ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                    : "bg-amber-50 text-amber-800 border-amber-200"
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    modelInfo.is_ready ? "bg-emerald-500" : "bg-amber-500"
                  }`}
                />
                {modelInfo.is_ready ? "Ready for Inference" : "Initializing"}
              </span>
            </div>

            <div className="space-y-3.5 text-xs">
              <div>
                <span className="text-slate-400 font-medium uppercase tracking-wider text-[10px] block">
                  Model Identifier
                </span>
                <span className="text-slate-900 font-mono font-semibold text-sm mt-0.5 block">
                  {modelInfo.model_id}
                </span>
              </div>

              <div>
                <span className="text-slate-400 font-medium uppercase tracking-wider text-[10px] block">
                  Architecture
                </span>
                <span className="text-slate-900 font-medium text-xs mt-0.5 block">
                  {modelInfo.architecture}
                </span>
              </div>

              <div>
                <span className="text-slate-400 font-medium uppercase tracking-wider text-[10px] block">
                  Checkpoint Location
                </span>
                <span className="font-mono text-slate-700 bg-slate-50 border border-slate-200 px-2.5 py-1.5 rounded-lg block mt-0.5 break-all">
                  {modelInfo.checkpoint}
                </span>
              </div>

              <div>
                <span className="text-slate-400 font-medium uppercase tracking-wider text-[10px] block mb-1">
                  Supported Strategies
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {modelInfo.supported_strategies.map((strat) => (
                    <span
                      key={strat}
                      className="px-2 py-0.5 rounded-md bg-slate-100 text-slate-700 font-mono text-[11px]"
                    >
                      {strat}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Hardware & Runtime */}
          <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-[#eef4ff] text-[#0066e6] flex items-center justify-center">
                  <Cpu className="w-4 h-4" />
                </div>
                <h3 className="text-sm font-bold text-slate-900">Runtime Environment</h3>
              </div>
              <span className="px-2 py-0.5 rounded-full text-[11px] font-medium bg-blue-50 text-[#0066e6] border border-blue-200">
                Local Execution
              </span>
            </div>

            <div className="space-y-4 text-xs">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-slate-400 font-medium uppercase tracking-wider text-[10px] block">
                    Execution Device
                  </span>
                  <span className="text-slate-900 font-mono font-bold text-sm mt-0.5 block">
                    {modelInfo.device}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 font-medium uppercase tracking-wider text-[10px] block">
                    Precision Format
                  </span>
                  <span className="text-slate-900 font-mono font-bold text-sm mt-0.5 block">
                    {modelInfo.precision}
                  </span>
                </div>
              </div>

              <div>
                <span className="text-slate-400 font-medium uppercase tracking-wider text-[10px] block">
                  Total Parameters
                </span>
                <span className="text-2xl font-bold text-slate-900 font-mono mt-0.5 block">
                  {formatNumber(modelInfo.total_parameters)}
                </span>
                <span className="text-[11px] text-slate-400">
                  Encoder-decoder weights loaded in VRAM/RAM
                </span>
              </div>

              <div className="border-t border-slate-100 pt-3">
                <span className="text-slate-400 font-medium uppercase tracking-wider text-[10px] block mb-1">
                  Hardware Constraints
                </span>
                <ul className="space-y-1 text-slate-600 text-[11px]">
                  <li>• Target Hardware: NVIDIA RTX 4050 (6 GB VRAM)</li>
                  <li>• Maximum Sequence Length: 512 input tokens / 256 target tokens</li>
                  <li>• Batch Capacity: 1 physical concurrent worker</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Default Production Decoding Hyperparameters */}
          <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs space-y-4 md:col-span-2">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-[#eef4ff] text-[#0066e6] flex items-center justify-center">
                  <Sliders className="w-4 h-4" />
                </div>
                <h3 className="text-sm font-bold text-slate-900">
                  Production Decoding Configuration (Defaults)
                </h3>
              </div>
              <span className="text-[11px] font-medium text-slate-500">
                Beam Search (Width 2)
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {Object.entries(modelInfo.default_config).map(([k, v]) => (
                <div
                  key={k}
                  className="rounded-xl bg-slate-50/80 p-3 border border-slate-200/60"
                >
                  <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block truncate">
                    {k.replace(/_/g, " ")}
                  </span>
                  <span className="text-sm font-mono font-bold text-slate-900 mt-1 block">
                    {String(v)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* FLAN-T5 Architecture Blueprint */}
          <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs space-y-4 md:col-span-2">
            <div className="flex items-center gap-2.5 pb-3 border-b border-slate-100">
              <div className="w-7 h-7 rounded-lg bg-slate-100 text-slate-700 flex items-center justify-center">
                <Binary className="w-4 h-4" />
              </div>
              <h3 className="text-sm font-bold text-slate-900">
                FLAN-T5 Architecture Blueprint
              </h3>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div className="space-y-0.5">
                <span className="text-slate-400 text-[11px]">Vocabulary Size</span>
                <p className="font-semibold text-slate-900">32,128 tokens</p>
              </div>
              <div className="space-y-0.5">
                <span className="text-slate-400 text-[11px]">Hidden Dimension</span>
                <p className="font-semibold text-slate-900">512</p>
              </div>
              <div className="space-y-0.5">
                <span className="text-slate-400 text-[11px]">Attention Heads</span>
                <p className="font-semibold text-slate-900">8 (64 dim / head)</p>
              </div>
              <div className="space-y-0.5">
                <span className="text-slate-400 text-[11px]">Feedforward Dim</span>
                <p className="font-semibold text-slate-900">2,048</p>
              </div>
              <div className="space-y-0.5">
                <span className="text-slate-400 text-[11px]">Encoder Layers</span>
                <p className="font-semibold text-slate-900">6 blocks</p>
              </div>
              <div className="space-y-0.5">
                <span className="text-slate-400 text-[11px]">Decoder Layers</span>
                <p className="font-semibold text-slate-900">6 blocks</p>
              </div>
              <div className="space-y-0.5">
                <span className="text-slate-400 text-[11px]">Positional Encoding</span>
                <p className="font-semibold text-slate-900">Relative Bias</p>
              </div>
              <div className="space-y-0.5">
                <span className="text-slate-400 text-[11px]">Activation Function</span>
                <p className="font-semibold text-slate-900">GELU / GEGLU</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
