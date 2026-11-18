"use client";

import * as React from "react";
import {
  Cpu,
  Layers,
  Zap,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  HardDrive,
  Sliders,
  Settings,
  Binary,
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
    <div className="space-y-6 pb-12">
      {/* Page Title & Refresh */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">
            Model & Architecture Diagnostics
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Real-time metadata retrieved directly from the FastAPI inference engine.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={fetchInfo}
          disabled={isLoading}
          className="self-start sm:self-auto"
        >
          <RefreshCw
            className={`h-4 w-4 mr-2 ${isLoading ? "animate-spin text-brand-600" : ""}`}
          />
          {isLoading ? "Refreshing..." : "Refresh Status"}
        </Button>
      </div>

      {/* Loading Skeleton */}
      {isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-pulse">
          <div className="h-64 rounded-xl bg-slate-200" />
          <div className="h-64 rounded-xl bg-slate-200" />
          <div className="h-48 rounded-xl bg-slate-200" />
          <div className="h-48 rounded-xl bg-slate-200" />
        </div>
      )}

      {/* Error state */}
      {!isLoading && error && (
        <Card className="border-red-200 bg-red-50/50">
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

      {/* Main Content */}
      {!isLoading && modelInfo && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Identity & Checkpoint */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <div className="flex items-center gap-2">
                <Layers className="h-5 w-5 text-brand-600" />
                <CardTitle className="text-base">Model Identity</CardTitle>
              </div>
              <Badge variant={modelInfo.is_ready ? "success" : "warning"}>
                {modelInfo.is_ready ? "Ready for Inference" : "Initializing"}
              </Badge>
            </CardHeader>
            <CardContent className="space-y-4 pt-2">
              <div>
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Model Identifier
                </span>
                <p className="text-sm font-semibold text-slate-900 mt-0.5 font-mono">
                  {modelInfo.model_id}
                </p>
              </div>

              <div>
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Architecture
                </span>
                <p className="text-sm text-slate-900 mt-0.5">
                  {modelInfo.architecture}
                </p>
              </div>

              <div>
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Checkpoint Location
                </span>
                <p className="text-xs text-slate-700 mt-0.5 font-mono bg-slate-100 p-2 rounded border border-slate-200 break-all">
                  {modelInfo.checkpoint}
                </p>
              </div>

              <div>
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Supported Strategies
                </span>
                <div className="mt-1 flex flex-wrap gap-1.5">
                  {modelInfo.supported_strategies.map((strat) => (
                    <Badge key={strat} variant="neutral">
                      {strat}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Hardware & Runtime */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <div className="flex items-center gap-2">
                <Cpu className="h-5 w-5 text-indigo-600" />
                <CardTitle className="text-base">Runtime Environment</CardTitle>
              </div>
              <Badge variant="brand">Local Execution</Badge>
            </CardHeader>
            <CardContent className="space-y-4 pt-2">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                    Execution Device
                  </span>
                  <p className="text-sm font-semibold text-slate-900 mt-0.5 font-mono">
                    {modelInfo.device}
                  </p>
                </div>
                <div>
                  <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                    Precision Format
                  </span>
                  <p className="text-sm font-semibold text-slate-900 mt-0.5 font-mono">
                    {modelInfo.precision}
                  </p>
                </div>
              </div>

              <div>
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Total Parameters
                </span>
                <p className="text-2xl font-bold text-slate-900 mt-0.5">
                  {formatNumber(modelInfo.total_parameters)}
                </p>
                <p className="text-xs text-slate-500 mt-0.5">
                  Trainable and non-trainable weights
                </p>
              </div>

              <div className="border-t border-slate-100 pt-3">
                <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Hardware Constraints
                </span>
                <ul className="mt-1.5 space-y-1 text-xs text-slate-600">
                  <li>• GPU Target: NVIDIA RTX 4050 Laptop (6 GB VRAM)</li>
                  <li>• Max Input Sequence: 512 tokens</li>
                  <li>• Max Target Sequence: 256 tokens</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          {/* Default Production Decoding Hyperparameters */}
          <Card className="md:col-span-2">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <div className="flex items-center gap-2">
                <Sliders className="h-5 w-5 text-amber-600" />
                <CardTitle className="text-base">
                  Production Decoding Configuration (Defaults)
                </CardTitle>
              </div>
              <Badge variant="neutral">Beam Search (Width 2)</Badge>
            </CardHeader>
            <CardContent className="pt-2">
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
                {Object.entries(modelInfo.default_config).map(([key, val]) => (
                  <div
                    key={key}
                    className="rounded-lg bg-slate-50 p-3 border border-slate-100"
                  >
                    <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider block truncate">
                      {key.replace(/_/g, " ")}
                    </span>
                    <span className="mt-1 text-sm font-mono font-semibold text-slate-900 block">
                      {String(val)}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Theoretical Foundations & Reference Dimensions */}
          <Card className="md:col-span-2">
            <CardHeader className="pb-2">
              <div className="flex items-center gap-2">
                <Binary className="h-5 w-5 text-slate-600" />
                <CardTitle className="text-base">
                  FLAN-T5 Architecture Specifications
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent className="pt-2">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
                <div className="space-y-1">
                  <span className="text-slate-500 font-medium">Vocabulary Size</span>
                  <p className="font-semibold text-slate-900">32,128 tokens</p>
                </div>
                <div className="space-y-1">
                  <span className="text-slate-500 font-medium">Hidden Dimension (d_model)</span>
                  <p className="font-semibold text-slate-900">512</p>
                </div>
                <div className="space-y-1">
                  <span className="text-slate-500 font-medium">Attention Heads</span>
                  <p className="font-semibold text-slate-900">8 (64 dim / head)</p>
                </div>
                <div className="space-y-1">
                  <span className="text-slate-500 font-medium">Feedforward Dim (d_ff)</span>
                  <p className="font-semibold text-slate-900">2,048</p>
                </div>
                <div className="space-y-1">
                  <span className="text-slate-500 font-medium">Encoder Layers</span>
                  <p className="font-semibold text-slate-900">6 Transformer blocks</p>
                </div>
                <div className="space-y-1">
                  <span className="text-slate-500 font-medium">Decoder Layers</span>
                  <p className="font-semibold text-slate-900">6 Transformer blocks</p>
                </div>
                <div className="space-y-1">
                  <span className="text-slate-500 font-medium">Positional Encoding</span>
                  <p className="font-semibold text-slate-900">Relative Bias (32 buckets)</p>
                </div>
                <div className="space-y-1">
                  <span className="text-slate-500 font-medium">Activation Function</span>
                  <p className="font-semibold text-slate-900">GELU / GEGLU</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
