"use client";

import * as React from "react";
import Link from "next/link";
import {
  Sparkles,
  Cpu,
  Zap,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Server,
  RefreshCw,
  Sliders,
  Layers,
  FileText,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { getBackendStatus, getModelInfo } from "@/lib/api/services";
import type { BackendConnectionStatus, ModelInfoData } from "@/lib/types/api";
import { formatNumber } from "@/lib/utils";

export default function OverviewPage() {
  const [status, setStatus] = React.useState<BackendConnectionStatus>("checking");
  const [modelInfo, setModelInfo] = React.useState<ModelInfoData | null>(null);
  const [isRefreshing, setIsRefreshing] = React.useState(false);

  const fetchStatus = React.useCallback(async () => {
    setIsRefreshing(true);
    try {
      const conn = await getBackendStatus();
      setStatus(conn.status);
      if (conn.status === "online_ready") {
        const infoRes = await getModelInfo();
        if (infoRes.success) {
          setModelInfo(infoRes.data);
        }
      } else {
        setModelInfo(null);
      }
    } catch {
      setStatus("offline");
      setModelInfo(null);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  React.useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return (
    <div className="space-y-8 pb-12">
      {/* Hero Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-800 to-brand-950 p-8 sm:p-10 text-white shadow-xl border border-slate-700/50">
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 rounded-full bg-brand-500/10 px-3 py-1 text-xs font-semibold text-brand-300 ring-1 ring-inset ring-brand-500/20">
            <Sparkles className="h-3.5 w-3.5 text-brand-400" />
            <span>Phase 3 — Production Frontend Foundation</span>
          </div>
          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-white">
            AI-Powered Product Description Generator
          </h1>
          <p className="text-slate-300 text-base sm:text-lg leading-relaxed">
            Local Seq2Seq Transformer pipeline fine-tuned on the Amazon Reviews 2023 All_Beauty dataset.
            Transforms structured product attributes into professional, marketing-grade e-commerce copy.
          </p>
          <div className="pt-2 flex flex-wrap gap-3">
            <Link href="/generate">
              <Button size="lg" className="shadow-lg shadow-brand-500/20">
                <Sparkles className="h-4 w-4 mr-2" />
                Launch Generator
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </Link>
            <Link href="/model">
              <Button
                variant="outline"
                size="lg"
                className="bg-slate-800/60 border-slate-600 text-slate-100 hover:bg-slate-700/80 hover:text-white"
              >
                <Cpu className="h-4 w-4 mr-2" />
                Model Diagnostics
              </Button>
            </Link>
          </div>
        </div>

        {/* Decorative background glow */}
        <div
          aria-hidden="true"
          className="absolute -top-24 -right-24 h-96 w-96 rounded-full bg-brand-500/20 blur-3xl pointer-events-none"
        />
        <div
          aria-hidden="true"
          className="absolute -bottom-24 -left-24 h-96 w-96 rounded-full bg-indigo-500/10 blur-3xl pointer-events-none"
        />
      </div>

      {/* Backend Live Status Card */}
      <Card className="border-slate-200">
        <CardHeader className="flex flex-row items-center justify-between pb-2 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <Server className="h-5 w-5 text-slate-600" />
            <CardTitle className="text-base font-semibold text-slate-900">
              Operational Backend Status
            </CardTitle>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchStatus}
            disabled={isRefreshing}
            className="text-xs text-slate-600 hover:text-slate-900"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 mr-1.5 ${isRefreshing ? "animate-spin" : ""}`}
            />
            {isRefreshing ? "Checking..." : "Refresh"}
          </Button>
        </CardHeader>
        <CardContent className="pt-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="rounded-lg bg-slate-50 p-3 border border-slate-100">
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                API Connection
              </p>
              <div className="mt-1 flex items-center gap-2">
                {status === "checking" && (
                  <Badge variant="neutral">Checking...</Badge>
                )}
                {status === "online_ready" && (
                  <Badge variant="success" className="gap-1">
                    <CheckCircle2 className="h-3 w-3" />
                    Online & Ready
                  </Badge>
                )}
                {status === "online_not_ready" && (
                  <Badge variant="warning" className="gap-1">
                    <AlertTriangle className="h-3 w-3" />
                    Model Loading
                  </Badge>
                )}
                {status === "offline" && (
                  <Badge variant="error" className="gap-1">
                    <AlertTriangle className="h-3 w-3" />
                    Offline
                  </Badge>
                )}
              </div>
            </div>

            <div className="rounded-lg bg-slate-50 p-3 border border-slate-100">
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                Serving Model
              </p>
              <p className="mt-1 text-sm font-semibold text-slate-900 truncate">
                {modelInfo ? modelInfo.model_id : "google/flan-t5-small"}
              </p>
            </div>

            <div className="rounded-lg bg-slate-50 p-3 border border-slate-100">
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                Hardware & Device
              </p>
              <p className="mt-1 text-sm font-semibold text-slate-900">
                {modelInfo ? `${modelInfo.device} (${modelInfo.precision})` : "cuda:0 / cpu"}
              </p>
            </div>

            <div className="rounded-lg bg-slate-50 p-3 border border-slate-100">
              <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                Parameters
              </p>
              <p className="mt-1 text-sm font-semibold text-slate-900">
                {modelInfo ? formatNumber(modelInfo.total_parameters) : "76,961,152"}
              </p>
            </div>
          </div>

          {status === "offline" && (
            <div className="mt-4 rounded-lg bg-amber-50 p-3 border border-amber-200 text-xs text-amber-800 flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
              <div>
                <p className="font-semibold">Backend Unreachable</p>
                <p className="mt-0.5">
                  The FastAPI backend is not responding at <code className="font-mono bg-amber-100/80 px-1 py-0.5 rounded">http://localhost:8000</code>.
                  To start the serving backend, execute:
                </p>
                <code className="block mt-1 font-mono text-[11px] bg-amber-100 p-1.5 rounded text-amber-950">
                  .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000
                </code>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Core Highlights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="hover:border-slate-300 transition-colors">
          <CardHeader>
            <div className="h-10 w-10 rounded-lg bg-brand-50 flex items-center justify-center text-brand-600 mb-2">
              <Layers className="h-5 w-5" />
            </div>
            <CardTitle className="text-lg">Fine-Tuned FLAN-T5</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-600 leading-relaxed">
            Encoder-Decoder Transformer fine-tuned via Supervised Fine-Tuning (SFT) specifically for product
            descriptions using strictly partitioned, leak-free Amazon Reviews data.
          </CardContent>
        </Card>

        <Card className="hover:border-slate-300 transition-colors">
          <CardHeader>
            <div className="h-10 w-10 rounded-lg bg-amber-50 flex items-center justify-center text-amber-600 mb-2">
              <Zap className="h-5 w-5" />
            </div>
            <CardTitle className="text-lg">Low-Latency Beam Search</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-600 leading-relaxed">
            Optimized Beam-2 decoding configuration balances generation fluency, attribute adherence,
            and inference latency on local NVIDIA GPU acceleration.
          </CardContent>
        </Card>

        <Card className="hover:border-slate-300 transition-colors">
          <CardHeader>
            <div className="h-10 w-10 rounded-lg bg-emerald-50 flex items-center justify-center text-emerald-600 mb-2">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <CardTitle className="text-lg">Production REST Serving</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-slate-600 leading-relaxed">
            FastAPI serving layer with bounded semaphore concurrency control, robust timeout enforcement,
            and strict Pydantic v2 schema validation.
          </CardContent>
        </Card>
      </div>

      {/* Workflow Section */}
      <div className="rounded-xl border border-slate-200 bg-white p-6 sm:p-8">
        <h2 className="text-xl font-bold text-slate-900">
          How Generation Works
        </h2>
        <p className="mt-1 text-sm text-slate-500">
          The 3-stage process from structured product data to generated copy.
        </p>

        <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-6 relative">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-brand-600 text-xs font-bold text-white">
                1
              </span>
              <h3 className="font-semibold text-slate-900 text-sm">Input Attributes</h3>
            </div>
            <p className="text-xs text-slate-600 pl-9">
              Provide product title, category, brand, bullet-point features, and key-value specifications.
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-brand-600 text-xs font-bold text-white">
                2
              </span>
              <h3 className="font-semibold text-slate-900 text-sm">Configure Decoding</h3>
            </div>
            <p className="text-xs text-slate-600 pl-9">
              Fine-tune beam width, length penalty, token limits, and n-gram repetition constraints.
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-brand-600 text-xs font-bold text-white">
                3
              </span>
              <h3 className="font-semibold text-slate-900 text-sm">Generate & Inspect</h3>
            </div>
            <p className="text-xs text-slate-600 pl-9">
              Review generated copy, latency metrics, token consumption, and formatted prompt template.
            </p>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-slate-100 flex justify-end">
          <Link href="/generate">
            <Button>
              Go to Generator
              <ArrowRight className="h-4 w-4 ml-2" />
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}
