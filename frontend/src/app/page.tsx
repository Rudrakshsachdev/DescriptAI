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
  Lock,
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
      {/* Hero Section — Clean, Professional SaaS Design */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-8 sm:p-10 shadow-xs">
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 rounded-full bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-700 border border-blue-100">
            <Sparkles className="h-3.5 w-3.5 text-blue-600" />
            <span>Local Seq2Seq Transformer • FLAN-T5 Small</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900 leading-tight">
            AI-Powered Product Description Generator
          </h1>

          <p className="text-slate-600 text-base sm:text-lg leading-relaxed">
            Generate high-converting, attribute-accurate e-commerce copy from structured product data.
            Powered by a locally hosted, fine-tuned FLAN-T5 encoder-decoder model with low-latency Beam-2 decoding.
          </p>

          <div className="pt-3 flex flex-wrap items-center gap-3">
            <Link href="/generate">
              <Button size="lg" className="shadow-xs">
                <Sparkles className="h-4 w-4 mr-2" />
                Launch Generator
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
            </Link>

            <Link href="/model">
              <Button variant="outline" size="lg">
                <Cpu className="h-4 w-4 mr-2 text-slate-500" />
                Model Diagnostics
              </Button>
            </Link>
          </div>
        </div>
      </div>

      {/* Real-Time Operational Telemetry Bar */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <Server className="h-4 w-4 text-slate-500" />
            <CardTitle className="text-sm font-semibold text-slate-900">
              Operational Inference Telemetry
            </CardTitle>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchStatus}
            disabled={isRefreshing}
            className="text-xs text-slate-600 hover:text-slate-900 h-8"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 mr-1.5 ${isRefreshing ? "animate-spin text-blue-600" : ""}`}
            />
            {isRefreshing ? "Checking..." : "Refresh"}
          </Button>
        </CardHeader>
        <CardContent className="pt-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="rounded-lg bg-slate-50/80 p-3.5 border border-slate-200/60">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                API Connection
              </span>
              <div className="mt-1.5 flex items-center gap-2">
                {status === "checking" && (
                  <Badge variant="neutral">Checking...</Badge>
                )}
                {status === "online_ready" && (
                  <Badge variant="success" className="gap-1.5 font-medium">
                    <CheckCircle2 className="h-3 w-3 text-emerald-600" />
                    Online & Ready
                  </Badge>
                )}
                {status === "online_not_ready" && (
                  <Badge variant="warning" className="gap-1.5 font-medium">
                    <AlertTriangle className="h-3 w-3 text-amber-600" />
                    Model Loading
                  </Badge>
                )}
                {status === "offline" && (
                  <Badge variant="error" className="gap-1.5 font-medium">
                    <AlertTriangle className="h-3 w-3 text-rose-600" />
                    Offline
                  </Badge>
                )}
              </div>
            </div>

            <div className="rounded-lg bg-slate-50/80 p-3.5 border border-slate-200/60">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                Serving Model
              </span>
              <p className="mt-1.5 text-sm font-semibold text-slate-900 font-mono truncate">
                {modelInfo ? modelInfo.model_id : "google/flan-t5-small"}
              </p>
            </div>

            <div className="rounded-lg bg-slate-50/80 p-3.5 border border-slate-200/60">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                Hardware Device
              </span>
              <p className="mt-1.5 text-sm font-semibold text-slate-900 font-mono">
                {modelInfo ? `${modelInfo.device} (${modelInfo.precision})` : "cuda:0 / cpu"}
              </p>
            </div>

            <div className="rounded-lg bg-slate-50/80 p-3.5 border border-slate-200/60">
              <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
                Parameter Count
              </span>
              <p className="mt-1.5 text-sm font-semibold text-slate-900 font-mono">
                {modelInfo ? formatNumber(modelInfo.total_parameters) : "76,961,152"}
              </p>
            </div>
          </div>

          {status === "offline" && (
            <div className="mt-4 rounded-xl bg-amber-50/80 p-4 border border-amber-200 text-xs text-amber-900 flex items-start gap-3">
              <AlertTriangle className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
              <div className="space-y-1">
                <p className="font-semibold">FastAPI Serving Server Offline</p>
                <p className="text-amber-800">
                  The backend service is currently unreachable at <code className="font-mono bg-amber-100/90 px-1 py-0.5 rounded text-amber-950">http://localhost:8000</code>.
                  Launch the local serving server using PowerShell:
                </p>
                <code className="block font-mono text-[11px] bg-amber-100/90 p-2 rounded text-amber-950 mt-1 select-all">
                  .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000
                </code>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 3-Step Interactive Workflow */}
      <div className="bg-white rounded-2xl border border-slate-200/90 p-8 shadow-xs">
        <div className="max-w-xl">
          <h2 className="text-lg font-bold text-slate-900 tracking-tight">
            How Descript AI Generates Copy
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Three structured stages translate raw catalog attributes into polished marketing prose.
          </p>
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="relative space-y-2">
            <div className="flex items-center gap-3">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-50 text-blue-700 text-xs font-bold border border-blue-200">
                1
              </span>
              <h3 className="font-semibold text-slate-900 text-sm">Input Attributes</h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed pl-10">
              Provide product title, category, brand, key bullet-point features, and technical specifications.
            </p>
          </div>

          <div className="relative space-y-2">
            <div className="flex items-center gap-3">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-50 text-blue-700 text-xs font-bold border border-blue-200">
                2
              </span>
              <h3 className="font-semibold text-slate-900 text-sm">Configure Decoding</h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed pl-10">
              Adjust beam search width, length penalty, token bounds, and repetition constraints.
            </p>
          </div>

          <div className="relative space-y-2">
            <div className="flex items-center gap-3">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-blue-50 text-blue-700 text-xs font-bold border border-blue-200">
                3
              </span>
              <h3 className="font-semibold text-slate-900 text-sm">Generate & Inspect</h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed pl-10">
              Receive formatted copy with word and token telemetry, latency timings, and raw prompt inspection.
            </p>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-slate-100 flex justify-end">
          <Link href="/generate">
            <Button size="sm">
              Open Generator Workspace
              <ArrowRight className="h-3.5 w-3.5 ml-1.5" />
            </Button>
          </Link>
        </div>
      </div>

      {/* Technical Foundations Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="hover:border-slate-300 transition-colors">
          <CardHeader className="pb-2">
            <div className="h-9 w-9 rounded-lg bg-blue-50 flex items-center justify-center text-blue-600 mb-2 border border-blue-100">
              <Layers className="h-4 w-4" />
            </div>
            <CardTitle className="text-base font-semibold">Fine-Tuned FLAN-T5</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-600 leading-relaxed">
            Encoder-Decoder Transformer checkpoint trained on the Amazon Reviews 2023 All_Beauty dataset
            using product-isolated partitions with zero data leakage.
          </CardContent>
        </Card>

        <Card className="hover:border-slate-300 transition-colors">
          <CardHeader className="pb-2">
            <div className="h-9 w-9 rounded-lg bg-amber-50 flex items-center justify-center text-amber-600 mb-2 border border-amber-100">
              <Zap className="h-4 w-4" />
            </div>
            <CardTitle className="text-base font-semibold">Low-Latency Beam Search</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-600 leading-relaxed">
            Production Beam-2 decoding configuration optimized on local GPU hardware, delivering coherent,
            hallucination-minimized product descriptions in sub-second inference passes.
          </CardContent>
        </Card>

        <Card className="hover:border-slate-300 transition-colors">
          <CardHeader className="pb-2">
            <div className="h-9 w-9 rounded-lg bg-emerald-50 flex items-center justify-center text-emerald-600 mb-2 border border-emerald-100">
              <Lock className="h-4 w-4" />
            </div>
            <CardTitle className="text-base font-semibold">100% Local & Private</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-slate-600 leading-relaxed">
            All inference runs locally via PyTorch and FastAPI. No customer data or catalog attributes are
            ever transmitted to third-party generative AI cloud APIs.
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
