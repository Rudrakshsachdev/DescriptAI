"use client";

import * as React from "react";
import Link from "next/link";
import {
  Sparkles,
  ArrowRight,
  FlaskConical,
  Unplug,
  Box,
  Clock,
  FileText,
  Check,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/Button";
import { getBackendStatus, getModelInfo } from "@/lib/api/services";
import type { BackendConnectionStatus, ModelInfoData } from "@/lib/types/api";

export default function OverviewPage() {
  const [status, setStatus] = React.useState<BackendConnectionStatus>("checking");
  const [modelInfo, setModelInfo] = React.useState<ModelInfoData | null>(null);

  React.useEffect(() => {
    async function load() {
      try {
        const conn = await getBackendStatus();
        setStatus(conn.status);
        if (conn.status === "online_ready") {
          const info = await getModelInfo();
          if (info.success) setModelInfo(info.data);
        }
      } catch {
        setStatus("offline");
      }
    }
    load();
  }, []);

  const isReady = status === "online_ready";

  return (
    <div className="space-y-8 pb-12 max-w-6xl mx-auto">
      {/* Top Header Section */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div>
          <span className="text-[11px] font-bold text-[#0066e6] tracking-[0.14em] uppercase block">
            YOUR WORKSPACE
          </span>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-[#0f172a] tracking-tight mt-1">
            Product copy, made clearer.
          </h1>
          <p className="text-sm text-[#64748b] mt-1.5 max-w-2xl font-normal leading-relaxed">
            Shape the details. Find the right words. Build descriptions that bring your products to life.
          </p>
        </div>

        <Link href="/generate" className="self-start sm:self-auto shrink-0">
          <Button
            size="md"
            className="bg-[#0066e6] hover:bg-[#0052cc] text-white font-medium px-4 py-2.5 text-sm rounded-lg flex items-center gap-2 shadow-xs transition-all"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Generate a description</span>
            <ArrowRight className="w-3.5 h-3.5 ml-1" />
          </Button>
        </Link>
      </div>

      {/* 4 Status Columns in a Unified Row with Dividers */}
      <div className="bg-white border border-slate-200/80 rounded-xl shadow-2xs divide-y sm:divide-y-0 sm:divide-x divide-slate-100 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        {/* Col 1: Model readiness */}
        <div className="p-4 sm:p-5 flex items-center gap-3.5">
          <div className="w-9 h-9 rounded-lg bg-[#eef4ff] text-[#0066e6] flex items-center justify-center shrink-0">
            <FlaskConical className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <span className="text-xs text-[#64748b] block font-normal">Model readiness</span>
            <div className="mt-1 flex items-center">
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium border ${
                  isReady
                    ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                    : "bg-[#fffdf5] text-amber-800 border-amber-200"
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    isReady ? "bg-emerald-500" : "bg-amber-500"
                  }`}
                />
                {isReady ? "Ready" : "Awaiting setup"}
              </span>
            </div>
          </div>
        </div>

        {/* Col 2: Backend connection */}
        <div className="p-4 sm:p-5 flex items-center gap-3.5">
          <div className="w-9 h-9 rounded-lg bg-slate-50 text-slate-400 border border-slate-100 flex items-center justify-center shrink-0">
            <Unplug className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <span className="text-xs text-[#64748b] block font-normal">Backend connection</span>
            <div className="mt-1 flex items-center">
              <span
                className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-medium border ${
                  isReady
                    ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                    : "bg-slate-50 text-slate-600 border-slate-200"
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    isReady ? "bg-emerald-500" : "bg-slate-400"
                  }`}
                />
                {isReady ? "Connected" : "Not connected"}
              </span>
            </div>
          </div>
        </div>

        {/* Col 3: Model family */}
        <div className="p-4 sm:p-5 flex items-center gap-3.5">
          <div className="w-9 h-9 rounded-lg bg-slate-50 text-slate-400 border border-slate-100 flex items-center justify-center shrink-0">
            <Box className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <span className="text-xs text-[#64748b] block font-normal">Model family</span>
            <span className="text-sm font-semibold text-[#0f172a] block mt-0.5">
              FLAN-T5
            </span>
          </div>
        </div>

        {/* Col 4: Generation mode */}
        <div className="p-4 sm:p-5 flex items-center gap-3.5">
          <div className="w-9 h-9 rounded-lg bg-slate-50 text-slate-400 border border-slate-100 flex items-center justify-center shrink-0">
            <Clock className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <span className="text-xs text-[#64748b] block font-normal">Generation mode</span>
            <span className="text-sm font-semibold text-[#0f172a] block mt-0.5">
              {isReady ? "Beam search" : "Local demonstration"}
            </span>
          </div>
        </div>
      </div>

      {/* Large 2-Column Studio Banner Card */}
      <div className="bg-white border border-slate-200/80 rounded-2xl shadow-2xs overflow-hidden flex flex-col md:flex-row">
        {/* Left Main Studio */}
        <div className="flex-1 p-8 sm:p-10 flex flex-col justify-between space-y-6">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 text-[11px] font-bold text-[#0066e6] tracking-[0.14em] uppercase">
              <Sparkles className="w-3.5 h-3.5" />
              <span>THE PRODUCT DESCRIPTION STUDIO</span>
            </div>

            <h2 className="font-serif text-3xl sm:text-4xl text-[#0f172a] font-normal leading-tight">
              Good products deserve{" "}
              <span className="text-[#0066e6]">better words.</span>
            </h2>

            <p className="text-sm text-[#64748b] max-w-xl leading-relaxed pt-1">
              Give your product a little context. Build a polished starting point from its features, specifications, and the people it&apos;s made for.
            </p>
          </div>

          <div className="flex items-center gap-4 pt-2">
            <Link href="/generate">
              <Button
                size="md"
                className="bg-[#0066e6] hover:bg-[#0052cc] text-white font-medium px-4 py-2.5 text-sm rounded-lg inline-flex items-center gap-2 shadow-xs transition-all"
              >
                <span>Open the generator</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Button>
            </Link>

            <Link
              href="/model"
              className="text-xs text-[#64748b] hover:text-[#0f172a] font-medium inline-flex items-center gap-1 transition-colors"
            >
              <span>Check model status</span>
              <ExternalLink className="w-3 h-3" />
            </Link>
          </div>
        </div>

        {/* Right Thoughtful Start Panel */}
        <div className="w-full md:w-[320px] lg:w-[340px] bg-[#fbfcfd] border-t md:border-t-0 md:border-l border-slate-200/80 p-8 flex flex-col justify-between">
          <div className="space-y-2">
            <div className="w-8 h-8 rounded-lg bg-[#eef4ff] text-[#0066e6] flex items-center justify-center">
              <FileText className="w-4 h-4 fill-[#0066e6]/20 stroke-[2]" />
            </div>

            <span className="text-[10px] font-bold text-[#94a3b8] tracking-[0.16em] uppercase block pt-2">
              A THOUGHTFUL START
            </span>

            <h3 className="font-serif text-xl sm:text-2xl font-normal text-[#0f172a] mt-1">
              From details to description.
            </h3>

            <p className="text-xs text-[#64748b] leading-relaxed pt-1">
              A considered first draft, shaped around the way you want to sound.
            </p>
          </div>

          <div className="pt-8">
            <div className="inline-flex items-center gap-2 text-xs text-[#334155] bg-white border border-slate-200/80 px-3.5 py-1.5 rounded-full shadow-2xs">
              <Check className="w-3.5 h-3.5 text-emerald-600 stroke-[2.5]" />
              <span>Your product, your tone, your call.</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section: Start with a product & Model status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 pt-2">
        {/* Left: Start with a product */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-[#0f172a]">Start with a product</h3>
            <span className="text-xs text-[#94a3b8]">A sample to explore the workspace</span>
          </div>

          <div className="bg-white border border-slate-200/80 rounded-xl p-4 sm:p-5 flex items-center justify-between gap-4 shadow-2xs">
            <div className="flex items-start gap-3.5 min-w-0">
              <div className="w-10 h-10 rounded-xl bg-[#eef4ff] text-[#0066e6] flex items-center justify-center shrink-0 mt-0.5">
                <Box className="w-5 h-5 stroke-[1.8]" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-bold text-[#0f172a] truncate">
                    LumaArc Desk Lamp
                  </span>
                  <span className="text-[10px] font-medium bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full">
                    Sample
                  </span>
                </div>
                <p className="text-xs text-[#94a3b8] mt-0.5">
                  Northline Studio &nbsp;&nbsp; Home &amp; lighting
                </p>
                <p className="text-xs text-[#64748b] mt-1 line-clamp-1">
                  Adjustable warm-to-cool light, touch dimmer, and USB-C charging.
                </p>
              </div>
            </div>

            <Link href="/generate" className="shrink-0">
              <Button
                variant="outline"
                size="sm"
                className="text-xs border-slate-200 hover:bg-slate-50 text-slate-700 font-medium px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-colors"
              >
                <span>Use sample</span>
                <ArrowRight className="w-3 h-3" />
              </Button>
            </Link>
          </div>
        </div>

        {/* Right: Model status */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-[#0f172a]">Model status</h3>
            <span className="text-xs text-[#94a3b8]">What&apos;s configured right now</span>
          </div>

          <div className="bg-white border border-slate-200/80 rounded-xl p-4 sm:p-5 shadow-2xs space-y-3 text-xs">
            <div className="flex items-center justify-between py-1 border-b border-slate-100">
              <span className="text-[#64748b] font-medium">Architecture</span>
              <span className="text-[#0f172a] font-semibold">
                {modelInfo ? modelInfo.architecture : "FLAN-T5 family"}
              </span>
            </div>

            <div className="flex items-center justify-between py-1 border-b border-slate-100">
              <span className="text-[#64748b] font-medium">Model variant</span>
              <span className="text-[#0f172a] font-semibold">
                {modelInfo ? `${modelInfo.model_id} (77M)` : "Not specified"}
              </span>
            </div>

            <div className="flex items-center justify-between py-1">
              <span className="text-[#64748b] font-medium">Execution device</span>
              <span className="text-[#0f172a] font-semibold">
                {modelInfo ? `${modelInfo.device} (${modelInfo.precision})` : "Local CUDA / CPU"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
