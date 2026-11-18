"use client";

import * as React from "react";
import { getBackendStatus } from "@/lib/api/services";
import { BackendConnectionStatus, ReadinessResponse } from "@/lib/types/api";
import { cn } from "@/lib/utils";
import { RefreshCw, Server, AlertCircle } from "lucide-react";

export function HealthIndicator({ className }: { className?: string }) {
  const [status, setStatus] = React.useState<BackendConnectionStatus>("checking");
  const [readiness, setReadiness] = React.useState<ReadinessResponse | null>(null);
  const [isRefreshing, setIsRefreshing] = React.useState(false);
  const [showTooltip, setShowTooltip] = React.useState(false);

  const fetchStatus = React.useCallback(async () => {
    setIsRefreshing(true);
    try {
      const res = await getBackendStatus();
      setStatus(res.status);
      setReadiness(res.readiness || null);
    } catch {
      setStatus("offline");
      setReadiness(null);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  React.useEffect(() => {
    fetchStatus();
    // Non-aggressive 30-second background probe
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const config = {
    checking: {
      dot: "bg-slate-400 animate-pulse",
      label: "Connecting...",
      badge: "bg-slate-50 text-slate-600 border-slate-200",
      description: "Probing backend service liveness...",
    },
    online_ready: {
      dot: "bg-emerald-500",
      label: "Model Ready",
      badge: "bg-emerald-50 text-emerald-800 border-emerald-200 hover:bg-emerald-100",
      description: readiness ? `FLAN-T5 Seq2Seq active on ${readiness.device}` : "AI Generator operational",
    },
    online_not_ready: {
      dot: "bg-amber-500 animate-pulse",
      label: "Model Unmounted",
      badge: "bg-amber-50 text-amber-800 border-amber-200 hover:bg-amber-100",
      description: readiness?.startup_error || "Backend online, but model checkpoint is unmounted.",
    },
    offline: {
      dot: "bg-rose-500",
      label: "Backend Offline",
      badge: "bg-rose-50 text-rose-800 border-rose-200 hover:bg-rose-100",
      description: "Cannot connect to FastAPI backend at port 8000.",
    },
  }[status];

  return (
    <div className={cn("relative inline-flex items-center", className)}>
      <button
        type="button"
        onClick={() => fetchStatus()}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        className={cn(
          "inline-flex items-center gap-2 px-2.5 py-1 text-xs font-medium rounded-full border transition-all cursor-pointer focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-slate-400",
          config.badge
        )}
        title="Click to re-probe backend connection"
        aria-label={`Backend Status: ${config.label}`}
      >
        <span className={cn("w-2 h-2 rounded-full flex-shrink-0", config.dot)} />
        <span className="truncate max-w-[120px]">{config.label}</span>
        <RefreshCw
          className={cn("w-3 h-3 text-slate-400 transition-transform", isRefreshing && "animate-spin")}
        />
      </button>

      {/* Hover tooltip for operational details */}
      {showTooltip && (
        <div className="absolute right-0 top-full mt-2 w-64 p-3 bg-white border border-slate-200 rounded-lg shadow-lg text-xs z-50 text-slate-700 animate-in fade-in zoom-in-95 duration-100">
          <div className="flex items-center gap-1.5 font-semibold text-slate-900 mb-1">
            <Server className="w-3.5 h-3.5 text-brand-600" />
            <span>Service Diagnostics</span>
          </div>
          <p className="text-slate-600 mb-2 leading-relaxed">{config.description}</p>
          {readiness && (
            <div className="border-t border-slate-100 pt-2 space-y-1 font-mono text-[11px] text-slate-500">
              <div className="flex justify-between">
                <span>Device:</span>
                <span className="font-semibold text-slate-700">{readiness.device}</span>
              </div>
              <div className="flex justify-between">
                <span>Precision:</span>
                <span>{readiness.half_precision ? "BFloat16 / Half" : "Float32"}</span>
              </div>
            </div>
          )}
          {status === "offline" && (
            <div className="flex items-start gap-1 text-rose-600 pt-1 text-[11px]">
              <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
              <span>Ensure backend is running: `uvicorn backend.app.main:app`</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
