"use client";

import * as React from "react";
import { getBackendStatus } from "@/lib/api/services";
import { BackendConnectionStatus, ReadinessResponse } from "@/lib/types/api";
import { cn } from "@/lib/utils";

export function HealthIndicator({ className }: { className?: string }) {
  const [status, setStatus] = React.useState<BackendConnectionStatus>("checking");
  const [readiness, setReadiness] = React.useState<ReadinessResponse | null>(null);

  const fetchStatus = React.useCallback(async () => {
    try {
      const res = await getBackendStatus();
      setStatus(res.status);
      setReadiness(res.readiness || null);
    } catch {
      setStatus("offline");
      setReadiness(null);
    }
  }, []);

  React.useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const config = {
    checking: {
      dot: "bg-slate-400 animate-pulse",
      label: "Connecting...",
      badge: "border-slate-200 bg-slate-50 text-slate-600",
    },
    online_ready: {
      dot: "bg-emerald-500",
      label: "Model ready",
      badge: "border-emerald-200/80 bg-emerald-50/70 text-emerald-800",
    },
    online_not_ready: {
      dot: "bg-amber-500",
      label: "Demo mode",
      badge: "border-amber-200/80 bg-[#fffdf5] text-amber-900",
    },
    offline: {
      dot: "bg-amber-500",
      label: "Demo mode",
      badge: "border-amber-200/80 bg-[#fffdf5] text-amber-900",
    },
  };

  const current = config[status];

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-full border font-medium select-none transition-all",
        current.badge,
        className
      )}
      title={
        status === "online_ready"
          ? `Serving FLAN-T5 on ${readiness?.device || "local GPU"}`
          : "Backend offline — operating in local demo preview mode"
      }
    >
      <span className={cn("w-1.5 h-1.5 rounded-full shrink-0", current.dot)} />
      <span>{current.label}</span>
    </div>
  );
}
