"use client";

import * as React from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/Button";

export default function ErrorBoundary({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  React.useEffect(() => {
    console.error("Application error:", error);
  }, [error]);

  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center text-center px-4">
      <div className="w-14 h-14 rounded-2xl bg-rose-50 flex items-center justify-center text-rose-600 mb-4 border border-rose-200">
        <AlertTriangle className="w-7 h-7" />
      </div>
      <h2 className="text-2xl font-bold tracking-tight text-slate-900">
        Something went wrong
      </h2>
      <p className="mt-2 text-sm text-slate-500 max-w-md">
        {error.message || "An unexpected error occurred while rendering the page."}
      </p>
      <div className="mt-6 flex gap-3">
        <Button onClick={() => reset()} size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Try Again
        </Button>
      </div>
    </div>
  );
}
