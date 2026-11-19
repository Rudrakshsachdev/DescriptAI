"use client";

import * as React from "react";
import Link from "next/link";
import { OutputPanel } from "@/components/generator/OutputPanel";
import { generateDescription, getBackendStatus } from "@/lib/api/services";
import { ApiError } from "@/lib/api/client";
import {
  GenerationData,
  ProductDescriptionRequest,
  BackendConnectionStatus,
} from "@/lib/types/api";
import {
  Sparkles,
  RotateCcw,
  FileText,
  Wand2,
  AlertTriangle,
  ArrowRight,
  Info,
} from "lucide-react";
import { Button } from "@/components/ui/Button";

const SAMPLE_DATA = {
  product_name: "LumaArc Desk Lamp",
  category: "Home & lighting",
  brand: "Northline Studio",
  features: `Adjustable warm-to-cool light
Touch-sensitive dimmer
USB-C charging port`,
  specifications: `Materials: brushed aluminum and recycled ABS
Dimensions: 42 × 16 cm
Power: 12 W LED, USB-C`,
  audience: "Design-conscious professionals setting up a home workspace",
};

export function GeneratorForm() {
  const [productName, setProductName] = React.useState(SAMPLE_DATA.product_name);
  const [category, setCategory] = React.useState(SAMPLE_DATA.category);
  const [brand, setBrand] = React.useState(SAMPLE_DATA.brand);
  const [featuresText, setFeaturesText] = React.useState(SAMPLE_DATA.features);
  const [specsText, setSpecsText] = React.useState(SAMPLE_DATA.specifications);
  const [audience, setAudience] = React.useState(SAMPLE_DATA.audience);

  const [tone, setTone] = React.useState<"Refined" | "Friendly" | "Confident" | "Playful">("Refined");
  const [length, setLength] = React.useState<"Short" | "Standard" | "Detailed">("Standard");

  const [formError, setFormError] = React.useState<string | null>(null);
  const [apiError, setApiError] = React.useState<string | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [outputData, setOutputData] = React.useState<GenerationData | null>(null);
  const [backendStatus, setBackendStatus] = React.useState<BackendConnectionStatus>("checking");

  React.useEffect(() => {
    async function check() {
      try {
        const res = await getBackendStatus();
        setBackendStatus(res.status);
      } catch {
        setBackendStatus("offline");
      }
    }
    check();
  }, []);

  const handleLoadSample = () => {
    setProductName(SAMPLE_DATA.product_name);
    setCategory(SAMPLE_DATA.category);
    setBrand(SAMPLE_DATA.brand);
    setFeaturesText(SAMPLE_DATA.features);
    setSpecsText(SAMPLE_DATA.specifications);
    setAudience(SAMPLE_DATA.audience);
    setTone("Refined");
    setLength("Standard");
    setFormError(null);
    setApiError(null);
  };

  const handleClear = () => {
    setProductName("");
    setCategory("Home & lighting");
    setBrand("");
    setFeaturesText("");
    setSpecsText("");
    setAudience("");
    setFormError(null);
    setApiError(null);
    setOutputData(null);
  };

  const performGeneration = async () => {
    setFormError(null);
    setApiError(null);

    const cleanName = productName.trim();
    if (!cleanName) {
      setFormError("Product name is required.");
      return;
    }
    if (cleanName.length < 2) {
      setFormError("Product name must be at least 2 characters.");
      return;
    }

    // Parse features from lines
    const featuresList = featuresText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);

    // Parse specifications from lines
    const specsRecord: Record<string, string> = {};
    specsText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .forEach((line) => {
        const colonIdx = line.indexOf(":");
        if (colonIdx > 0) {
          const k = line.substring(0, colonIdx).trim();
          const v = line.substring(colonIdx + 1).trim();
          if (k && v) specsRecord[k] = v;
        } else {
          specsRecord[line] = "Included";
        }
      });

    // Map Length to token limits
    const tokenLimits = {
      Short: { max: 64, min: 10 },
      Standard: { max: 128, min: 20 },
      Detailed: { max: 200, min: 40 },
    }[length];

    const payload: ProductDescriptionRequest = {
      product_name: cleanName,
      category: category.trim() || undefined,
      brand: brand.trim() || undefined,
      features: featuresList.length > 0 ? featuresList : undefined,
      specifications: Object.keys(specsRecord).length > 0 ? specsRecord : undefined,
      audience: audience.trim() ? `${audience.trim()} (${tone} tone)` : `${tone} tone`,
      generation_config: {
        num_beams: 2,
        max_new_tokens: tokenLimits.max,
        min_new_tokens: tokenLimits.min,
        no_repeat_ngram_size: 3,
        length_penalty: length === "Detailed" ? 1.2 : length === "Short" ? 0.8 : 1.0,
      },
    };

    setIsLoading(true);

    try {
      const response = await generateDescription(payload);
      if (response?.data) {
        setOutputData(response.data);
      }
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        if (err.statusCode === 503) {
          setApiError(
            err.errorCode === "service_overloaded"
              ? "Service is currently experiencing high load. Please retry in a few moments."
              : "AI model is not ready. The checkpoint may still be loading."
          );
        } else if (err.statusCode === 504) {
          setApiError("Inference request timed out on the backend. The GPU worker may be busy.");
        } else if (err.statusCode === 422) {
          setApiError(`Validation Error: ${err.message}`);
        } else {
          setApiError(err.message);
        }
      } else {
        // Fallback demo generation for offline preview if backend is unreachable
        setOutputData({
          generated_description: `${cleanName} by ${brand || "Northline Studio"} brings effortless illumination and intentional design to your space. Featuring ${featuresList.slice(0, 2).join(" and ") || "modern styling"}, it combines durable construction with refined aesthetic appeal for ${audience || "design-conscious individuals"}.`,
          model: "google/flan-t5-small (Local Template)",
          generation_strategy: "beam_search",
          latency_ms: 180,
          output_tokens: 38,
          input_prompt: `Product: ${cleanName}\nCategory: ${category}\nFeatures: ${featuresList.join(", ")}`,
        });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await performGeneration();
  };

  const isReady = backendStatus === "online_ready";

  return (
    <div className="space-y-6 pb-12 max-w-6xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <div>
          <span className="text-[11px] font-bold text-[#0066e6] tracking-wider uppercase block">
            THE STUDIO
          </span>
          <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight mt-1">
            Description generator
          </h1>
          <p className="text-sm text-slate-500 mt-1 max-w-2xl">
            Start with the details that make your product yours. Refine the direction, then shape a first draft.
          </p>
        </div>

        <Link
          href="/model"
          className="text-xs text-slate-500 hover:text-slate-900 font-medium inline-flex items-center gap-1 transition-colors self-start sm:self-auto shrink-0 pt-2"
        >
          <span>Model status</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </div>

      {/* Model / Demo Status Sub-Bar */}
      <div className="flex items-center justify-between text-xs py-1 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium border border-amber-200/80 bg-[#fffdf5] text-amber-900">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                isReady ? "bg-emerald-500" : "bg-amber-500"
              }`}
            />
            {isReady ? "Model ready" : "Local demo"}
          </span>
          <span className="text-slate-400">
            {isReady
              ? "FastAPI inference serving active on local GPU"
              : "No model service connected"}
          </span>
        </div>

        <button
          type="button"
          onClick={handleClear}
          className="text-xs text-slate-500 hover:text-slate-800 flex items-center gap-1.5 transition-colors"
          title="Clear all fields"
        >
          <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
          <span>Reset sample</span>
        </button>
      </div>

      {/* Main 2-Column Generator Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start pt-2">
        {/* Left Column: Product Details Form */}
        <div className="lg:col-span-7 space-y-6">
          <form
            onSubmit={handleSubmit}
            className="bg-white border border-slate-200/90 rounded-2xl p-6 sm:p-7 shadow-xs space-y-6"
          >
            {/* Section Header */}
            <div className="flex items-center justify-between pb-4 border-b border-slate-100">
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-[#eef4ff] text-[#0066e6] flex items-center justify-center shrink-0">
                  <FileText className="w-4 h-4 fill-[#0066e6]/20 stroke-[2]" />
                </div>
                <div>
                  <h2 className="text-sm font-bold text-slate-900">
                    Product details
                  </h2>
                  <p className="text-[11px] text-slate-400">
                    Tell us what you&apos;re describing.
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={handleLoadSample}
                disabled={isLoading}
                className="text-xs font-medium text-slate-600 bg-slate-100 hover:bg-slate-200/80 px-2.5 py-1 rounded-md transition-colors"
              >
                Sample product
              </button>
            </div>

            {/* Error Banner */}
            {apiError && (
              <div className="p-3.5 bg-rose-50 border border-rose-200 rounded-xl text-xs text-rose-800 flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <span className="font-semibold block mb-0.5">Generation Failed</span>
                  <p className="leading-relaxed">{apiError}</p>
                </div>
              </div>
            )}

            {/* Product Name & Category Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="block text-xs font-semibold text-slate-800">
                  Product name <span className="text-rose-500">*</span>
                </label>
                <input
                  type="text"
                  disabled={isLoading}
                  value={productName}
                  onChange={(e) => {
                    setProductName(e.target.value);
                    if (formError) setFormError(null);
                  }}
                  placeholder="e.g., LumaArc Desk Lamp"
                  className="w-full px-3 py-2 text-xs sm:text-sm text-slate-900 bg-white border border-slate-200 rounded-lg shadow-2xs placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-[#0066e6]/20 focus:border-[#0066e6] transition-all"
                />
                {formError && (
                  <p className="text-[11px] text-rose-600 font-medium">{formError}</p>
                )}
              </div>

              <div className="space-y-1.5">
                <label className="block text-xs font-semibold text-slate-800">
                  Category
                </label>
                <select
                  disabled={isLoading}
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full px-3 py-2 text-xs sm:text-sm text-slate-900 bg-white border border-slate-200 rounded-lg shadow-2xs focus:outline-none focus:ring-2 focus:ring-[#0066e6]/20 focus:border-[#0066e6] transition-all"
                >
                  <option value="Home & lighting">Home & lighting</option>
                  <option value="Beauty & Personal Care">Beauty & Personal Care</option>
                  <option value="Electronics & Tech">Electronics & Tech</option>
                  <option value="Apparel & Fashion">Apparel & Fashion</option>
                  <option value="Health & Wellness">Health & Wellness</option>
                  <option value="Kitchen & Dining">Kitchen & Dining</option>
                  <option value="Office Products">Office Products</option>
                  <option value="Sports & Outdoors">Sports & Outdoors</option>
                </select>
              </div>
            </div>

            {/* Brand */}
            <div className="space-y-1.5">
              <label className="block text-xs font-semibold text-slate-800">
                Brand
              </label>
              <input
                type="text"
                disabled={isLoading}
                value={brand}
                onChange={(e) => setBrand(e.target.value)}
                placeholder="e.g., Northline Studio"
                className="w-full px-3 py-2 text-xs sm:text-sm text-slate-900 bg-white border border-slate-200 rounded-lg shadow-2xs placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-[#0066e6]/20 focus:border-[#0066e6] transition-all"
              />
            </div>

            {/* Key Features Textarea */}
            <div className="space-y-1.5">
              <label className="block text-xs font-semibold text-slate-800">
                Key features
              </label>
              <textarea
                rows={4}
                disabled={isLoading}
                value={featuresText}
                onChange={(e) => setFeaturesText(e.target.value)}
                placeholder={`Adjustable warm-to-cool light\nTouch-sensitive dimmer\nUSB-C charging port`}
                className="w-full px-3 py-2 text-xs sm:text-sm text-slate-900 bg-white border border-slate-200 rounded-lg shadow-2xs placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-[#0066e6]/20 focus:border-[#0066e6] transition-all resize-y font-normal"
              />
              <p className="text-[11px] text-slate-400">
                One feature per line works best.
              </p>
            </div>

            {/* Specifications Textarea */}
            <div className="space-y-1.5">
              <label className="block text-xs font-semibold text-slate-800">
                Specifications
              </label>
              <textarea
                rows={4}
                disabled={isLoading}
                value={specsText}
                onChange={(e) => setSpecsText(e.target.value)}
                placeholder={`Materials: brushed aluminum and recycled ABS\nDimensions: 42 × 16 cm\nPower: 12 W LED, USB-C`}
                className="w-full px-3 py-2 text-xs sm:text-sm text-slate-900 bg-white border border-slate-200 rounded-lg shadow-2xs placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-[#0066e6]/20 focus:border-[#0066e6] transition-all resize-y font-normal"
              />
              <p className="text-[11px] text-slate-400">
                Include only the technical details you want mentioned.
              </p>
            </div>

            {/* Target Audience */}
            <div className="space-y-1.5">
              <label className="block text-xs font-semibold text-slate-800">
                Target audience
              </label>
              <input
                type="text"
                disabled={isLoading}
                value={audience}
                onChange={(e) => setAudience(e.target.value)}
                placeholder="e.g., Design-conscious professionals setting up a home workspace"
                className="w-full px-3 py-2 text-xs sm:text-sm text-slate-900 bg-white border border-slate-200 rounded-lg shadow-2xs placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-[#0066e6]/20 focus:border-[#0066e6] transition-all"
              />
            </div>

            {/* Writing Direction Section */}
            <div className="pt-2 border-t border-slate-100 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold text-slate-900">
                  Writing direction
                </h3>
                <span className="text-[11px] text-slate-400">
                  Shape the first draft
                </span>
              </div>

              {/* Tone Selection */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-600 block">
                  Tone
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {(["Refined", "Friendly", "Confident", "Playful"] as const).map((t) => (
                    <button
                      key={t}
                      type="button"
                      disabled={isLoading}
                      onClick={() => setTone(t)}
                      className={`px-3 py-1.5 text-xs rounded-lg border font-medium transition-all text-center ${
                        tone === t
                          ? "border-[#0066e6] bg-[#eef4ff] text-[#0066e6] font-semibold shadow-2xs"
                          : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                      }`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>

              {/* Length Selection */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-semibold text-slate-600 block">
                  Length
                </span>
                <div className="grid grid-cols-3 gap-2">
                  {(["Short", "Standard", "Detailed"] as const).map((l) => (
                    <button
                      key={l}
                      type="button"
                      disabled={isLoading}
                      onClick={() => setLength(l)}
                      className={`px-3 py-1.5 text-xs rounded-lg border font-medium transition-all text-center ${
                        length === l
                          ? "border-[#0066e6] bg-[#eef4ff] text-[#0066e6] font-semibold shadow-2xs"
                          : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                      }`}
                    >
                      {l}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Generate Action Button */}
            <div className="pt-3">
              <Button
                type="submit"
                size="lg"
                className="w-full bg-[#0066e6] hover:bg-[#0052cc] text-white py-3 rounded-xl shadow-xs font-semibold flex items-center justify-center gap-2"
                isLoading={isLoading}
                disabled={isLoading}
              >
                <Sparkles className="w-4 h-4 text-white" />
                <span>
                  {isLoading ? "Generating with FLAN-T5..." : "Generate description"}
                </span>
                <Wand2 className="w-3.5 h-3.5 text-white/80" />
              </Button>

              <div className="flex items-start gap-1.5 text-[11px] text-slate-400 mt-3 leading-relaxed">
                <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
                <span>
                  This interactive preview creates example copy locally. It does not send your information to an external AI model or verify product claims.
                </span>
              </div>
            </div>
          </form>
        </div>

        {/* Right Column: Output Panel */}
        <div className="lg:col-span-5 sticky top-20">
          <OutputPanel
            data={outputData}
            isLoading={isLoading}
            onLoadExample={handleLoadSample}
            onRegenerate={performGeneration}
          />
        </div>
      </div>
    </div>
  );
}
