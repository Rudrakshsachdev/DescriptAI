"use client";

import * as React from "react";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";
import { FeatureTagInput } from "@/components/generator/FeatureTagInput";
import { SpecificationInput } from "@/components/generator/SpecificationInput";
import { ConfigPanel, DEFAULT_CONFIG } from "@/components/generator/ConfigPanel";
import { OutputPanel } from "@/components/generator/OutputPanel";
import { generateDescription } from "@/lib/api/services";
import { ApiError } from "@/lib/api/client";
import {
  GenerationConfig,
  GenerationData,
  ProductDescriptionRequest,
} from "@/lib/types/api";
import { Sparkles, AlertTriangle, RotateCcw } from "lucide-react";

const EXAMPLE_PRODUCT: ProductDescriptionRequest = {
  product_name: "Hydrating Daily Face Moisturizer with Hyaluronic Acid",
  category: "Beauty & Personal Care",
  brand: "DermaPure",
  features: [
    "Formulated with 2% pure hyaluronic acid and ceramides",
    "Delivers 48-hour continuous hydration",
    "Fragrance-free, non-comedogenic, and hypoallergenic",
  ],
  specifications: {
    Volume: "50ml / 1.7 fl oz",
    "Skin Type": "Sensitive, Dry, Combination",
    Formulation: "Lightweight Cream-Gel",
  },
  audience: "Sensitive skin daily routine",
  generation_config: { ...DEFAULT_CONFIG },
};

export function GeneratorForm() {
  const [productName, setProductName] = React.useState("");
  const [category, setCategory] = React.useState("");
  const [brand, setBrand] = React.useState("");
  const [features, setFeatures] = React.useState<string[]>([]);
  const [specifications, setSpecifications] = React.useState<Record<string, string>>({});
  const [audience, setAudience] = React.useState("");
  const [config, setConfig] = React.useState<GenerationConfig>({ ...DEFAULT_CONFIG });

  const [formError, setFormError] = React.useState<string | null>(null);
  const [apiError, setApiError] = React.useState<string | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [outputData, setOutputData] = React.useState<GenerationData | null>(null);

  const handleLoadExample = () => {
    setProductName(EXAMPLE_PRODUCT.product_name);
    setCategory(EXAMPLE_PRODUCT.category || "");
    setBrand(EXAMPLE_PRODUCT.brand || "");
    setFeatures(EXAMPLE_PRODUCT.features || []);
    setSpecifications(EXAMPLE_PRODUCT.specifications || {});
    setAudience(EXAMPLE_PRODUCT.audience || "");
    setConfig(EXAMPLE_PRODUCT.generation_config || { ...DEFAULT_CONFIG });
    setFormError(null);
    setApiError(null);
  };

  const handleClear = () => {
    setProductName("");
    setCategory("");
    setBrand("");
    setFeatures([]);
    setSpecifications({});
    setAudience("");
    setConfig({ ...DEFAULT_CONFIG });
    setFormError(null);
    setApiError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
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

    const payload: ProductDescriptionRequest = {
      product_name: cleanName,
      category: category.trim() || undefined,
      brand: brand.trim() || undefined,
      features: features.length > 0 ? features : undefined,
      specifications: Object.keys(specifications).length > 0 ? specifications : undefined,
      audience: audience.trim() || undefined,
      generation_config: config,
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
        setApiError(
          err instanceof Error
            ? err.message
            : "An unexpected error occurred while communicating with the backend."
        );
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
      {/* Left Column: Form Inputs */}
      <div className="lg:col-span-7 space-y-6">
        <form onSubmit={handleSubmit} className="bg-white border border-slate-200/90 rounded-xl p-6 shadow-xs space-y-5">
          <div className="flex items-center justify-between pb-4 border-b border-slate-100">
            <div>
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800">
                Product Information
              </h2>
              <p className="text-xs text-slate-500">
                Enter details for the Seq2Seq Transformer model.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={handleLoadExample}
                disabled={isLoading}
              >
                Sample Product
              </Button>
              <button
                type="button"
                onClick={handleClear}
                disabled={isLoading || !productName}
                className="text-xs text-slate-400 hover:text-slate-600 disabled:opacity-30 p-1"
                title="Clear all fields"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Form Error Banner */}
          {apiError && (
            <div className="p-3.5 bg-rose-50 border border-rose-200 rounded-lg text-xs text-rose-800 flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <span className="font-semibold block mb-0.5">Generation Failed</span>
                <p className="leading-relaxed">{apiError}</p>
              </div>
            </div>
          )}

          {/* Product Name */}
          <Input
            label="Product Title / Name"
            required
            disabled={isLoading}
            value={productName}
            onChange={(e) => {
              setProductName(e.target.value);
              if (formError) setFormError(null);
            }}
            error={formError || undefined}
            placeholder="e.g., Organic Hydrating Face Cream"
            helperText="The primary commercial product title."
          />

          {/* Category & Brand (2 Column) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Input
              label="Category"
              disabled={isLoading}
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              placeholder="e.g., Beauty & Personal Care"
            />
            <Input
              label="Brand / Manufacturer"
              disabled={isLoading}
              value={brand}
              onChange={(e) => setBrand(e.target.value)}
              placeholder="e.g., DermaPure"
            />
          </div>

          {/* Target Audience */}
          <Input
            label="Target Audience (Optional)"
            disabled={isLoading}
            value={audience}
            onChange={(e) => setAudience(e.target.value)}
            placeholder="e.g., Sensitive skin daily routine, athletes, professionals"
          />

          {/* Structured Features Tag Input */}
          <FeatureTagInput
            features={features}
            onChange={setFeatures}
            disabled={isLoading}
          />

          {/* Key-Value Specifications */}
          <SpecificationInput
            specifications={specifications}
            onChange={setSpecifications}
            disabled={isLoading}
          />

          {/* Bounded Generation Configuration */}
          <ConfigPanel
            config={config}
            onChange={setConfig}
            disabled={isLoading}
          />

          {/* Submit Action */}
          <div className="pt-2">
            <Button
              type="submit"
              size="lg"
              className="w-full shadow-sm"
              isLoading={isLoading}
              disabled={isLoading}
              leftIcon={<Sparkles className="w-4 h-4" />}
            >
              {isLoading ? "Generating with FLAN-T5..." : "Generate Description"}
            </Button>
            <p className="text-[11px] text-center text-slate-400 mt-2">
              Powered by local FLAN-T5-small checkpoint with Beam-2 decoding.
            </p>
          </div>
        </form>
      </div>

      {/* Right Column: Output Panel */}
      <div className="lg:col-span-5 sticky top-24">
        <OutputPanel
          data={outputData}
          isLoading={isLoading}
          onLoadExample={handleLoadExample}
        />
      </div>
    </div>
  );
}
