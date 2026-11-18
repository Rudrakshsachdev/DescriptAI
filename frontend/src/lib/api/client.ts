import { ApiErrorResponse } from "@/lib/types/api";

export class ApiError extends Error {
  statusCode: number;
  errorCode: string;
  details?: unknown;
  rawBody?: unknown;

  constructor(
    statusCode: number,
    errorCode: string,
    message: string,
    details?: unknown,
    rawBody?: unknown
  ) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    this.errorCode = errorCode;
    this.details = details;
    this.rawBody = rawBody;
  }
}

const DEFAULT_BASE_URL = "http://localhost:8000";

export function getApiBaseUrl(): string {
  if (typeof window !== "undefined" && process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL.replace(/\/+$/, "");
  }
  return (process.env.NEXT_PUBLIC_API_BASE_URL || DEFAULT_BASE_URL).replace(/\/+$/, "");
}

interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

export async function apiClient<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { timeoutMs = 45000, ...fetchOptions } = options;
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const headers = new Headers(fetchOptions.headers);
  if (!headers.has("Content-Type") && fetchOptions.body && typeof fetchOptions.body === "string") {
    headers.set("Content-Type", "application/json");
  }
  headers.set("Accept", "application/json");

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      headers,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    // Handle empty 204 or similar
    if (response.status === 204) {
      return {} as T;
    }

    const contentType = response.headers?.get ? response.headers.get("content-type") : "application/json";
    const isJson = !contentType || contentType.includes("application/json");
    const responseData = isJson ? await response.json() : await response.text();

    if (!response.ok) {
      const errorData = (isJson ? responseData : { message: responseData }) as ApiErrorResponse;
      const errorCode = errorData.error || `http_${response.status}`;
      const errorMessage =
        errorData.message ||
        (response.status === 503
          ? "Service or model is currently unavailable."
          : response.status === 504
          ? "Inference generation timed out on backend."
          : `HTTP error ${response.status}`);

      throw new ApiError(response.status, errorCode, errorMessage, errorData.details, responseData);
    }

    return responseData as T;
  } catch (error: unknown) {
    clearTimeout(timeoutId);

    if (error instanceof ApiError) {
      throw error;
    }

    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(
        504,
        "request_timeout",
        `Request to backend timed out after ${timeoutMs / 1000} seconds. The model may still be busy.`
      );
    }

    const message =
      error instanceof Error
        ? error.message.includes("Failed to fetch")
          ? "Cannot connect to the Descript AI backend. Please verify the service is running at " + baseUrl
          : error.message
        : "An unknown network error occurred while reaching the backend.";

    throw new ApiError(0, "network_error", message);
  }
}
