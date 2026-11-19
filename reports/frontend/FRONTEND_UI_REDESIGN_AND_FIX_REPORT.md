# Descript AI — Frontend Static Asset 404 Diagnosis & UI/UX Redesign Report

**Date:** November 18, 2026  
**Branch:** `new/frontend`  
**Application:** Descript AI (FastAPI + Next.js + Fine-Tuned FLAN-T5)

---

## 1. Root Cause Analysis: Static Asset 404 Errors

### 1.1 The Symptom
During development execution (`npm run dev`), the Next.js server yielded recurring 404 errors for internal CSS and JavaScript chunk requests:
```text
GET /_next/static/css/app/layout.css?v=... 404
GET /_next/static/chunks/app-pages-internals.js 404
GET /_next/static/chunks/main-app.js?v=... 404
GET /_next/static/chunks/app/page.js 404
```

### 1.2 Diagnostic Investigation & Findings
Through deep environment and process inspection, two interlocking root causes were identified:

1. **Orphaned / Stale Development Server Process (Port Collision):**
   - A previous Node.js server process (`PID 7448`, along with workers `PID 26004` and `PID 30452`) had been spawned earlier and remained persistently bound to `0.0.0.0:3000`.
   - When subsequent production builds (`next build`) or file modifications were made, the on-disk `.next` artifacts were regenerated with new hashes.
   - However, incoming browser requests to `http://localhost:3000` were still being intercepted and served by the stale zombie process holding outdated in-memory route manifests.
   - Any new `npm run dev` command detected port 3000 occupied and automatically fell back to port 3001 (`Port 3000 is in use, trying 3001 instead.`), while the browser continued querying port 3000 against the obsolete process.

2. **Missing App Router Error & Fallback Boundaries (`/_error` Loop):**
   - When requests encountered stale chunk mismatches or not-found routes, Next.js 14 App Router searched for `app/not-found.tsx` and `app/error.tsx`.
   - Because neither existed, Next.js fell back to the legacy Pages router internal `/_error.js` compiler (`Compiling /_error ...`).
   - This caused Next.js to attempt serving Pages-router-specific chunks that did not exist in an App Router-only project, locking the development server into a 404 cycle accompanied by `missing required error components, refreshing...`.

---

## 2. Technical Fixes Implemented

1. **Process Cleanup & Port Recovery:**
   - Identified and terminated orphaned Node.js processes holding port 3000 (`PID 7448`, `26004`, `30452`).
   - Purged stale `.next` cache artifacts to ensure clean chunk compilation.

2. **App Router Error & Fallback Handlers:**
   - Created `frontend/src/app/not-found.tsx`: Clean App Router 404 component providing user guidance and a quick link back to Overview.
   - Created `frontend/src/app/error.tsx`: Root interactive error boundary with retry capability (`reset()`).
   - Created `frontend/src/app/global-error.tsx`: Critical error boundary catching root layout rendering failures.

3. **Asset Resolution Verification:**
   - Verified that running `next dev` binds directly to port 3000 and resolves all assets:
     - Root HTML: **HTTP 200** (27,036 bytes)
     - `main-app.js`: **HTTP 200** (6,033,930 bytes)
     - `app-pages-internals.js`: **HTTP 200** (133,075 bytes)

---

## 3. UI/UX Redesign & Modern SaaS Polish

The interface was completely transformed into a modern, commercial-grade AI SaaS application adhering to the aesthetic standards of Stripe, Linear, and Vercel.

### 3.1 Design System & Component Upgrades
- **Color Palette & Surfaces:** Neutral slate foundation (`bg-slate-50`) paired with crisp white card containers (`bg-white border border-slate-200/90 shadow-xs`) and restrained blue accents (`#2563eb`).
- **Typography & Hierarchy:** Refined font smoothing, semantic headings, balanced tracking, and dedicated monospace formatting for code, tokens, and latency metrics.
- **Buttons (`Button.tsx`):** Modern micro-interactions, subtle shadow elevation (`shadow-xs`), consistent padding across `sm`, `md`, and `lg`, and integrated loading spinners.
- **Navigation (`Sidebar.tsx`):**
  - Removed placeholder / dead links (`History & Logs` disabled item removed).
  - Clean three-link workspace navigation: **Overview**, **Generate Description**, and **Model Information**.
  - Accessible `aria-current="page"` indicators and smooth mobile drawer backdrop.

### 3.2 Redesigned Pages

#### 1. Overview Dashboard (`/`)
- Replaced dark decorative banner with an executive SaaS welcome header and clear value proposition.
- Integrated **Real-Time Operational Inference Telemetry** bar displaying live API status, model identifier, device/precision, and total parameter count (`76,961,152`).
- 3-step structured generation workflow guide explaining the pipeline from attributes to Beam-2 generated copy.
- 3 technical foundation cards highlighting Local Model Execution, Beam Search Decoding, and 100% Private Offline Processing.

#### 2. Generator Workspace (`/generate`)
- **Structured Left Column:**
  - Product Information form with clear required (`*`) vs. optional fields.
  - One-click **Sample Product** loader and **Clear Form** reset.
  - Tag chips for bullet features and key-value specification table.
  - Target audience helper text guiding user input.
  - Collapsible **Decoding & Generation Settings** panel with clean sliders for beam width (1–5), max tokens (20–256), repetition penalty, and length penalty.
- **Dedicated Right Column (Output Panel):**
  - High-readability prose typography with generous line-height.
  - Telemetry bar displaying Latency (ms), Output word/token counts, and Model ID.
  - **Copy Description** button with instant `Copied!` visual confirmation.
  - **Regenerate** button allowing quick re-runs with identical or tweaked parameters.
  - Expandable **Raw Input Prompt** inspector displaying the exact canonical template fed to the FLAN-T5 encoder.

#### 3. Model Diagnostics (`/model`)
- Operational metadata retrieved live via `GET /api/v1/model/info`.
- Parameter breakdown, runtime device (`cuda:0` / `cpu`), precision format (`bfloat16` / `float32`).
- Default production decoding hyperparameters card.
- FLAN-T5 architecture reference blueprint (Vocabulary 32,128, Hidden Dim 512, Attention Heads 8, Layers 6/6).
- Graceful offline state with retry handler.

---

## 4. Verification & Testing Results

### 4.1 Frontend Unit & Component Tests (`npm run test`)
```text
 ✓ src/test/api-services.test.ts (7 tests)
 ✓ src/test/model-page.test.tsx (2 tests)
 ✓ src/test/navigation.test.tsx (4 tests)
 ✓ src/test/generator-form.test.tsx (8 tests)

 Test Files  4 passed (4)
      Tests  21 passed (21)
```

### 4.2 Next.js Production Build (`npm run build`)
```text
Route (app)                              Size     First Load JS
┌ ○ /                                    5.87 kB         109 kB
├ ○ /_not-found                          138 B          87.4 kB
├ ○ /generate                            9.19 kB         104 kB
└ ○ /model                               5.01 kB        99.7 kB
+ First Load JS shared by all            87.3 kB

✓ Generating static pages (6/6)
✓ Finalizing page optimization
✓ Build succeeded with zero errors or warnings
```

### 4.3 End-to-End Live Backend Inference Verification
Tested live against the running FastAPI backend and trained checkpoint:
- `GET /health` → **HTTP 200** (`status: healthy`, `version: 1.0.0`)
- `GET /health/ready` → **HTTP 200** (`status: ready`, `model_loaded: true`, `device: cuda:0`, `half_precision: true`)
- `GET /api/v1/model/info` → **HTTP 200** (`google/flan-t5-small`, `76,961,152` parameters, `cuda:0`)
- `POST /api/v1/generate` → **HTTP 200**
  - **Input:** `Hydrating Facial Cream`, features: `["2% Hyaluronic Acid", "Ceramides"]`
  - **Generated Output:** *"Hydrating Facial Cream is a hydrating facial cream with 2% Hyaluronic Acid and Ceramides."*
  - **Latency:** `1575.58 ms`, **Tokens:** `28`

### 4.4 Phase 1 Regression Suite
```text
pytest tests/ -q
53 passed in 41.22s (100% pass rate)
```

---

## 5. Remaining Limitations

1. **Database Persistence & Generation History:**
   - The FastAPI backend currently operates as a stateless inference serving layer.
   - Persistent description history requires a database tier (PostgreSQL / SQLAlchemy) in a future backend release.
   - Per design guidelines, no fake history or browser-memory masquerading was introduced; placeholder navigation items were removed to maintain a zero-dead-link application standard.

---

## 6. How to Run Locally

### Terminal 1: FastAPI Backend
```powershell
cd C:\DescriptAI
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000 --reload
```

### Terminal 2: Next.js Frontend
```powershell
cd C:\DescriptAI\frontend
npm.cmd run dev
```

- **Frontend URL:** [http://localhost:3000](http://localhost:3000)
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
