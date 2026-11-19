# Bugfix Report: Backend Validation and Model Loading Warnings

**Date:** 2026-11-19  
**System:** Descript AI (Project Descra)  
**Author:** Principal ML & Full-Stack Integration Engineer  

---

## 1. Executive Summary

This engineering bugfix resolves critical request validation errors on the `body.audience` field that were blocking product description generation, fixes deprecation warnings in Hugging Face Transformers model loading, and updates Starlette HTTP status codes to current RFC 9110 standards.

All 34 backend unit and integration tests and all 22 frontend unit tests pass cleanly.

---

## 2. Issues Identified & Root Cause Analysis

### Issue 1: Critical Audience Validation Failure (`String should have at most 100 characters`)
- **Observed Behavior:** Repeated HTTP 422 Unprocessable Content errors on generation requests when using sample templates or typing detailed audience inputs.
- **Root Cause:**
  1. The frontend (`GeneratorForm.tsx`) was concatenating writing tone directly into the audience field:
     ```tsx
     audience: audience.trim() ? `${audience.trim()} (${tone} tone)` : `${tone} tone`
     ```
  2. The backend schema (`ProductDescriptionRequest.audience`) enforces a strict boundary of `max_length=100`.
  3. Pre-configured templates in the frontend had audience strings of ~80–105 characters. When ` (${tone} tone)` was appended (adding 15–16 characters), the string reached 95–120 characters, exceeding the 100-character boundary.
- **Fix Implemented:**
  1. Removed string concatenation in `GeneratorForm.tsx`. `audience` is now transmitted strictly as clean target audience text (`cleanAudience || undefined`).
  2. All pre-configured template audience values in `templates.ts` were standardized to concise, realistic strings between 64 and 86 characters (well below the 100-character ceiling).
  3. Added frontend validation in `GeneratorForm.tsx`: checks `audience.length <= 100`, adds an interactive character counter (`{audience.length}/100`), enforces `maxLength={100}` on the input element, and presents an explicit validation error if violated.

### Issue 2: Deprecated `torch_dtype` Parameter
- **Observed Behavior:** Loading models logged `FutureWarning: torch_dtype is deprecated! Use dtype instead!`.
- **Root Cause:** Hugging Face Transformers v5.20.2 deprecated `torch_dtype` in favor of standard `dtype` in `AutoModelForSeq2SeqLM.from_pretrained`.
- **Fix Implemented:**
  1. Updated `src/descra/inference/predictor.py` line 83 from `torch_dtype=model_dtype` to `dtype=model_dtype`.
  2. Updated `src/descra/models/loader.py` line 74 from `torch_dtype=dtype` to `dtype=dtype`.
  3. Updated `src/descra/evaluation/test_runner.py` line 200 from `torch_dtype=torch.bfloat16` to `dtype=torch.bfloat16`.

### Issue 3: Tied Weights Mapping Notice
- **Observed Behavior:**
  ```text
  The tied weights mapping and config for this model specifies to tie shared.weight to lm_head.weight, but both are present in the checkpoints with different values, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning.
  ```
- **Investigation & Finding:**
  1. In `checkpoints/sft/best_checkpoint/model.safetensors`, both `shared.weight` (token embeddings, 32128x512) and `lm_head.weight` (output projection, 32128x512) are present with distinct learned values (`max_diff = 176.41`), giving the model 76,961,152 parameters.
  2. `T5ForConditionalGeneration` ties `encoder.embed_tokens.weight` and `decoder.embed_tokens.weight` to `shared.weight`.
  3. If `tie_word_embeddings` is set to `False` in `config.json`, Transformers assumes `encoder.embed_tokens` and `decoder.embed_tokens` are also untied. Because their weights were not saved separately into `model.safetensors`, they become `MISSING` and are randomly initialized, which breaks generation (producing blank output).
  4. With `tie_word_embeddings=True`, Transformers v5 automatically and safely leaves `shared.weight` and `lm_head.weight` untied while preserving the tied `encoder.embed_tokens` and `decoder.embed_tokens`.
- **Resolution:**
  - Preserved `tie_word_embeddings=True` to safeguard learned checkpoint weights and prevent random embedding reinitialization.
  - Documented as an expected protective notice from Transformers that does not degrade inference.

### Issue 4: Deprecated Starlette HTTP 422 Constant
- **Observed Behavior:** `StarletteDeprecationWarning: 'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated. Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead.`.
- **Root Cause:** Starlette / FastAPI deprecated `HTTP_422_UNPROCESSABLE_ENTITY` to align with the RFC 9110 HTTP specification (`HTTP_422_UNPROCESSABLE_CONTENT`).
- **Fix Implemented:** Replaced `status.HTTP_422_UNPROCESSABLE_ENTITY` with `status.HTTP_422_UNPROCESSABLE_CONTENT` in `backend/app/main.py`.

---

## 3. Files Modified

| File | Changes |
| :--- | :--- |
| `frontend/src/components/generator/GeneratorForm.tsx` | Removed tone concatenation into audience; added client-side audience validation, `maxLength={100}`, and character counter. |
| `frontend/src/lib/constants/templates.ts` | Refined all template audience values to be between 64 and 86 characters. |
| `frontend/src/test/generator-form.test.tsx` | Added template selector unit tests and verified form validations. |
| `backend/app/main.py` | Replaced `status.HTTP_422_UNPROCESSABLE_ENTITY` with `status.HTTP_422_UNPROCESSABLE_CONTENT`. |
| `src/descra/inference/predictor.py` | Replaced `torch_dtype` with `dtype`. |
| `src/descra/models/loader.py` | Replaced `torch_dtype` with `dtype`. |
| `src/descra/evaluation/test_runner.py` | Replaced `torch_dtype` with `dtype`. |

---

## 4. Verification & Test Results

1. **Backend Tests:** `34/34` passed in `4.73s`:
   - `test_app.py` (6 passed)
   - `test_config.py` (4 passed)
   - `test_generation_endpoints.py` (10 passed)
   - `test_generation_service.py` (6 passed)
   - `test_health.py` (3 passed)
   - `test_reliability.py` (5 passed)
2. **Frontend Tests:** `22/22` passed in `6.08s` (all Vitest suites).
3. **End-to-End Generation:** Tested and confirmed generating fluent product descriptions without validation failures.
