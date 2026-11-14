# Contributing to DESCЯA
## Developer Guide, Team Protocols & Engineering Standards

Welcome to the **Descra** engineering codebase. This guide outlines the conventions, workflows, and quality standards for all team members (Rudraksh Sachdeva, Shivansee, K Jahnvi, Kriti Abrol) contributing to Phase 1 and subsequent phases.

---

## 1. Environment & Setup (Windows 11 / PowerShell)

Descra uses an isolated virtual environment (`.venv`) powered by **Python 3.10.11** with CUDA 12.4-enabled PyTorch.

### 1.1 Activating the Virtual Environment
Always ensure the virtual environment is active before running scripts, notebooks, or tests.

In PowerShell:
```powershell
# Navigate to repository root
cd C:\DescriptAI

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks execution of scripts, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once).*

### 1.2 Verifying Hardware & CUDA Availability
Verify that PyTorch recognizes your NVIDIA RTX 4050 GPU:
```powershell
python scripts/inspect_hardware.py
```
Expected output:
- CUDA Available: `True`
- Device Name: `NVIDIA GeForce RTX 4050 Laptop GPU`
- Dedicated VRAM: `~6,140 MB`

---

## 2. Git Workflow & Branching Strategy

To maintain an audit trail for academic evaluation, follow this structured Git workflow:

```
main (Production-ready milestones only)
 └── feature/step4-eda
 └── feature/step5-foundations
 └── feature/step6-flan-t5-baseline
 └── fix/token-truncation-bug
```

### 2.1 Branch Naming Convention
- **Features / Milestones:** `feature/step<N>-<short-description>` (e.g., `feature/step6-baseline-inference`)
- **Bug Fixes:** `fix/<issue-name>` (e.g., `fix/oom-batch-collation`)
- **Documentation:** `docs/<topic>` (e.g., `docs/evaluation-metrics`)
- **Refactoring:** `refactor/<module-name>` (e.g., `refactor/logging-utility`)

### 2.2 Branch Lifecycle Commands
```powershell
# 1. Ensure you have the latest main branch
git checkout main
git pull origin main

# 2. Create and switch to your feature branch
git checkout -b feature/step6-baseline-inference

# 3. Work on your changes, test them locally
pytest tests/

# 4. Stage your changes
git add src/ configs/ tests/

# 5. Commit with a conventional commit message
git commit -m "feat(model): implement baseline inference engine for FLAN-T5"

# 6. Push to remote and open a Pull Request
git push -u origin feature/step6-baseline-inference
```

---

## 3. Commit Message Standards (Conventional Commits)

Commit messages must be detailed, descriptive, and follow the **Conventional Commits** specification:

```
<type>(<scope>): <short imperative summary>

[Optional detailed body explaining WHAT changed and WHY]

[Optional footer: references issue or milestone]
```

### Allowed Types:
- `feat`: A new feature or pipeline capability (e.g., `feat(eda): generate subword token length percentiles`).
- `fix`: A bug fix (e.g., `fix(dataset): prevent cross-split description leakage`).
- `docs`: Documentation updates (e.g., `docs(arch): complete mathematical formulation of attention`).
- `test`: Adding or updating automated tests (e.g., `test(hardware): add GPU GEMM memory test`).
- `refactor`: Code changes that neither fix bugs nor add features.
- `chore`: Maintenance tasks, config tweaks, or gitignore updates.

---

## 4. Code Quality & Python Conventions

### 4.1 General Principles
1. **PEP 8 Compliance:** Follow standard Python formatting conventions (4-space indentation, snake_case functions/variables, PascalCase classes).
2. **Type Hints:** Add static type annotations for function signatures and dataclasses.
   ```python
   def format_model_input(record: dict[str, Any]) -> str:
       ...
   ```
3. **Docstrings:** Public modules, classes, and methods must have docstrings explaining arguments, returns, and exceptions.
4. **No Raw Print Statements:** Never use `print()` in production library code (`src/descra/`). Always use the centralized logger:
   ```python
   from descra.utils.logging import get_logger
   logger = get_logger("descra.module")
   logger.info("Batch processed successfully.")
   ```
5. **No Hardcoded Configurations:** All paths, batch sizes, learning rates, and thresholds belong in YAML files under `configs/`.

---

## 5. Testing & Verification Standards

No code should be merged into `main` without automated test coverage.

### 5.1 Running the Test Suite
```powershell
# Run all unit tests
pytest tests/ -v

# Run tests with output logging
pytest tests/ -v -s
```

### 5.2 Test Placement Rules
- Unit tests for `src/descra/data/` live in `tests/test_dataset.py` and `tests/test_preprocessing.py`.
- Hardware and CUDA tests live in `tests/test_hardware.py`.
- Model and inference tests live in `tests/test_model.py`.
- All tests must pass before submitting commits or pull requests.

---

## 6. Data & Hardware Hygiene

### 6.1 What NEVER Goes Into Git
- **Raw or processed datasets:** `data/raw/*`, `data/processed/*`, `data/interim/*`. (Only `.gitkeep` and manifests are committed).
- **Model weights & checkpoints:** `artifacts/models/*`, `artifacts/checkpoints/*`.
- **Virtual environments:** `.venv/`, `env/`.
- **OS artifacts:** `Thumbs.db`, `.DS_Store`.

### 6.2 GPU VRAM Budgeting (RTX 4050 6GB Rule)
- Never configure a batch size or sequence length that exceeds **4.5 GB peak VRAM**, leaving headroom for OS and background window managers.
- Always use FP16 mixed precision during training and inference.
- Call `torch.cuda.empty_cache()` after executing memory-intensive evaluations or scripts.

