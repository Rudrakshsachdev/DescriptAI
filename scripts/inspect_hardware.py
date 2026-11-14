"""CLI script to run hardware inspection and output the hardware report."""

import json
import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from descra.utils.hardware import (
    generate_hardware_markdown_report,
    inspect_hardware,
)


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    report_dir = os.path.join(root_dir, "reports", "hardware")
    os.makedirs(report_dir, exist_ok=True)

    print("=" * 60)
    print("DESCRA - PHASE 1: HARDWARE & ENVIRONMENT INSPECTION")
    print("=" * 60)

    profile = inspect_hardware(root_dir)

    print(f"OS: {profile.os_name} {profile.os_release} ({profile.machine_arch})")
    print(f"Python: {profile.python_version}")
    print(f"CPU: {profile.processor}")
    print(f"CPU Cores: {profile.cpu_physical_cores} Physical / {profile.cpu_logical_cores} Logical")
    print(f"System RAM: {profile.total_ram_gb:.2f} GB (Available: {profile.available_ram_gb:.2f} GB, {profile.ram_usage_percent}% used)")
    print(f"Disk Free: {profile.disk_free_gb:.2f} GB (Total: {profile.disk_total_gb:.2f} GB)")
    print(f"GPU Available: {profile.gpu_available}")
    if profile.gpu_available:
        print(f"GPU Model: {profile.gpu_name}")
        print(f"GPU VRAM: {profile.gpu_total_vram_mb:.0f} MB (Free: {profile.gpu_free_vram_mb:.0f} MB)")
        print(f"NVIDIA Driver: {profile.gpu_driver_version}")
        print(f"CUDA Version: {profile.cuda_version}")
    print(f"PyTorch Installed: {profile.torch_installed}")
    print(f"Recommended Model: {profile.recommended_model}")
    print(f"Recommended Batch Size: {profile.recommended_batch_size}")
    print(f"Recommended Precision: {profile.recommended_precision}")
    print("-" * 60)

    # Save Markdown report
    md_path = os.path.join(report_dir, "HARDWARE_ASSESSMENT.md")
    report_md = generate_hardware_markdown_report(profile)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Hardware assessment markdown report saved to:\n  {md_path}")

    # Save JSON profile
    json_path = os.path.join(report_dir, "hardware_profile.json")
    from dataclasses import asdict
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, indent=2)
    print(f"Hardware profile JSON saved to:\n  {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

