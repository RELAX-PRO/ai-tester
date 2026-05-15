#!/usr/bin/env python
"""
Smoke test / validation script for the training pipeline.

Verifies that all dependencies are installed and the dataset/model loading works
without actually running a full training cycle.

Usage:
    python verify_training_setup.py
"""

import json
import sys
from pathlib import Path

print("=" * 80)
print("CVE-SYNTH TRAINING SETUP VERIFICATION")
print("=" * 80)

# Check 1: Dataset exists
print("\n[CHECK 1] Dataset file exists...")
dataset_path = Path("data/dataset.jsonl")
if not dataset_path.exists():
    print(f"❌ FAIL: {dataset_path} not found")
    print("   Run the pipeline first: python -m cve_synth.cli --limit 10")
    sys.exit(1)

dataset_records = []
try:
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                dataset_records.append(json.loads(line))
    print(f"✓ PASS: Loaded {len(dataset_records)} records from {dataset_path}")
except Exception as e:
    print(f"❌ FAIL: Could not parse {dataset_path}: {e}")
    sys.exit(1)

# Check 2: Dataset structure
print("\n[CHECK 2] Dataset structure...")
required_keys = ["analysis", "source"]
sample_record = dataset_records[0]
if not all(key in sample_record for key in required_keys):
    print(f"❌ FAIL: Record missing required keys. Has: {sample_record.keys()}")
    sys.exit(1)

analysis = sample_record["analysis"]
required_analysis = ["vulnerability_summary", "root_cause", "reasoning_chain", "tags", "assembly_fix"]
if not all(key in analysis for key in required_analysis):
    print(f"❌ FAIL: Analysis missing required keys. Has: {analysis.keys()}")
    sys.exit(1)

print(f"✓ PASS: Dataset structure is valid")
print(f"   Sample record keys: {list(sample_record.keys())}")
print(f"   Sample analysis keys: {list(analysis.keys())}")

# Check 3: Import transformers
print("\n[CHECK 3] Checking transformers library...")
try:
    import transformers
    print(f"✓ PASS: transformers {transformers.__version__} installed")
except ImportError as e:
    print(f"❌ FAIL: transformers not installed: {e}")
    print("   Run: pip install -e '.[train]'")
    sys.exit(1)

# Check 4: Import datasets
print("\n[CHECK 4] Checking datasets library...")
try:
    import datasets
    print(f"✓ PASS: datasets {datasets.__version__} installed")
except ImportError as e:
    print(f"❌ FAIL: datasets not installed: {e}")
    print("   Run: pip install -e '.[train]'")
    sys.exit(1)

# Check 5: Import torch
print("\n[CHECK 5] Checking PyTorch...")
try:
    import torch
    print(f"✓ PASS: torch {torch.__version__} installed")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU device(s): {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠️  WARNING: CUDA not available. Training will be very slow on CPU.")
except ImportError as e:
    print(f"❌ FAIL: torch not installed: {e}")
    print("   Run: pip install -e '.[train]'")
    sys.exit(1)

# Check 6: Import unsloth
print("\n[CHECK 6] Checking Unsloth...")
try:
    from unsloth import FastLanguageModel
    print(f"✓ PASS: Unsloth installed and importable")
except ImportError as e:
    print(f"❌ FAIL: Unsloth not installed: {e}")
    print("   Run: pip install -e '.[train]'")
    print("   Or: pip install unsloth")
    sys.exit(1)

# Check 7: Import TRL
print("\n[CHECK 7] Checking TRL (Transformer Reinforcement Learning)...")
try:
    from trl import SFTTrainer
    print(f"✓ PASS: TRL installed with SFTTrainer")
except ImportError as e:
    print(f"❌ FAIL: TRL not installed: {e}")
    print("   Run: pip install -e '.[train]'")
    sys.exit(1)

# Check 8: Import PEFT
print("\n[CHECK 8] Checking PEFT (Parameter-Efficient Fine-Tuning)...")
try:
    from peft import get_peft_model
    print(f"✓ PASS: PEFT installed")
except ImportError as e:
    print(f"❌ FAIL: PEFT not installed: {e}")
    print("   Run: pip install -e '.[train]'")
    sys.exit(1)

# Check 9: Verify train.py exists
print("\n[CHECK 9] Checking train.py script...")
train_script = Path("src/cve_synth/train.py")
if not train_script.exists():
    print(f"❌ FAIL: {train_script} not found")
    sys.exit(1)
print(f"✓ PASS: {train_script} exists")

# Check 10: Output directory
print("\n[CHECK 10] Checking output directory...")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
print(f"✓ PASS: Output directory is ready at {output_dir.absolute()}")

# Final summary
print("\n" + "=" * 80)
print("VERIFICATION COMPLETE ✓")
print("=" * 80)
print("\nAll checks passed! You're ready to run training.")
print("\nQuick start command:")
print("  python -m cve_synth.train \\")
print("    --dataset-path data/dataset.jsonl \\")
print("    --output-dir ./output/train_results \\")
print("    --num-train-epochs 3")
print("\nFor more options, see TRAINING_GUIDE.md or run:")
print("  python -m cve_synth.train --help")
print()
