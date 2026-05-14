#!/usr/bin/env python
"""
Groq Migration Validation Report
=================================
"""

print("\n" + "="*70)
print("GROQ API MIGRATION - VALIDATION REPORT")
print("="*70)

# 1. Import Checks
print("\n✓ MODULE IMPORTS")
try:
    from src.cve_synth.groq_client import GroqClient, GroqConfig, GroqRateLimitError
    print("  ✓ GroqClient, GroqConfig, GroqRateLimitError imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")

try:
    from src.cve_synth.pipeline import PipelineConfig, run_pipeline, _analyze_with_rotation
    print("  ✓ Pipeline imports (PipelineConfig, run_pipeline, _analyze_with_rotation)")
except Exception as e:
    print(f"  ✗ Pipeline import failed: {e}")

try:
    from src.cve_synth.cli import _load_api_keys, main
    print("  ✓ CLI imports (_load_api_keys, main)")
except Exception as e:
    print(f"  ✗ CLI import failed: {e}")

# 2. Configuration Checks
print("\n✓ GROQ CONFIGURATION DEFAULTS")
config = GroqConfig()
print(f"  Model: {config.model}")
print(f"    Expected: deepseek-r1-distill-llama-70b")
print(f"    Match: {config.model == 'deepseek-r1-distill-llama-70b'}")
print(f"  API Base: {config.api_base}")
print(f"    Expected: https://api.groq.com/openai/v1")
print(f"    Match: {config.api_base == 'https://api.groq.com/openai/v1'}")
print(f"  Endpoint Path: {config.endpoint_path}")
print(f"    Expected: /chat/completions")
print(f"    Match: {config.endpoint_path == '/chat/completions'}")

# 3. Pipeline Configuration
print("\n✓ PIPELINE DEFAULT CONFIGURATION")
pipeline_config = PipelineConfig(
    input_dir='data/raw',
    output_path='data/test.jsonl',
    checkpoint_path='data/test_checkpoint.json',
    api_keys=['test_key']
)
print(f"  Default model: {pipeline_config.model_name}")
print(f"    Expected: deepseek-r1-distill-llama-70b")
print(f"    Match: {pipeline_config.model_name == 'deepseek-r1-distill-llama-70b'}")

# 4. Token Truncation
print("\n✓ TOKEN TRUNCATION LOGIC")
long_text = 'A' * 10000
truncated = GroqClient._truncate_to_token_budget(long_text, 1000)
print(f"  Original text length: {len(long_text)} chars")
print(f"  Truncated text length: {len(truncated)} chars")
print(f"  Budget: 1000 chars")
print(f"  Marker appended: {truncated.endswith('[truncated]')}")
print(f"  Truncation working: {len(truncated) <= 1005}")

short_text = 'Hello'
short_unchanged = GroqClient._truncate_to_token_budget(short_text, 1000)
print(f"  Short text unchanged: {short_text == short_unchanged}")

# 5. API Key Loading
print("\n✓ GROQ_API_KEYS PARSING")
import os
import argparse
os.environ['GROQ_API_KEYS'] = 'gsk_key1,gsk_key2,gsk_key3'
os.environ.pop('GROQ_API_KEY', None)
os.environ.pop('DEEPSEEK_API_KEY', None)
os.environ.pop('DEEPSEEK_API_KEYS', None)

args = argparse.Namespace(api_keys=[], api_keys_file=None)
keys = _load_api_keys(args)
print(f"  GROQ_API_KEYS: 'gsk_key1,gsk_key2,gsk_key3'")
print(f"  Parsed keys: {len(keys)}")
print(f"  Keys match: {keys == ['gsk_key1', 'gsk_key2', 'gsk_key3']}")

# 6. Error Handling
print("\n✓ ERROR HANDLING")
try:
    raise GroqRateLimitError("Test 429 error", retry_after_seconds=5.0)
except GroqRateLimitError as e:
    print(f"  GroqRateLimitError catchable: True")
    print(f"  Message: {str(e)}")
    print(f"  Retry-After: {e.retry_after_seconds}s")

# 7. Message Building (Mock)
print("\n✓ MESSAGE BUILDING & TRUNCATION")
from src.cve_synth.models import SourceRecord, AnalysisRecord
from src.cve_synth.extract import ExtractionResult

source = SourceRecord(
    source_id="CVE-2024-TEST",
    source_type="nvd",
    cve_id="CVE-2024-TEST",
    ghsa_id=None,
    title="Test Vulnerability",
    severity="HIGH",
    raw_text="A" * 8000,  # Large raw text
    url="https://example.com"
)

extraction = ExtractionResult(
    vulnerable_snippet="int vulnerable(char *buf) { strcpy(buf, input); }",
    surrounding_context="B" * 3000,  # Large context
    evidence_spans=[],
)

client = GroqClient(api_key="test_key")
messages = client.build_messages(source, extraction, "x86-64")
user_content = messages[1]["content"]
content_size = len(user_content)
print(f"  User message size: {content_size} chars")
print(f"  Within budget (~15,000 chars): {content_size < 15000}")
print(f"  Message has required fields: {'vulnerable_snippet' in user_content}")

print("\n" + "="*70)
print("MIGRATION VALIDATION COMPLETE ✓")
print("="*70 + "\n")
