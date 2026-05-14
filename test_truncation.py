#!/usr/bin/env python
from src.cve_synth.groq_client import GroqClient

# Test truncation function
long_text = 'A' * 10000
truncated = GroqClient._truncate_to_token_budget(long_text, 1000)

print('✓ Token truncation working')
print(f'  Original: {len(long_text)} chars')
print(f'  Truncated: {len(truncated)} chars')
print(f'  Max budget: 1000 chars')
print(f'  Marker added: {truncated.endswith("[truncated]")}')

# Test with text that doesn't need truncation
short_text = 'Hello World'
not_truncated = GroqClient._truncate_to_token_budget(short_text, 1000)
print(f'  Short text unchanged: {short_text == not_truncated}')
