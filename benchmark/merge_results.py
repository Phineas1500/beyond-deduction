#!/usr/bin/env python3
"""Merge n500 and n1500 results into combined n2000 dataset."""

import pickle
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load both result files
with open(os.path.join(script_dir, 'factorial_results/factorial_gemma29b_n500.pkl'), 'rb') as f:
    r500 = pickle.load(f)

with open(os.path.join(script_dir, 'factorial_results/factorial_gemma29b_n1500_extra.pkl'), 'rb') as f:
    r1500 = pickle.load(f)

# Merge
merged = {
    'h1_results': r500['h1_results'] + r1500['h1_results'],
    'h2_results': r500['h2_results'] + r1500['h2_results'],
    'pairs': r500['pairs'] + r1500['pairs'],
}

n = len(merged['h1_results'])
print(f'Total pairs: {n}')

# Recompute statistics
h1_strong = np.mean([r['strong'] for r in merged['h1_results']])
h2_strong = np.mean([r['strong'] for r in merged['h2_results']])

print(f'H1 Strong: {h1_strong:.1%}')
print(f'H2 Strong: {h2_strong:.1%}')
print(f'Conservation Law Sum: {h1_strong + h2_strong:.1%}')

# Stratify
h1_fail_h2_success = sum(1 for h1, h2 in zip(merged['h1_results'], merged['h2_results'])
                         if h1['strong'] == 0 and h2['strong'] == 1)
h1_success_h2_fail = sum(1 for h1, h2 in zip(merged['h1_results'], merged['h2_results'])
                         if h1['strong'] == 1 and h2['strong'] == 0)
h1_success_h2_success = sum(1 for h1, h2 in zip(merged['h1_results'], merged['h2_results'])
                            if h1['strong'] == 1 and h2['strong'] == 1)
h1_fail_h2_fail = sum(1 for h1, h2 in zip(merged['h1_results'], merged['h2_results'])
                      if h1['strong'] == 0 and h2['strong'] == 0)

print()
print('Behavioral Stratification:')
print(f'  h1_fail_h2_success (GOLD): {h1_fail_h2_success} ({h1_fail_h2_success/n:.1%})')
print(f'  h1_success_h2_fail:        {h1_success_h2_fail} ({h1_success_h2_fail/n:.1%})')
print(f'  h1_success_h2_success:     {h1_success_h2_success} ({h1_success_h2_success/n:.1%})')
print(f'  h1_fail_h2_fail:           {h1_fail_h2_fail} ({h1_fail_h2_fail/n:.1%})')

# Get gold indices
gold_indices = [i for i, (h1, h2) in enumerate(zip(merged['h1_results'], merged['h2_results']))
                if h1['strong'] == 0 and h2['strong'] == 1]

print()
print(f'GOLD PAIRS (parent output): {len(gold_indices)} total')
print(f'Indices: {gold_indices}')

# Save merged
merged['gold_indices'] = gold_indices
merged['h1_strong'] = h1_strong
merged['h2_strong'] = h2_strong
merged['stratification'] = {
    'h1_fail_h2_success': h1_fail_h2_success,
    'h1_success_h2_fail': h1_success_h2_fail,
    'h1_success_h2_success': h1_success_h2_success,
    'h1_fail_h2_fail': h1_fail_h2_fail,
}

output_path = os.path.join(script_dir, 'factorial_results/factorial_gemma29b_n2000_merged.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(merged, f)
print()
print(f'Saved merged results to {output_path}')
