import json

with open('all_classifiers_30_split_results.json') as f:
    data = json.load(f)

print(f"{'Classifier':<22} | {'Raw Baseline':<18} | {'Barak Or (S-T)':<18} | {'Ali et al. (S-T)':<18} | {'Proposed':<18}")
print("-" * 105)

for clf, v in data['classifiers'].items():
    metrics = v.get('metrics', {})
    raw = metrics.get('raw', {}).get('accuracy', {})
    barak = metrics.get('barak_or', {}).get('accuracy', {})
    ali = metrics.get('ali_et_al', {}).get('accuracy', {})
    prop = metrics.get('proposed', {}).get('accuracy', {})
    
    r_str = f"{raw.get('mean', 0)*100:.2f}%" if raw else "N/A"
    b_str = f"{barak.get('mean', 0)*100:.2f}%" if barak else "N/A"
    a_str = f"{ali.get('mean', 0)*100:.2f}%" if ali else "N/A"
    p_str = f"{prop.get('mean', 0)*100:.2f}%" if prop else "N/A"
    
    print(f"{clf:<22} | {r_str:<18} | {b_str:<18} | {a_str:<18} | {p_str:<18}")
