import sys

path = r'C:\Users\노경민\.openclaw\workspace\mt5_bot\config.yaml'
snippet_path = r'C:\Users\노경민\.openclaw\workspace\mt5_bot\strategies\config_snippet.yaml'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

with open(snippet_path, 'r', encoding='utf-8') as f:
    new_snippet = f.read()

import re
# liquidity_sweep_reversal: 부터 trend_regime_sm: 전까지 교체
pattern = re.compile(r'  liquidity_sweep_reversal:.*?(\n\n|  trend_regime_sm:)', re.DOTALL)
result = pattern.sub(new_snippet + '\n\n  trend_regime_sm:', content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(result)
print("Rollback config.yaml successful")
