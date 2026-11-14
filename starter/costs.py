"""Offline generation-cost estimates using user-supplied standard API rates."""
import argparse
import json
import shutil
from decimal import Decimal
from pathlib import Path

MODEL = 'gemini-3.5-flash-lite'
def price_call(call):
    usage = call.get('usage')
    if not usage:
        return 0.0
    if call.get('model') != MODEL or usage.get('cached_content_token_count') or usage.get('thoughts_token_count'):
        return None
    inputs, outputs = usage.get('prompt_token_count'), usage.get('candidates_token_count')
    if any(type(n) is not int or n < 0 for n in (inputs, outputs)):
        return None
    return float((Decimal(inputs)*Decimal('0.30') + Decimal(outputs)*Decimal('2.50')) / Decimal(1000000))


def total(values):
    return None if any(v is None for v in values) else float(sum((Decimal(str(v)) for v in values), Decimal(0)))


def clean(item):
    for key in ('generation_cost_usd', 'generation_pricing', 'missing_usage_calls_assumed_free', 'cost_status'):
        item.pop(key, None)


def annotate(traces, summary):
    # service_cost_usd is the text-generation estimate only; see SETUP_CMD.md.
    for trace in traces:
        clean(trace)
        calls = trace.get('model_calls', [])
        for call in calls:
            clean(call)
        trace['service_cost_usd'] = total([price_call(call) for call in calls])
    clean(summary)
    summary['service_cost_usd'] = total([t['service_cost_usd'] for t in traces])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path)
    args = parser.parse_args()
    for folder in args.folders:
        paths = [folder/'traces.json', folder/'summary.json']
        traces, summary = [json.loads(p.read_text(encoding='utf-8-sig')) for p in paths]
        annotate(traces, summary)
        for path, data in zip(paths, [traces, summary]):
            backup = path.with_suffix('.json.before-cost.bak')
            if not backup.exists():
                shutil.copy2(path, backup)
            temp = path.with_suffix('.json.tmp')
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            temp.replace(path)
        print(folder.name, summary['service_cost_usd'])


if __name__ == '__main__':
    main()

