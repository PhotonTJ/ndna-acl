"""Export the recursive self-training runs (and their human-data control) to results/recursive_runs.json.

    python export_recursive_runs.py --ng <path to source repository>

Reads plots/collapse_runs_multi/llama3_instruct/{exp1-2500-inbreed,exp2-2500-cross}/gen*/metrics/
{method5_unified.json, spectral_curvature.json} and keeps, per generation, the per-layer Fisher-Rao step
length (Delta), the parameter-effort terms (E, Eta), the archived curvature and the per-prompt step speeds
of the 128-prompt curvature run, together with a SHA-256 hash of every source file.
"""
import argparse, hashlib, json, os, re

ARMS = {'self-training': 'exp1-2500-inbreed', 'control': 'exp2-2500-cross'}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ng', default=os.path.join('..', 'source_repository'))
    a = ap.parse_args()
    base = os.path.join(a.ng, 'plots', 'collapse_runs_multi', 'llama3_instruct')
    out = dict(model='meta-llama/Meta-Llama-3-8B-Instruct', source='source_repository/plots/collapse_runs_multi/llama3_instruct', arms={})
    for arm, d in ARMS.items():
        gens = sorted(int(g[3:]) for g in os.listdir(os.path.join(base, d)) if re.fullmatch(r'gen\d+', g))
        G = {}
        for g in gens:
            mp = os.path.join(base, d, f'gen{g}', 'metrics', 'method5_unified.json')
            sp = os.path.join(base, d, f'gen{g}', 'metrics', 'spectral_curvature.json')
            m, sc = json.load(open(mp)), json.load(open(sp))
            G[g] = dict(delta=m['Delta'], E=m['E'], eta=m['Eta'], fr_total=m['FR_total'], n_tokens=m['n_tokens'], e_examples=m['E_examples'],
                        kappa_legacy=sc['curvature_mean'], speeds=[p['speeds'] for p in sc['per_prompt']],
                        sha256={'method5_unified.json': sha(mp), 'spectral_curvature.json': sha(sp)})
        out['arms'][arm] = dict(directory=d, generations=G)
    json.dump(out, open(os.path.join('results', 'recursive_runs.json'), 'w'))
    print('wrote results/recursive_runs.json:', {k: sorted(v['generations']) for k, v in out['arms'].items()})


if __name__ == '__main__':
    main()
