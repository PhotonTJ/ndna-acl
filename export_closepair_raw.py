"""Export the unscaled profiles of the close sibling pair (Llama-3 8B, R1-Distill-Llama 8B) to results/closepair_raw.json.

    python export_closepair_raw.py --ng <path to source repository>

Reads plots/method_5_generic/{ag-news,automathtext,stanford_plato}/method5_*.npz and keeps, per model and dataset, the
per-layer Fisher-Rao step length (Delta), the recorded field norms (Vnorm, belief_norms), the alignment cosine (Alpha),
the archived legacy curvature, the sample counts and a SHA-256 hash of every source file. The experiment dashboards
of the same runs store Delta rescaled into [0.005, 1]; these files keep its amplitude.
"""
import argparse, glob, hashlib, json, os
import numpy as np

DATASETS = ['ag-news', 'automathtext', 'stanford_plato']
MODELS = {'meta-llama/Meta-Llama-3-8B': 'Llama-3 8B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': 'R1-Distill-Llama 8B'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ng', default=os.path.join('..', 'source_repository'))
    a = ap.parse_args()
    out = dict(source='source_repository/plots/method_5_generic/<dataset>/method5_*.npz', runs=[])
    for ds in DATASETS:
        for f in sorted(glob.glob(os.path.join(a.ng, 'plots', 'method_5_generic', ds, 'method5_*.npz'))):
            z = np.load(f, allow_pickle=False)
            m = str(z['model'][0])
            out['runs'].append(dict(model=MODELS[m], hf_id=m, dataset=ds, file=os.path.basename(f),
                                    sha256=hashlib.sha256(open(f, 'rb').read()).hexdigest(),
                                    n_examples=int(z['n_examples'][0]), n_tokens=int(z['n_tokens'][0]),
                                    tokens_per_ex=int(z['tokens_per_ex'][0]), max_len=int(z['max_len'][0]), tau=float(z['tau'][0]),
                                    delta=z['Delta'].tolist(), vnorm=z['Vnorm'].astype(float).tolist(),
                                    belief_norms=z['belief_norms'].tolist(), alpha=z['Alpha'].astype(float).tolist(),
                                    kappa_legacy=z['kappa'].tolist(), mean_total_fr=float(z['mean_total_fr'][0])))
    json.dump(out, open(os.path.join('results', 'closepair_raw.json'), 'w'))
    print('wrote results/closepair_raw.json:', [(r['model'], r['dataset'], r['n_examples']) for r in out['runs']])


if __name__ == '__main__':
    main()
