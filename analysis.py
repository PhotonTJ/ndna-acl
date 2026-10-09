"""Regenerates every derived number, figure and table macro used in the paper.

    python analysis.py --ng <path to source repository>      # first run: extracts profiles from the HTML dashboards
    python analysis.py                                       # later runs: reads results/dashboard_profiles.json

Inputs : results/exp0{3,4,7,8}*.json   (archived experiment exports)
         <source repository>/plots/**  (Plotly dashboards and similarity reports, first run only)
         <source repository>/model_zoo.json
Outputs: results/dashboard_profiles.json, results/derived.json, tables.tex (LaTeX macros), figures/fig_*.png
"""
import argparse, base64, glob, itertools, json, math, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
CH = ['Spectral', 'Thermo', 'Belief']       # stored 'Spectral' values use the original interior-angle implementation
LV = ['Thermo', 'Belief']                    # curvature-free channels used for every dashboard result in the paper
MODELS = ['gemma3_1b', 'llama31_8b', 'qwen3_4b', 'qwen3_8b']
MNAME = {'gemma3_1b': 'Gemma-3 1B', 'llama31_8b': 'Llama-3.1 8B', 'qwen3_4b': 'Qwen-3 4B', 'qwen3_8b': 'Qwen-3 8B'}
# public architecture constants (model cards / config.json); not stored in the archived exports
ARCH = {'gemma3_1b': dict(layers=26, H=1152, V=262144), 'llama31_8b': dict(layers=32, H=4096, V=128256),
        'qwen3_4b': dict(layers=36, H=2560, V=151936), 'qwen3_8b': dict(layers=36, H=4096, V=151936)}
PRETTY = {'EleutherAI_pythia_410m_deduped': 'Pythia-410M', 'Qwen_Qwen1.5_7B': 'Qwen1.5-7B', 'Qwen_Qwen1.5_7B_Chat': 'Qwen1.5-7B-Chat',
          'TinyLlama_TinyLlama_1.1B_Chat_v1.0': 'TinyLlama-1.1B', 'allenai_Llama_3.1_Tulu_3_8B': 'Tulu-3.1-8B', 'deepseek_ai_deepseek_llm_7b_base': 'DeepSeek-7B',
          'deepseek_ai_deepseek_llm_7b_chat': 'DeepSeek-7B-Chat', 'google_gemma_7b': 'Gemma-7B', 'google_gemma_7b_it': 'Gemma-7B-IT', 'lmsys_vicuna_13b_v1.5': 'Vicuna-13B',
          'lmsys_vicuna_7b_v1.5': 'Vicuna-7B', 'meta_llama_Llama_2_7b_hf': 'Llama-2-7B', 'meta_llama_Meta_Llama_3_8B': 'Llama-3-8B',
          'meta_llama_Meta_Llama_3_8B_Instruct': 'Llama-3-8B-Instruct', 'microsoft_Orca_2_13b': 'Orca-2-13B', 'microsoft_Orca_2_7b': 'Orca-2-7B',
          'microsoft_phi_2': 'Phi-2', 'mistralai_Mistral_7B_v0.3': 'Mistral-7B-v0.3', 'mistralai_Mistral_7B_Instruct_v0.3': 'Mistral-7B-Instruct-v0.3', 'mistralai_Mistral_7B_v0.1': 'Mistral-7B-v0.1', 'tiiuae_falcon_7b': 'Falcon-7B', 'gpt2': 'GPT-2'}


def pn(n):
    return PRETTY.get(n, n.replace('_', ' '))


OUT = {}      # everything that goes into derived.json
TEX = []      # table macros


# ----------------------------------------------------------------------------- helpers
def jload(p):
    with open(p, encoding='utf8') as f:
        return json.load(f)


def arr(d, k):
    a = d['arrays'][k]
    return np.array(a['data'] if isinstance(a, dict) and 'data' in a else a, dtype=float)


def macro(name, body):
    TEX.append('\\newcommand{\\%s}{%%\n%s}\n' % (name, body))


def fmt(x, n=3):
    return ('%.' + str(n) + 'f') % x


def sci(x):
    if x == 0:
        return '0'
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return '%.1f{\\times}10^{%d}' % (m, e)


def row(cells):
    return ' & '.join(str(c) for c in cells) + ' \\\\\n'


def dtw(a, b, band=0.2):
    """Banded DTW.

    The optimal path minimises accumulated squared Euclidean cost; the reported
    dissimilarity is that cost divided by the path length (aligned mean squared
    cost, the definition used by the archived exports and throughout the paper).
    Warp is mean displacement on relative depth, comparable across unequal depths.
    """
    a = np.asarray(a, float).reshape(len(a), -1)
    b = np.asarray(b, float).reshape(len(b), -1)
    n, m = len(a), len(b)
    C = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        lo = max(1, int(math.floor((i / n - band) * m)))
        hi = min(m, int(math.ceil((i / n + band) * m)))
        for j in range(lo, hi + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, [(n, m)]
    while (i, j) != (1, 1):
        i, j = min([(D[i - 1, j - 1], (i - 1, j - 1)), (D[i - 1, j], (i - 1, j)), (D[i, j - 1], (i, j - 1))], key=lambda x: x[0])[1]
        path.append((i, j))
    cost = D[n, m] / len(path)
    rel = [abs((p - 1) / max(n - 1, 1) - (q - 1) / max(m - 1, 1)) for p, q in path]
    return cost, float(np.mean(rel))


def dmat(curves, band=0.2):
    k = len(curves)
    M, W = np.zeros((k, k)), np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d, w = dtw(curves[i], curves[j], band)
            M[i, j] = M[j, i] = d
            W[i, j] = W[j, i] = w
    return M, W


def anova_stats(M, labels, power=2):
    """One-way PERMANOVA from a dissimilarity matrix: (R2, pseudo-F, omega2).

    The sums of squares are formed from M ** power.  The paper's default (power=2) squares the
    aligned mean squared DTW cost again, i.e. treats that cost as the distance; power=1 uses the
    cost itself as the squared distance (the square root of the cost as the distance).
    """
    labels = np.asarray(labels)
    N = len(labels)
    k = len(set(labels.tolist()))
    D2 = M ** power
    sst = D2[np.triu_indices(N, 1)].sum() / N
    ssw = 0.0
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        n = len(idx)
        if n > 1:
            ssw += D2[np.ix_(idx, idx)][np.triu_indices(n, 1)].sum() / n
    ssb = sst - ssw
    msw = ssw / (N - k)
    F = (ssb / (k - 1)) / msw
    om = (ssb - (k - 1) * msw) / (sst + msw)
    return ssb / sst, F, om


def perm_p(M, labels, stat_idx, nperm, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    obs = anova_stats(M, labels)[stat_idx]
    c = sum(anova_stats(M, rng.permutation(labels))[stat_idx] >= obs - 1e-12 for _ in range(nperm))
    return (c + 1) / (nperm + 1)


def nn_lineage(M, groups, nperm=3000, seed=0):
    """Leave-one-out nearest neighbour shares the lineage label (models with >=1 lineage mate only)."""
    g = np.array(groups)
    N = len(g)
    idx = [i for i in range(N) if (g == g[i]).sum() > 1]

    def acc(gl):
        c = 0
        for i in idx:
            d = M[i].copy()
            d[i] = np.inf
            d[np.isnan(d)] = np.inf
            c += gl[i] == gl[int(np.argmin(d))]
        return c
    obs = acc(g)
    rng = np.random.default_rng(seed)
    cnt = sum(acc(rng.permutation(g)) >= obs for _ in range(nperm))
    same, diff = [], []
    for i in range(N):
        for j in range(i + 1, N):
            (same if g[i] == g[j] else diff).append(M[i, j])
    auc = float(np.mean([(s < d) + 0.5 * (s == d) for s in same for d in diff]))
    chance = sum((g == g[i]).sum() - 1 for i in idx) / (len(idx) * (N - 1))
    rr = []
    for i in idx:
        d = M[i].copy()
        d[i] = np.inf
        order = np.argsort(d)
        for r, j in enumerate(order, 1):
            if g[j] == g[i]:
                rr.append(1 / r)
                break
    return dict(hit=int(obs), k=len(idx), p=(cnt + 1) / (nperm + 1), auc=auc, chance=float(chance), mrr=float(np.mean(rr)))


# ----------------------------------------------------------------------------- dashboard extraction (first run)
def _decode(o):
    if isinstance(o, dict):
        if 'bdata' in o and 'dtype' in o:
            dt = {'i1': 'int8', 'u1': 'uint8', 'i2': '<i2', 'u2': '<u2', 'i4': '<i4', 'u4': '<u4', 'f4': '<f4', 'f8': '<f8', 'i8': '<i8'}[o['dtype']]
            return np.frombuffer(base64.b64decode(o['bdata']), dtype=dt).tolist()
        return {k: _decode(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_decode(x) for x in o]
    return o


def _plotly(path):
    s = open(path, encoding='utf8', errors='ignore').read()
    figs = []
    for m in re.finditer(r'Plotly\.newPlot\(\s*"[^"]+",\s*', s):
        data, _ = json.JSONDecoder().raw_decode(s[m.end():])
        figs.append(_decode(data))
    return s, figs


def extract_profiles(ng):
    prof = {}
    for f in sorted(glob.glob(os.path.join(ng, 'plots', '**', '*_dashboard.html'), recursive=True)):
        key = os.path.relpath(f, os.path.join(ng, 'plots')).replace('\\', '/')
        s, figs = _plotly(f)
        t = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', re.sub(r'<script.*?</script>', '', s, flags=re.S)))
        m = re.search(r'Total nDNA Scalar Score:\s*([0-9.eE+-]+)', t)
        d = {}
        for fig in figs:
            for tr in fig:
                if tr.get('name') in CH:
                    d[tr['name']] = [float(v) for v in tr['y']]
        if len(d) == 3:
            d['score'] = float(m.group(1)) if m else None
            prof[key] = d
    return prof


def extract_reports(ng):
    out = {}
    for f in sorted(glob.glob(os.path.join(ng, 'plots', '**', '*similarity_report.html'), recursive=True)):
        key = os.path.relpath(f, os.path.join(ng, 'plots')).replace('\\', '/')
        s = open(f, encoding='utf8', errors='ignore').read()
        sec = s[s.find('3) Pairwise'):]
        tabs = []
        for m in re.finditer(r'<table[^>]*>(.*?)</table>', sec, flags=re.S):
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', m.group(1), flags=re.S)
            names, mat = [], []
            for r in rows[1:]:
                cells = [re.sub('<.*?>', '', c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, flags=re.S)]
                names.append(cells[0])
                vals = []
                for c in cells[1:]:
                    try:
                        vals.append(float(c))
                    except ValueError:
                        vals.append(float('nan'))
                mat.append(vals)
            tabs.append(dict(names=names, mat=mat))
        out[key] = tabs
    return out


# ----------------------------------------------------------------------------- loading
def load_all(ng):
    p = os.path.join(RES, 'dashboard_profiles.json')
    q = os.path.join(RES, 'squad_report_matrices.json')
    if not os.path.exists(p):
        if not ng:
            sys.exit('first run needs --ng <source repository path>')
        prof = extract_profiles(ng)
        json.dump(prof, open(p, 'w'))
        rep = extract_reports(ng)
        keep = {k: v[:28] for k, v in rep.items() if 'method_5_generic/squad/' in k}
        json.dump(keep, open(q, 'w'))
        zoo = os.path.join(ng, 'model_zoo.json')
        if os.path.exists(zoo):
            json.dump(jload(zoo), open(os.path.join(RES, 'model_zoo.json'), 'w'), indent=1)
    return jload(p), jload(q)


def curve(d, chs=CH):
    return np.stack([np.array(d[c], float) for c in chs], 1)


def short(k):
    k = k.split('/')[-1]
    k = re.sub(r'_dashboard\.html$', '', k)
    return re.sub(r'^(method5_squad_|squad__method5_|method5_)', '', k)


# ----------------------------------------------------------------------------- archived experiment exports
def exports():
    E = {}
    for m in MODELS:
        E[m] = dict(
            e03=jload(f'{RES}/exp03_behavior_vs_geometry__{m}__seed0.data.json'),
            e04=jload(f'{RES}/exp04_task_stability__{m}__seed0.data.json'),
            e07=jload(f'{RES}/exp07_patching__{m}__seed0.data.json'),
            e08=jload(f'{RES}/exp08_absolute_triad__{m}__seed0.data.json'),
            c04=jload(f'{RES}/exp04_task_stability__{m}__seed0.json'),
            c08=jload(f'{RES}/exp08_absolute_triad__{m}__seed0.json'))
    return E


def part_exports(E):
    # ---- cost
    rows = ''
    for m in MODELS:
        a = ARCH[m]
        nodes = len(arr(E[m]['e08'], 'belief'))
        N = 256 * 8
        fl = 2 * nodes * N * a['H'] * a['V']
        nv = N * a['V'] * 4
        hs = nodes * N * a['H'] * 2
        wall = float(arr(E[m]['e04'], 'wall_seconds'))
        blocks = len(arr(E[m]['e04'], 'warp_extent')) if False else arr(E[m]['e04'], 'joint_matrix').shape[0]
        rows += row([MNAME[m], nodes, a['H'], '%dk' % round(a['V'] / 1000), N, fmt(fl / 1e12, 1), fmt(nv / 1e9, 2), fmt(3 * nv / 1e9, 2),
                     fmt(nodes * nv / 1e9, 1), fmt(hs / 1e6, 0), fmt(wall, 0), blocks, fmt(wall / blocks, 1)])
    macro('tabCost', rows)

    # ---- stability
    rows = ''
    st = {}
    for m in MODELS:
        d = E[m]['e04']['arrays']
        td = arr(E[m]['e04'], 'task_dtw')
        iu = np.triu_indices(10, 1)
        v = td[iu]
        # one convention everywhere: the 0.9 quantile of the 45 unique task pairs (the stored value used all 90 ordered entries)
        th = float(np.quantile(v, 0.9))
        st_stored = float(arr(E[m]['e04'], 'theta'))
        vn = [str(x) for x in d['variant_names']['data']]
        vd = arr(E[m]['e04'], 'variant_dtw_from_base')
        vv = dict(zip(vn, vd))
        sib = [k for k in vn if k.startswith('sibling')]
        sibtxt = ', '.join(fmt(vv[k], 4) for k in sib)
        st[m] = dict(min=float(v.min()), med=float(np.median(v)), mean=float(v.mean()), max=float(v.max()), theta=th, theta_stored=st_stored, q4=float(vv['quant_4bit']),
                     pr=float(vv['prune_40']), sib={k: float(vv[k]) for k in sib}, ratio_max_med=float(v.max() / np.median(v)))
        rows += row([MNAME[m], fmt(v.min(), 4), fmt(np.median(v), 4), fmt(v.mean(), 4), fmt(v.max(), 4), fmt(th, 4),
                     fmt(vv['quant_4bit'], 4), '\\textbf{%s}' % fmt(vv['prune_40'], 4), sibtxt,
                     fmt(vv['prune_40'] / th, 1) + '$\\times$'])
    OUT['stability'] = st
    macro('tabStability', rows)

    # ---- per-task mean DTW
    names = [str(x) for x in E[MODELS[0]]['e04']['arrays']['task_names']['data']]
    rows = ''
    mean_by = {m: arr(E[m]['e04'], 'task_dtw').sum(1) / 9 for m in MODELS}
    task_pretty = {'ai2_arc': 'ARC-Challenge', 'hellaswag': 'HellaSwag', 'winogrande': 'WinoGrande',
                   'mnli': 'MNLI', 'qqp': 'QQP', 'squad_v2': 'SQuAD v2', 'cnn_dailymail': 'CNN/DailyMail',
                   'common_gen': 'CommonGen', 'wmt16': 'WMT16', 'imdb': 'IMDb'}
    for i, t in enumerate(names):
        rows += row([task_pretty.get(t, t.replace('_', '\\_'))] + [fmt(mean_by[m][i], 4) for m in MODELS])
    macro('tabTaskMean', rows)
    OUT['task_outlier'] = {m: names[int(np.argmax(mean_by[m]))] for m in MODELS}

    # ---- behaviour vs geometry (all variants)
    rows = ''
    lab_pretty = {'quant_4bit': 'quant-4bit', 'quant_2bit': 'quant-2bit', 'prune_30': 'prune-30\\%', 'prune_60': 'prune-60\\%',
                  'sibling_instruct': 'instruct sibling', 'sibling_think': 'think sibling', 'sibling_base': 'base sibling'}
    e3 = {}
    for m in MODELS:
        d = E[m]['e03']['arrays']
        labs = [str(x) for x in d['labels']['data']]
        dp, dd, ww = arr(E[m]['e03'], 'delta_ppl'), arr(E[m]['e03'], 'dtw_from_base'), arr(E[m]['e03'], 'warp_from_base')
        base = float(arr(E[m]['e03'], 'base_ppl'))
        e3[m] = dict(base=base, rows=[])
        for i, l in enumerate(labs):
            ratio = (base + dp[i]) / base
            rt = ('%.2f' % ratio) if ratio < 100 else sci(ratio)
            rows += row([MNAME[m] if i == 0 else '', lab_pretty.get(l, l), fmt(base, 2) if i == 0 else '', ('$%s$' % rt) if ratio >= 100 else rt,
                         fmt(dd[i], 4), fmt(ww[i], 2)])
            e3[m]['rows'].append(dict(label=l, ratio=float(ratio), dtw=float(dd[i]), warp=float(ww[i])))
        rows += '\\midrule\n' if m != MODELS[-1] else ''
    OUT['e03'] = e3
    macro('tabBehaviour', rows)

    # ---- patching raw vs partial
    from scipy.stats import rankdata, spearmanr, t as tdist

    def partial(x, y, z):
        rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

        def resid(a, c):
            c = c - c.mean()
            a = a - a.mean()
            return a - (c @ a) / (c @ c) * c
        e1, e2 = resid(rx, rz), resid(ry, rz)
        den = math.sqrt((e1 @ e1) * (e2 @ e2))
        if den < 1e-12:
            return float('nan'), float('nan')
        r = (e1 @ e2) / den
        n = len(x)
        tt = r * math.sqrt((n - 2) / (1 - r * r))
        return float(r), float(2 * (1 - tdist.cdf(abs(tt), n - 2)))
    pt = {}
    for m in MODELS:
        e = E[m]['e07']
        lay, pe, L, B, T = (arr(e, k) for k in ['layer', 'patch_effect', 'length', 'belief', 'tail_length'])
        raw = [spearmanr(v, pe)[0] for v in (L, B, T)]
        par = [partial(v, pe, lay) for v in (L, B, T)]
        pt[m] = dict(raw=[float(x) for x in raw], partial=par)
    OUT['patching'] = pt
    rows_clear = ''
    for m in MODELS:
        v = pt[m]
        rows_clear += row([MNAME[m], fmt(v['raw'][0], 3), '%s ($p{=}%s$)' % (fmt(v['partial'][0][0], 3), fmt(v['partial'][0][1], 2)),
                           fmt(v['raw'][1], 3), '%s ($p{=}%s$)' % (fmt(v['partial'][1][0], 3), fmt(v['partial'][1][1], 2)), fmt(v['raw'][2], 3)])
    macro('tabPatchNarrative', rows_clear)

    # ---- absolute profiles
    rows = ''
    ab = {}
    for m in MODELS:
        e = E[m]['e08']
        ln, bl, k = arr(e, 'length'), arr(e, 'belief'), arr(e, 'kappa')
        nv = arr(e, 'n_valid_curvature')
        th = [t.sum() / ln.sum() * 100 for t in np.array_split(ln, 3)]
        cd = arr(e, 'commitment_depth_frac')
        cfg = E[m]['c08']['config']
        ab[m] = dict(thirds=[float(x) for x in th], total=float(ln.sum()), vbar=float(bl.mean()), nvalid_min=int(nv.min()), prompts=cfg['n_prompts'])
        rows += row([MNAME[m], cfg['n_prompts'], fmt(ln.sum(), 2), fmt(bl.mean(), 4), ' / '.join(fmt(x, 1) for x in th),
                     '%d' % nv.min(), fmt(cd[2], 2), fmt(cd[4], 2)])
    OUT['abs'] = ab
    macro('tabAbs', rows)
    macro('tabAbsNarrative', ''.join(row([MNAME[m], fmt(ab[m]['total'], 2), fmt(ab[m]['vbar'], 4),
                                          ' / '.join(fmt(x, 1) for x in ab[m]['thirds'])]) for m in MODELS))

    # ---- original curvature-channel diagnostics (interior-angle score is mostly an inverse step size)
    rows = ''
    lg = {}
    for m in MODELS:
        e = E[m]['e08']
        ln, k = arr(e, 'length'), arr(e, 'kappa')
        ab_ = (ln[:-1] + ln[1:]) / 2
        r_inv = float(np.corrcoef(k, math.pi / ab_)[0, 1])
        r_chord = float(arr(e, 'kappa_estimator_pearson'))
        lg[m] = dict(mean=float(k.mean()), lo=float(k.min()), hi=float(k.max()), r_chord=r_chord, r_inv=r_inv)
        rows += row([MNAME[m], fmt(k.mean(), 3), '[%s, %s]' % (fmt(k.min(), 2), fmt(k.max(), 2)), fmt(r_chord, 3), fmt(r_inv, 3)])
    OUT['legacy_kappa'] = lg
    macro('tabLegacy', rows)

    # ---- runtimes of the archived experiments
    rows = ''
    for m in MODELS:
        cells = [MNAME[m]]
        for key in ('e03', 'e04', 'e07', 'e08'):
            cells.append(fmt(float(arr(E[m][key], 'wall_seconds')), 0))
        rows += row(cells)
    macro('tabRuntime', rows)

    # ---- sample depth 32 -> 256 (exports) ; breadth theta_b from stored matrices
    rows = ''
    sd = {}
    for m in MODELS:
        l32 = arr(E[m]['e08'], 'length')
        l256 = arr(E[m]['e07'], 'length')
        r = float(np.corrcoef(l32, l256)[0, 1])
        rel = float(np.abs(l32 - l256).sum() / np.abs(l256).sum())
        sd[m] = dict(r=r, rel=rel, tot32=float(l32.sum()), tot256=float(l256.sum()))
        rows += row([MNAME[m], fmt(l32.sum(), 2), fmt(l256.sum(), 2), fmt(r, 3), fmt(100 * rel, 1) + '\\%'])
    OUT['depth_exports'] = sd
    macro('tabDepthExp', rows)

    rows = ''
    br = {}
    for b in range(3, 11):
        cells = [b]
        for m in MODELS:
            td = arr(E[m]['e04'], 'task_dtw')
            th10 = OUT['stability'][m]['theta']
            vn = [str(x) for x in E[m]['e04']['arrays']['variant_names']['data']]
            vd = dict(zip(vn, arr(E[m]['e04'], 'variant_dtw_from_base')))
            ths = np.array([np.quantile([td[i, j] for i, j in itertools.combinations(S, 2)], 0.9) for S in itertools.combinations(range(10), b)])
            margin = vd['prune_40'] / ths.max()
            sibs = [vd[k] for k in vd if k.startswith('sibling')]
            sibfrac = max(float((s > ths).mean()) for s in sibs)
            br.setdefault(m, {})[b] = dict(med=float(np.median(ths) / th10), lo=float(ths.min() / th10), hi=float(ths.max() / th10), margin=float(margin), sib=sibfrac,
                                           q4=float((vd['quant_4bit'] > ths).mean()))
            cells.append('%s' % fmt(np.median(ths) / th10, 2))
            cells.append('%s' % fmt(margin, 1))
        rows += row(cells)
    OUT['breadth_theta'] = br
    macro('tabBreadthTheta', rows)
    rows = ''
    for b in [3, 5, 7, 9]:
        rows += row([b] + [fmt(100 * br[m][b]['sib'], 0) + '\\%' for m in MODELS])
    macro('tabBreadthSib', rows)


def part_dash(P, R, ng):
    # ============ sample depth n=1000 vs n=2500 =========================================
    g1 = {short(k): v for k, v in P.items() if k.startswith('1000/')}
    g2 = {short(k): v for k, v in P.items() if k.startswith('2500/')}
    common = sorted(n for n in g1 if n in g2)
    # Restrict candidates to checkpoints represented at both sample sizes.
    names2 = common.copy()
    # identity under a change of sample size, for three channel sets; the per-model table uses length only
    sets = {}
    for lab, chs in (('$\\mathcal L$', ['Thermo']), ('$\\mathcal L,\\lVert\\widetilde{\\mathbf v}\\rVert$', LV), ('original angle, $\\mathcal L,\\lVert\\widetilde{\\mathbf v}\\rVert$', CH)):
        top, mg_ = 0, []
        for n in common:
            a = curve(g1[n], chs)
            ds = {m: dtw(a, curve(g2[m], chs))[0] for m in names2 if abs(len(a) - len(g2[m]['Thermo'])) <= 0.5 * max(len(a), len(g2[m]['Thermo']))}
            o = min(v for m, v in ds.items() if m != n)
            top += ds[n] < o
            mg_.append(o / ds[n])
        sets[lab] = dict(top1=int(top), n=len(common), med=float(np.median(mg_)), mn=float(min(mg_)))
    OUT['sample_depth_sets'] = sets
    macro('tabDepthSets', ''.join(row([k, '%d/%d' % (v['top1'], v['n']), fmt(v['med'], 1) + '$\\times$', fmt(v['mn'], 2) + '$\\times$']) for k, v in sets.items()))
    c1 = [curve(g1[n], ['Thermo']) for n in common]
    c2all = [curve(g2[n], ['Thermo']) for n in names2]
    cross = np.full((len(common), len(names2)), np.inf)
    for i in range(len(common)):
        for j in range(len(names2)):
            if abs(len(c1[i]) - len(c2all[j])) <= 0.5 * max(len(c1[i]), len(c2all[j])):
                cross[i, j] = dtw(c1[i], c2all[j])[0]
    rows, tex, top1 = [], '', 0
    for i, n in enumerate(common):
        js = names2.index(n)
        order = np.argsort(cross[i])
        rank = int(np.where(order == js)[0][0]) + 1
        d_self = cross[i, js]
        oth = [(cross[i, j], names2[j]) for j in range(len(names2)) if j != js]
        d_o, nn = min(oth)
        r = [float(np.corrcoef(g1[n][c], g2[n][c])[0, 1]) for c in CH]
        mx = [float(np.abs(np.array(g1[n][c]) - np.array(g2[n][c])).max()) for c in CH]
        sc1, sc2 = g1[n]['score'], g2[n]['score']
        rows.append(dict(model=n, L=len(c1[i]), r=r, maxdiff=mx, d_self=float(d_self), nn=nn, d_other=float(d_o), margin=float(d_o / d_self), rank=rank, s1=sc1, s2=sc2))
        top1 += rank == 1
    OUT['sample_depth'] = dict(rows=rows, top1=int(top1), n=len(common), median_margin=float(np.median([r['margin'] for r in rows])),
                               scalar_ratio=[float(r['s1'] / r['s2']) for r in rows])
    for r in rows:
        fdist = lambda x: ('$%s$' % sci(x)) if 0 < x < 1e-3 else fmt(x, 3)
        tex += row([pn(r['model']), r['L'], fmt(r['r'][0], 3), fmt(r['r'][1], 3), fmt(r['r'][2], 3), fdist(r['d_self']),
                    pn(r['nn']), fdist(r['d_other']), fmt(r['margin'], 1) + '$\\times$', '%.2f' % (r['s1'] / r['s2'])])
    macro('tabDepthDash', tex)
    sr = np.array(OUT['sample_depth']['scalar_ratio'])
    from scipy.stats import spearmanr
    OUT['sample_depth']['scalar_spearman'] = float(spearmanr([r['s1'] for r in rows], [r['s2'] for r in rows])[0])
    OUT['sample_depth']['scalar_ratio_range'] = [float(sr.min()), float(sr.max()), float(np.median(sr))]
    OUT['sample_depth']['channel_r_min'] = [float(min(r['r'][k] for r in rows)) for k in range(3)]
    OUT['sample_depth']['channel_maxdiff_med'] = [float(np.median([r['maxdiff'][k] for r in rows])) for k in range(3)]
    OUT['_cross_1000_2500'] = (common, names2, cross.tolist())

    # ============ lineage =================================================================
    def lin(n):
        n = n.lower()
        if 'deepseek_llm' in n or ('deepseek' in n and 'r1' not in n):
            return 'DeepSeek-LLM'
        if any(s in n for s in ['llama_2', 'vicuna', 'orca']):
            return 'Llama-2'
        if any(s in n for s in ['tulu', 'meta_llama_3', 'r1_distill']):
            return 'Llama-3'
        if 'qwen1.5' in n:
            return 'Qwen-1.5'
        if 'qwen3' in n:
            return 'Qwen-3'
        if 'gemma_3' in n:
            return 'Gemma-3'
        if 'gemma' in n:
            return 'Gemma'
        if 'mistral' in n:
            return 'Mistral'
        if 'olmo' in n:
            return 'Olmo-3'
        return n
    L = {}
    rep = R['method_5_generic/squad/LLM_curve_similarity_report.html']
    rn = rep[0]['names']
    g19 = {short(k): v for k, v in P.items() if k.startswith('method_5_generic/squad/') and 'method5_' + short(k) in rn}
    for tag, g in (('n=2500', g2), ('n=1000', g1), ('19-model set', g19)):
        names = [n for n in sorted(g) if len(g[n]['Spectral']) >= 15]
        groups = [lin(n) for n in names]
        for chs, lab in ((LV, 'LV'), (['Thermo'], 'L'), (CH, '3ch')):
            M, _ = dmat([curve(g[n], chs) for n in names])
            L[f'{tag} {lab}'] = dict(N=len(names), **nn_lineage(M, groups))
            if tag == 'n=2500' and lab == 'LV':
                nnl = []
                for i, n in enumerate(names):
                    d = M[i].copy()
                    d[i] = np.inf
                    j = int(np.argmin(d))
                    nnl.append((n, groups[i], names[j], float(d[j]), groups[i] == groups[j]))
                OUT['nn_list'] = nnl
                OUT['_M2500'] = (names, groups, M.tolist())
    # squad report (19 models): DTW from report matrices (32 points, original interior-angle score), 7 lineage groups
    rg = [lin(n.replace('method5_', '')) for n in rn]
    L['squad-19 report DTW 3ch'] = dict(N=len(rn), **nn_lineage(np.array(rep[19]['mat']), rg))
    OUT['lineage'] = L

    def lrow(k, v, lab):
        return row([lab, v['N'], '%d/%d' % (v['hit'], v['k']), fmt(v['chance'], 2), ('$<$0.001' if v['p'] < 0.001 else fmt(v['p'], 3)), fmt(v['auc'], 2), fmt(v['mrr'], 2)])
    rows = ''
    for t, tl in (('n=2500', '$n{=}2500$'), ('n=1000', '$n{=}1000$'), ('19-model set', '19-model set')):
        rows += lrow(t, L[f'{t} LV'], tl)
    macro('tabLineage', rows)
    rows = ''
    for t, tl in (('n=2500', '$n{=}2500$'), ('n=1000', '$n{=}1000$'), ('19-model set', '19-model set')):
        for lab, ll in (('LV', '$\\mathcal L,\\lVert\\widetilde{\\mathbf v}\\rVert$'), ('L', '$\\mathcal L$ only'), ('3ch', 'three, original angle')):
            v = L[f'{t} {lab}']
            rows += row([tl, ll, '%d/%d' % (v['hit'], v['k']), fmt(v['auc'], 2), fmt(v['mrr'], 2)])
    v = L['squad-19 report DTW 3ch']
    rows += row(['19-model report', 'three, original angle, 32 points', '%d/%d' % (v['hit'], v['k']), fmt(v['auc'], 2), fmt(v['mrr'], 2)])
    macro('tabLineageAll', rows)
    tex = ''
    eligible = {'DeepSeek-LLM', 'Llama-2', 'Llama-3', 'Qwen-1.5', 'Gemma'}
    for (n, gl, nn, d, ok) in OUT['nn_list']:
        if gl in eligible:
            tex += row([pn(n), gl, pn(nn), fmt(d, 3)])
    macro('tabNN', tex)

    # ============ alternative metrics (squad report, 7 metrics) ===========================
    meth = ['PCM', 'Area between curves', 'Curve length', 'Discrete Fr\\\'echet', 'DTW', 'MAE (resampled)', 'MSE (resampled)']
    am = {}
    rows = ''
    for mi, m in enumerate(meth):
        r2 = nn_lineage(np.array(rep[mi * 4 + 0]['mat']), rg, nperm=2000)
        r3 = nn_lineage(np.array(rep[mi * 4 + 3]['mat']), rg, nperm=2000) if mi >= 3 else None
        am[m] = dict(two=r2, three=r3)
        rows += row([m, '%d/%d' % (r2['hit'], r2['k']), fmt(r2['auc'], 3), fmt(r2['mrr'], 3)])
    OUT['alt_metrics'] = am
    macro('tabMetrics', rows)

    # ============ decoder vs dataset (10 tasks x 4 models) ================================
    tasks = ['ai2_arc', 'hellaswag', 'winogrande', 'mnli', 'qqp', 'squad_v2', 'cnn_dailymail', 'common_gen', 'wmt16', 'imdb']
    dirs = {'Gemma3_base': 'Gemma-3 1B PT', 'Gemma3_instruct': 'Gemma-3 1B IT', 'Llama_3_8B': 'Llama-3 8B', 'Qwen3_4B': 'Qwen-3 4B'}
    raw, lm, lt = [], [], []
    for d, mn in dirs.items():
        for t in tasks:
            k = [k for k in P if f'/10_tasks/{d}/' in k and os.path.basename(k).startswith(t + '__')][0]
            raw.append(P[k])
            lm.append(mn)
            lt.append(t)
    N = len(raw)
    combos = [('$\\mathcal{L}$ only', ['Thermo']), ('$\\lVert\\widetilde{\\mathbf{v}}\\rVert$ only', ['Belief']), ('$\\mathcal{L},\\lVert\\widetilde{\\mathbf{v}}\\rVert$', LV),
              ('original angle only', ['Spectral']), ('original angle, $\\mathcal{L}$', ['Spectral', 'Thermo']), ('all three (original angle)', CH)]
    dv, rows = {}, ''
    mats = {}
    for lab, chs in combos:
        M, _ = dmat([np.stack([r[c] for c in chs], 1) for r in raw])
        mats[lab] = M
        rm, Fm, om = anova_stats(M, lm)
        rt, Ft, ot = anova_stats(M, lt)
        pm, pt_ = perm_p(M, lm, 1, 1000), perm_p(M, lt, 1, 1000)
        nnm = nnt = 0
        for i in range(N):
            d = M[i].copy()
            d[i] = np.inf
            j = int(np.argmin(d))
            nnm += lm[i] == lm[j]
            nnt += lt[i] == lt[j]
        dv[lab] = dict(R2m=rm, w2m=om, pm=pm, R2t=rt, w2t=ot, pt=pt_, nnm=nnm, nnt=nnt)
        rows += row([lab, fmt(rm, 2), fmt(om, 2), ('$<$0.001' if pm < 0.0015 else fmt(pm, 3)), fmt(rt, 2), fmt(ot, 2),
                     ('$<$0.001' if pt_ < 0.0015 else fmt(pt_, 3)), '%d/%d' % (nnm, N), '%d/%d' % (nnt, N)])
    OUT['decoder_vs_task'] = dv
    macro('tabPerm', rows)
    # sensitivity of omega^2 to the PERMANOVA input: squared aligned cost (paper) against the aligned cost itself
    sens, rows_s = {}, ''
    for lab, chs in combos[:3]:
        M = mats[lab]
        a2, a1 = (anova_stats(M, lm, 2)[2], anova_stats(M, lt, 2)[2]), (anova_stats(M, lm, 1)[2], anova_stats(M, lt, 1)[2])
        sens[lab] = dict(squared_cost=a2, cost=a1)
        rows_s += row([lab, fmt(a2[0], 2), fmt(a2[1], 2), fmt(a1[0], 2), fmt(a1[1], 2)])
    OUT['permanova_input_sensitivity'] = sens
    macro('tabPermSens', rows_s)
    OUT['_dec_items'] = (lm, lt, mats['$\\mathcal{L}$ only'].tolist(), mats['$\\lVert\\widetilde{\\mathbf{v}}\\rVert$ only'].tolist())
    OUT['_dec_raw'] = [(lm[i], lt[i], list(raw[i]['Thermo']), list(raw[i]['Belief'])) for i in range(N)]
    OUT['_dec_mats'] = {k: v.tolist() for k, v in mats.items()}

    # Fixed leave-one-family-out evaluation.  Channel scalers and task
    # references are fitted only on the calibration families in each fold.
    families = {
        'reasoning / multiple choice': {'ai2_arc', 'hellaswag', 'winogrande'},
        'classification / matching': {'mnli', 'qqp', 'imdb'},
        'generation': {'squad_v2', 'cnn_dailymail', 'common_gen', 'wmt16'},
    }
    model_order = list(dirs.values())
    by_key = {(lm[i], lt[i]): raw[i] for i in range(N)}
    heldout_rows, heldout = '', {}
    total_hit = total_ref = total_n = 0
    for family, test_tasks in families.items():
        train_tasks = [t for t in tasks if t not in test_tasks]
        # One scaler for the fold, fitted to retained task profiles only.
        vals = [[] for _ in LV]
        for mn in model_order:
            for t in train_tasks:
                for ci, ch in enumerate(LV):
                    vals[ci].extend(by_key[(mn, t)][ch])
        lo = np.array([min(v) for v in vals], float)
        hi = np.array([max(v) for v in vals], float)
        span = np.maximum(hi - lo, 1e-12)

        def folded(mn, t):
            x = np.stack([by_key[(mn, t)][ch] for ch in LV], 1).astype(float)
            return (x - lo) / span

        references = {}
        theta = {}
        for mn in model_order:
            references[mn] = [folded(mn, t) for t in train_tasks]
            ds = [dtw(references[mn][i], references[mn][j])[0]
                  for i in range(len(train_tasks)) for j in range(i + 1, len(train_tasks))]
            theta[mn] = float(np.quantile(ds, 0.9))
        hit = within = count = 0
        for mn in model_order:
            for t in sorted(test_tasks):
                q = folded(mn, t)
                score = {cand: float(np.median([dtw(q, r)[0] for r in references[cand]]))
                         for cand in model_order}
                hit += min(score, key=score.get) == mn
                within += score[mn] <= theta[mn]
                count += 1
        heldout[family] = dict(train=len(train_tasks), test=len(test_tasks), top1=int(hit),
                               within_reference=int(within), n=int(count), theta=theta)
        heldout_rows += row([family, len(train_tasks), len(test_tasks), '%d/%d' % (hit, count),
                             '%d/%d' % (within, count)])
        total_hit += hit
        total_ref += within
        total_n += count
    heldout['all'] = dict(top1=int(total_hit), within_reference=int(total_ref), n=int(total_n))
    heldout_rows += row(['all folds', '', 10, '%d/%d' % (total_hit, total_n),
                         '%d/%d' % (total_ref, total_n)])
    OUT['heldout_task_families'] = heldout
    macro('tabHeldoutTasks', heldout_rows)

    # five-task pair (near siblings)
    g5 = {os.path.basename(k).replace('_dashboard.html', ''): v for k, v in P.items() if '5_tasks' in k}
    items = {('Llama-3 8B', 'ag-news'): 'method5_meta_llama_Meta_Llama_3_8B (ag-news)', ('Llama-3 8B', 'automathtext'): 'method5_meta_llama_Meta_Llama_3_8B (automathtext)',
             ('Llama-3 8B', 'squad'): 'method5_meta_llama_Meta_Llama_3_8B (squad)', ('Llama-3 8B', 'stanford_plato'): 'method5_meta_llama_Meta_Llama_3_8B (stanford_plato)',
             ('R1-Distill-Llama 8B', 'ag-news'): 'method5_deepseek_ai_DeepSeek_R1_Distill_Llama_8B (ag-news)',
             ('R1-Distill-Llama 8B', 'automathtext'): 'method5_deepseek_ai_DeepSeek_R1_Distill_Llama_8B (automathtext)',
             ('R1-Distill-Llama 8B', 'stanford_plato'): 'method5_deepseek_ai_DeepSeek_R1_Distill_Llama_8B (stanford_plato)'}
    keys = list(items)
    rows, f5 = '', {}
    for lab, chs in [('$\\mathcal{L}$ only', ['Thermo']), ('$\\lVert\\widetilde{\\mathbf{v}}\\rVert$ only', ['Belief']), ('$\\mathcal{L},\\lVert\\widetilde{\\mathbf{v}}\\rVert$', LV),
                     ('original angle only', ['Spectral']), ('all three (original angle)', CH)]:
        M, _ = dmat([np.stack([g5[items[k]][c] for c in chs], 1) for k in keys])
        n5 = len(keys)
        sm = np.mean([M[i, j] for i in range(n5) for j in range(i + 1, n5) if keys[i][0] == keys[j][0]])
        sd_ = np.mean([M[i, j] for i in range(n5) for j in range(i + 1, n5) if keys[i][1] == keys[j][1] and keys[i][0] != keys[j][0]])
        nnm = nnd = 0
        for i in range(n5):
            d = M[i].copy()
            d[i] = np.inf
            j = int(np.argmin(d))
            nnm += keys[i][0] == keys[j][0]
            nnd += keys[i][1] == keys[j][1]
        f5[lab] = dict(same_model=float(sm), same_data=float(sd_), nnm=nnm, nnd=nnd)
        rows += row([lab, fmt(sm, 4), fmt(sd_, 4), '%d/%d' % (nnm, n5), '%d/%d' % (nnd, n5)])
    OUT['five_tasks'] = f5
    macro('tabFive', rows)

    # breadth: aggregate of b tasks -> identity of the model
    mn_idx = {mn: [i for i in range(N) if lm[i] == mn] for mn in dirs.values()}
    curves = [np.stack([r[c] for c in LV], 1) for r in raw]
    full = {mn: np.mean([curves[i] for i in idx], 0) for mn, idx in mn_idx.items()}
    names = list(dirs.values())
    res, rows = [], ''
    for b in range(1, 10):
        dist, acc, marg, mx, ho = [], [], [], [], []
        for mn, idx in mn_idx.items():
            dd, aa = [], []
            for S in itertools.combinations(range(10), b):
                agg = np.mean([curves[idx[s]] for s in S], 0)
                d_self = dtw(agg, full[mn])[0]
                oth = [dtw(agg, full[o])[0] for o in names if o != mn and abs(len(full[o]) - len(agg)) <= 0.5 * len(agg)]
                dd.append(d_self)
                aa.append(d_self < min(oth))
                marg.append(min(oth) / d_self)
                # held out: every decoder's reference is built from the 10-b tasks not in S
                comp = [t for t in range(10) if t not in S]
                refs = {o: np.mean([curves[mn_idx[o][t]] for t in comp], 0) for o in names}
                dh = {o: dtw(agg, refs[o])[0] for o in names if o == mn or abs(len(refs[o]) - len(agg)) <= 0.5 * len(agg)}
                ho.append(min(dh, key=dh.get) == mn)
            dist.append(np.median(dd))
            mx.append(np.max(dd))
            acc.append(np.mean(aa))
        res.append(dict(b=b, med=float(np.mean(dist)), mx=float(np.max(mx)), acc=float(np.mean(acc)), margin=float(np.median(marg)), heldout=float(np.mean(ho))))
        rows += row([b, fmt(np.mean(dist), 4), fmt(np.max(mx), 4), fmt(100 * np.mean(acc), 1) + '\\%', fmt(np.median(marg), 1) + '$\\times$', fmt(100 * np.mean(ho), 1) + '\\%'])
    OUT['breadth_id'] = res
    macro('tabBreadthID', rows)

    # ============ merging (Qwen3 LoRA, measured dashboards) ===============================
    gq = {os.path.basename(k).replace('squad__method5_lora_', '').replace('_dashboard.html', ''): v for k, v in P.items() if 'merging_lora_qwen3' in k}
    regs = ['AF', 'AS', 'AU', 'CH', 'EU', 'LA', 'ME', 'NA']
    B = np.stack([curve(gq['finetuned_' + r], LV) for r in regs])

    def rms(a, b):
        return float(np.sqrt(((a - b) ** 2).mean()))
    D = np.array([[rms(B[i], B[j]) for j in range(8)] for i in range(8)])
    pairs = list(itertools.combinations(range(8), 2))
    mrows, rec, fit = [], 0, 0
    for k in sorted(k for k in gq if k.startswith('merged_')):
        a, b = k.split('_')[1:3]
        ia, ib = regs.index(a), regs.index(b)
        c = curve(gq[k], LV)
        dist = [rms(c, B[i]) for i in range(8)]
        nn2 = set(np.argsort(dist)[:2].tolist())
        fits = {}
        for (i, j) in pairs:
            A_, B_, y = B[i].ravel(), B[j].ravel(), c.ravel()
            v = A_ - B_
            t = float(np.clip(((y - B_) @ v) / (v @ v), 0, 1))
            fits[(i, j)] = (float(np.sqrt(((y - (t * A_ + (1 - t) * B_)) ** 2).mean())), t)
        ranked = sorted(fits, key=lambda x: fits[x][0])
        true = (min(ia, ib), max(ia, ib))
        rk = ranked.index(true) + 1
        res_, t = fits[true]
        ta = t if ia == true[0] else 1 - t
        la = np.sqrt(((c - B[ia]) ** 2).sum(1))
        lb = np.sqrt(((c - B[ib]) ** 2).sum(1))
        sg = np.sign(lb - la)
        sg = sg[sg != 0]
        sw = int(np.sum(sg[1:] * sg[:-1] < 0))
        # reference that keeps each child's own counts of the two signs: random reorderings of the same signs
        prng = np.random.default_rng(0)
        perm = np.array([np.sum((lambda s: s[1:] * s[:-1] < 0)(prng.permutation(sg))) for _ in range(2000)])
        sw_exp, sw_p = float(perm.mean()), float((np.sum(perm <= sw) + 1) / (len(perm) + 1))
        mrows.append(dict(child=f'{a}+{b}', rank=rk, nn2=bool(nn2 == {ia, ib}), ta=ta, ratio=res_ / D[ia, ib], dom=(dist[ib] - dist[ia]) / (dist[ia] + dist[ib]), sw=sw,
                          sw_exp=sw_exp, sw_p=sw_p, pd=float(D[ia, ib])))
        rec += nn2 == {ia, ib}
        fit += rk == 1
    OUT['merge'] = dict(rows=mrows, nn2=int(rec), fit1=int(fit), med_rank=float(np.median([r['rank'] for r in mrows])),
                        med_ratio=float(np.median([r['ratio'] for r in mrows])), med_switch=float(np.median([r['sw'] for r in mrows])),
                        med_switch_exp=float(np.median([r['sw_exp'] for r in mrows])), n_sw_sig=int(sum(r['sw_p'] < 0.05 for r in mrows)),
                        n_outside=int(sum(r['ratio'] > 1 for r in mrows)), base_pd=[float(D[np.triu_indices(8, 1)].min()), float(D[np.triu_indices(8, 1)].mean()), float(D[np.triu_indices(8, 1)].max())])
    rows = ''
    for r in mrows:
        rows += row([r['child'], fmt(r['pd'], 4), '%d' % r['rank'], 'yes' if r['nn2'] else 'no', fmt(r['ta'], 2), fmt(r['ratio'], 2), fmt(r['dom'], 2),
                     '%d (%s)' % (r['sw'], fmt(r['sw_exp'], 1))])
    macro('tabMerge', rows)
    summary = row(['True-parent mixture', 'best-fitting pair', f"{fit}/28; median rank {fmt(np.median([r['rank'] for r in mrows]), 0)}"])
    summary += row(['Nearest bases', 'both true parents', f"{rec}/28; chance $1/28$"])
    summary += row(['Mixture residual', 'residual / parent distance', 'median %s' % fmt(np.median([r['ratio'] for r in mrows]), 2)])
    summary += row(['Dominance persistence', 'fewer switches than reordered signs', f"{sum(r['sw_p'] < 0.05 for r in mrows)}/28; unadjusted $p<0.05$"])
    macro('tabMergeSummary', summary)

    # ============ alignment ladder (base / SFT / DPO) ======================================
    def grp(sub):
        return {os.path.basename(k).replace('_dashboard.html', ''): v for k, v in P.items() if f'/{sub}/' in k}
    gl, gh = grp('alignment_litmus'), grp('harmbench_plots')
    rows, al = '', {}
    for fam, (a, b) in {'Llama-3 8B': ('llama', 'Llama'), 'Qwen-3 4B': ('Qwen', 'Qwen')}.items():
        for probe, g, pre in (('Litmus', gl, a), ('HarmBench', gh, b)):
            cb, cs, cd = (curve(g[f'{pre}_{s}'], LV) for s in ('base', 'SFT', 'DPO'))
            d1, _ = dtw(cb, cs)
            d2, _ = dtw(cs, cd)
            d3, _ = dtw(cb, cd)
            al[f'{fam} {probe}'] = dict(base_sft=d1, sft_dpo=d2, base_dpo=d3)
            rows += row([fam, probe, fmt(d1, 4), fmt(d2, 4), fmt(d3, 4)])
    items = []
    lab = []
    for fam, (a, b) in {'Llama': ('llama', 'Llama'), 'Qwen': ('Qwen', 'Qwen')}.items():
        for st in ['base', 'SFT', 'DPO']:
            items.append(curve(gl[f'{a}_{st}'], LV))
            lab.append((fam, st, 'litmus'))
            items.append(curve(gh[f'{b}_{st}'], LV))
            lab.append((fam, st, 'harmbench'))
    M, _ = dmat(items)
    for k, nm in enumerate(['family', 'stage', 'probe']):
        al['R2_' + nm] = float(anova_stats(M, [l[k] for l in lab])[0])
    nnf = nns = nnp = 0
    for i in range(len(items)):
        d = M[i].copy()
        d[i] = np.inf
        j = int(np.argmin(d))
        nnf += lab[i][0] == lab[j][0]
        nns += lab[i][1] == lab[j][1]
        nnp += lab[i][2] == lab[j][2]
    al['nn'] = [nnf, nns, nnp, len(items)]
    OUT['alignment'] = al
    macro('tabAlign', rows)
    macro('tabAlignR', row(['family', fmt(al['R2_family'], 3), '%d/%d' % (nnf, len(items))]) + row(['stage (base, SFT, DPO)', fmt(al['R2_stage'], 3), '%d/%d' % (nns, len(items))]) +
          row(['probe set', fmt(al['R2_probe'], 3), '%d/%d' % (nnp, len(items))]))

    # ============ distillation ============================================================
    gd = grp('distillation_plots')
    dist_rows = ''
    dd = {}
    for lab_, trio in [('Llama-3 8B (30 layers)', ['Llama_3', 'Llama_3.1', 'Llama_3_distilled']), ('Qwen-2 7B (26 layers)', ['Qwen_2', 'Qwen_2.5', 'Qwen_2_distilled'])]:
        c = [curve(gd[n], LV) for n in trio]
        d_t = dtw(c[0], c[2])[0]
        d_n = dtw(c[1], c[2])[0]
        d_tn = dtw(c[0], c[1])[0]
        dd[lab_] = dict(teacher_student=d_t, successor_student=d_n, teacher_successor=d_tn)
        dist_rows += row([lab_, fmt(d_t, 4), fmt(d_n, 4), fmt(d_tn, 4)])
    OUT['distill'] = dd
    macro('tabDistill', dist_rows)

    # ============ composite scalar tally ==================================================
    from collections import defaultdict
    tally = defaultdict(lambda: [0, 0, []])
    for k, v in P.items():
        if k.startswith('ndna_') or v.get('score') is None:
            continue
        g_ = '/'.join(k.split('/')[:-1]).replace('method_5_generic/', '')
        tally[g_][0] += v['score'] == 0
        tally[g_][1] += 1
        tally[g_][2].append(v['score'])
    z, t_ = 0, 0
    for g_, (zz, nn, sc) in sorted(tally.items()):
        z += zz
        t_ += nn
    OUT['composite'] = dict(zero=int(z), total=int(t_))
    return P


def _pf(p):
    return '$<$0.001' if p < 0.001 else fmt(p, 3)


def _sphere_curve(kfun, total, x0, T0, h=2e-4):
    """Unit-speed curve on the unit 2-sphere with geodesic curvature kfun(s): x' = T, T' = -x + k (x cross T) (RK4)."""
    def f(x, T, s):
        return T, -x + kfun(s) * np.cross(x, T)
    x, T = x0 / np.linalg.norm(x0), T0 - (T0 @ x0) * x0 / (x0 @ x0)
    T = T / np.linalg.norm(T)
    xs = [x.copy()]
    for i in range(int(round(total / h))):
        s = i * h
        k1 = f(x, T, s)
        k2 = f(x + h / 2 * k1[0], T + h / 2 * k1[1], s + h / 2)
        k3 = f(x + h / 2 * k2[0], T + h / 2 * k2[1], s + h / 2)
        k4 = f(x + h * k3[0], T + h * k3[1], s + h)
        x = x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        T = T + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        x /= np.linalg.norm(x)
        T -= (T @ x) * x
        T /= np.linalg.norm(T)
        xs.append(x.copy())
    return np.array(xs), h


def _turn_estimators(U):
    """Corrected exterior turn and original interior-angle score at interior nodes of a sphere path."""
    cs = lambda p, q: np.clip((p * q).sum(-1), -1, 1)
    a, b, c = np.arccos(cs(U[:-2], U[1:-1])), np.arccos(cs(U[1:-1], U[2:])), np.arccos(cs(U[:-2], U[2:]))
    ca = np.clip((np.cos(c) - np.cos(a) * np.cos(b)) / (np.sin(a) * np.sin(b)), -1, 1)
    alpha = np.arccos(ca)
    return (np.pi - alpha) / (a + b), alpha / (a + b), np.pi / (a + b)


def part_offline(E):
    """Checks that need no new model runs: task bootstrap of theta, restricted permutations, a curvature-free task-only
    reference, the behaviour link, and a simulation of the curvature estimators on curves of known curvature."""
    rng = np.random.default_rng(0)
    # (A) bootstrap over tasks for theta and for every decision read against it
    rows, bt = '', {}
    for m in MODELS:
        td = arr(E[m]['e04'], 'task_dtw')
        st = OUT['stability'][m]
        ths = []
        for _ in range(2000):
            S = rng.integers(0, 10, 10)
            vals = [td[S[i], S[j]] for i in range(10) for j in range(i + 1, 10) if S[i] != S[j]]
            ths.append(np.quantile(vals, 0.9))
        ths = np.array(ths)
        lo, hi = (float(x) for x in np.quantile(ths, [0.05, 0.95]))
        sibmax = max(st['sib'].values())
        bt[m] = dict(lo=lo, hi=hi, pr_over_hi=st['pr'] / hi, q4_over_lo=st['q4'] / lo, sib_over_lo=sibmax / lo,
                     share_prune_above=float(np.mean(st['pr'] > ths)), share_q4_below=float(np.mean(st['q4'] <= ths)),
                     share_sib_below=float(np.mean(sibmax <= ths)))
        rows += row([MNAME[m], fmt(st['theta'], 4), '[%s, %s]' % (fmt(lo, 4), fmt(hi, 4)), fmt(st['pr'] / hi, 1) + '$\\times$',
                     fmt(st['q4'] / lo, 2), fmt(sibmax / lo, 2), '%d\\%%' % round(100 * bt[m]['share_sib_below'])])
    OUT['theta_boot'] = bt
    macro('tabBootTheta', rows)

    # (B) permutations restricted to strata of the crossed design (model labels within each task, task labels within each model)
    lm, lt = np.array(OUT['_dec_items'][0]), np.array(OUT['_dec_items'][1])

    def strat_p(M, lab, strata, nperm=2000):
        obs = anova_stats(M, lab)[1]
        c = 0
        for _ in range(nperm):
            pl = lab.copy()
            for s_ in np.unique(strata):
                ii = np.where(strata == s_)[0]
                pl[ii] = lab[rng.permutation(ii)]
            c += anova_stats(M, pl)[1] >= obs - 1e-12
        return (c + 1) / (nperm + 1)
    rows, sp = '', {}
    for lab in ['$\\mathcal{L}$ only', '$\\lVert\\widetilde{\\mathbf{v}}\\rVert$ only', '$\\mathcal{L},\\lVert\\widetilde{\\mathbf{v}}\\rVert$']:
        M = np.array(OUT['_dec_mats'][lab])
        pm, pt_ = strat_p(M, lm, lt), strat_p(M, lt, lm)
        dv = OUT['decoder_vs_task'][lab]
        # size of the two effects on one scale: distance between decoders on the same task over distance between tasks within a decoder
        n_ = len(lm)
        btw = np.median([M[i, j] for i in range(n_) for j in range(i + 1, n_) if lt[i] == lt[j] and lm[i] != lm[j]])
        wtn = np.median([M[i, j] for i in range(n_) for j in range(i + 1, n_) if lm[i] == lm[j]])
        sp[lab] = dict(pm=pm, pt=pt_, ratio=float(btw / wtn))
        rows += row([lab, fmt(dv['w2m'], 2), _pf(dv['pm']), _pf(pm), fmt(dv['w2t'], 2), _pf(dv['pt']), _pf(pt_), fmt(btw / wtn, 1) + '$\\times$'])
    OUT['strat_perm'] = sp
    macro('tabStratPerm', rows)
    primary = ''
    for lab in ['$\\mathcal{L}$ only', '$\\lVert\\widetilde{\\mathbf{v}}\\rVert$ only', '$\\mathcal{L},\\lVert\\widetilde{\\mathbf{v}}\\rVert$']:
        dv, ss = OUT['decoder_vs_task'][lab], sp[lab]
        primary += row([lab, fmt(dv['w2m'], 2), _pf(ss['pm']), fmt(dv['w2t'], 2), _pf(ss['pt']),
                        '%d/%d' % (dv['nnm'], len(lm)), '%d/%d' % (dv['nnt'], len(lm)), fmt(ss['ratio'], 1) + '$\\times$'])
    macro('tabPermPrimary', primary)

    # (C) curvature-free reference from the ten-task dashboards (scaled on the ten tasks of each model only, no variants)
    M = np.array(OUT['_dec_mats']['$\\mathcal{L},\\lVert\\widetilde{\\mathbf{v}}\\rVert$'])
    dt, rows = {}, ''
    for mn in dict.fromkeys(lm.tolist()):
        ii = np.where(lm == mn)[0]
        v = np.array([M[ii[a], ii[b]] for a in range(10) for b in range(a + 1, 10)])
        dt[mn] = dict(med=float(np.median(v)), theta=float(np.quantile(v, 0.9)), mx=float(v.max()))
        rows += row([mn, fmt(dt[mn]['med'], 4), fmt(dt[mn]['theta'], 4), fmt(dt[mn]['mx'], 4)])
    tp = {lt[i]: i for i in np.where(lm == 'Gemma-3 1B PT')[0]}
    ti = {lt[i]: i for i in np.where(lm == 'Gemma-3 1B IT')[0]}
    sib = np.array([M[tp[t], ti[t]] for t in tp])
    dt['sibling'] = dict(med=float(np.median(sib)), lo=float(sib.min()), hi=float(sib.max()),
                         above_pt=int((sib > dt['Gemma-3 1B PT']['theta']).sum()), above_it=int((sib > dt['Gemma-3 1B IT']['theta']).sum()))
    OUT['dash_theta'] = dt
    macro('tabDashTheta', rows)

    # (D) profile distance against behaviour (behaviour block of the exports, separately scaled)
    from scipy.stats import spearmanr
    pts = [(r['ratio'], r['dtw'], r['warp']) for m in MODELS for r in OUT['e03'][m]['rows']]
    lr = np.log10([p[0] for p in pts])
    d, w = np.array([p[1] for p in pts]), np.array([p[2] for p in pts])
    mild = lr < 2
    bl = dict(n=len(pts), n_mild=int(mild.sum()))
    for key, x in (('dtw', d), ('warp', w)):
        r_, p_ = spearmanr(lr, x)
        rm_, pm_ = spearmanr(lr[mild], x[mild])
        bl[key] = dict(rho=float(r_), p=float(p_), rho_mild=float(rm_), p_mild=float(pm_))
    OUT['behav_link'] = bl

    # (E) curvature estimators on curves of known geodesic curvature (no model involved)
    sims, rows = [], ''
    curves = []
    while len(curves) < 40:
        A, B, fr, ph = rng.uniform(1.0, 3.0), rng.uniform(0.3, 1.0), rng.uniform(2.0, 6.0), rng.uniform(0, 2 * np.pi)
        kfun = (lambda s, A=A, B=B, fr=fr, ph=ph: A + B * np.sin(fr * s + ph))
        x0 = np.ones(3) / np.sqrt(3) + rng.normal(0, 0.05, 3)
        T0 = rng.normal(0, 1, 3)
        X, h = _sphere_curve(kfun, 1.0, x0, T0)
        if (X > 0.02).all():                                  # stays inside the positive orthant, so X**2 is an interior distribution
            curves.append((X, h, kfun))
    for L in (16, 32, 64):
        for eta in (0.0, 0.1, 0.3):
            e_rel, r_c, r_l, r_inv = [], [], [], []
            for X, h, kfun in curves:
                step = np.exp(-np.arange(L) / (L / 3.0))      # front-loaded steps, as in the measured length profiles
                sig = np.concatenate([[0], np.cumsum(step / step.sum())]) * (len(X) - 1) * h
                U = X[np.round(sig / h).astype(int)]
                if eta > 0:
                    gap = np.minimum(np.r_[np.inf, np.linalg.norm(np.diff(U, axis=0), axis=1)], np.r_[np.linalg.norm(np.diff(U, axis=0), axis=1), np.inf])
                    nz = rng.normal(0, 1, U.shape)
                    nz -= (nz * U).sum(1, keepdims=True) * U
                    nz /= np.linalg.norm(nz, axis=1, keepdims=True)
                    U = U + eta * gap[:, None] * nz
                    U /= np.linalg.norm(U, axis=1, keepdims=True)
                kc, kl, inv = _turn_estimators(U)
                truth = np.array([kfun(s) for s in sig[1:-1]]) / 2.0   # Fisher-Rao geodesic curvature = spherical / 2
                e_rel.append(np.median(np.abs(kc - truth) / truth))
                r_c.append(np.corrcoef(kc, truth)[0, 1])
                r_l.append(np.corrcoef(kl, truth)[0, 1])
                r_inv.append(np.corrcoef(kl, inv)[0, 1])
            res = dict(L=L, eta=eta, err=float(np.median(e_rel)), r_c=float(np.median(r_c)), r_l=float(np.median(r_l)), r_inv=float(np.median(r_inv)))
            sims.append(res)
            rows += row([L, fmt(eta, 1), fmt(100 * res['err'], 1) + '\\%', fmt(res['r_c'], 2), fmt(res['r_l'], 2), fmt(res['r_inv'], 3)])
    OUT['curv_sim'] = sims
    macro('tabCurvSim', rows)


def part_recursive():
    """Recursive self-training of Llama-3 8B Instruct and its human-data control (results/recursive_runs.json)."""
    from scipy.stats import spearmanr
    R = jload(os.path.join(RES, 'recursive_runs.json'))['arms']
    arms = {k: {int(g): v for g, v in a['generations'].items()} for k, a in R.items()}
    st, ct = arms['self-training'], arms['control']
    L = len(st[0]['delta'])
    th = np.array_split(np.arange(L), 3)
    rng = np.random.default_rng(0)
    gens = sorted(set(st) | set(ct))
    m0 = np.mean(st[0]['delta'])
    rows, res = '', dict(rows=[])
    for g in gens:
        cells = [g]
        rec = dict(gen=g)
        for name, A in (('self', st), ('control', ct)):
            if g in A:
                d = np.array(A[g]['delta'])
                rec[name] = dict(mean=float(d.mean()), thirds=[float(d[t].mean()) for t in th])
                cells += [fmt(d.mean(), 3), ('%+d\\%%' % round(100 * (d.mean() / m0 - 1))).replace('-', '$-$') if g else ''] + [fmt(d[t].mean(), 3) for t in th]
            else:
                cells += ['', '', '', '', '']
        if g in st and g in ct and g > 0:
            a, b = np.array(st[g]['speeds']), np.array(ct[g]['speeds'])
            dp = a.mean(1) - b.mean(1)
            boots = np.array([dp[rng.integers(0, len(dp), len(dp))].mean() for _ in range(4000)])
            lo, hi = np.quantile(boots, [0.025, 0.975])
            rec['paired'] = dict(diff=float(dp.mean()), lo=float(lo), hi=float(hi), lower=int((dp < 0).sum()), n=len(dp))
            cells += [('%s [%s, %s]' % (fmt(dp.mean(), 3), fmt(lo, 3), fmt(hi, 3))).replace('-', '$-$'), '%d/%d' % ((dp < 0).sum(), len(dp))]
        else:
            cells += ['', '']
        rows += row(cells)
        res['rows'].append(rec)
    macro('tabRecursive', rows)
    for name, A in (('self', st), ('control', ct)):
        gg = sorted(A)
        mm = [np.mean(A[g]['delta']) for g in gg]
        res[name] = dict(last=gg[-1], change=float(mm[-1] / mm[0] - 1), rho=float(spearmanr(gg, mm)[0]),
                         thirds_change=[float(np.mean(np.array(A[gg[-1]]['delta'])[t]) / np.mean(np.array(A[0]['delta'])[t]) - 1) for t in th])
    res['tokens_per_gen'] = int(st[0]['n_tokens'])
    res['n_prompts'] = len(st[0]['speeds'])
    OUT['recursive'] = res
    return arms


def part_sftdpo():
    """SFT against DPO for two published post-training pipelines (length only; results/alignment_sft_dpo.json)."""
    S = jload(os.path.join(RES, 'alignment_sft_dpo.json'))
    rows = ''
    for k, v in S.items():
        ratio = v['thirds'][2] / np.mean(v['thirds'][:2])
        rows += row([k, fmt(v['thirds'][0], 3), fmt(v['thirds'][1], 3), fmt(v['thirds'][2], 3), fmt(ratio, 1) + '$\\times$', '%s (layer %d)' % (fmt(v['max_diff'], 3), v['max_layer'])])
    macro('tabSftDpo', rows)
    OUT['sft_dpo'] = {k: dict(thirds=v['thirds'], max_diff=v['max_diff'], max_layer=v['max_layer']) for k, v in S.items()}
    return S


def figures_recursive(arms, S):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 200, 'savefig.dpi': 300, 'legend.frameon': False})
    BL, OR = '#2a78d6', '#eb6834'
    st, ct = arms['self-training'], arms['control']
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for A, c, lab in ((st, OR, 'self-training'), (ct, BL, 'control (human data)')):
        gg = sorted(A)
        ax[0].plot(gg, [np.mean(A[g]['delta']) for g in gg], color=c, marker='o', ms=2.5, label=lab)
    ax[0].set_xlabel('generation')
    ax[0].set_ylabel('mean per-layer length')
    ax[0].set_title('(a) length across generations')
    ax[0].legend(fontsize=6)
    x = np.arange(1, len(st[0]['delta']) + 1)
    ax[1].plot(x, st[0]['delta'], color='#555555', lw=1.3, label='generation 0')
    ax[1].plot(x, st[14]['delta'], color=OR, lw=1.1, label='self-training, gen 14')
    ax[1].plot(x, ct[14]['delta'], color=BL, lw=1.1, ls='--', label='control, gen 14')
    ax[1].set_xlabel('layer')
    ax[1].set_ylabel('length $\\mathcal{L}_\\ell$')
    ax[1].set_title('(b) layerwise length')
    ax[1].legend(fontsize=6)
    rr = [r for r in OUT['recursive']['rows'] if 'paired' in r]
    gx = [r['gen'] for r in rr]
    ax[2].errorbar(gx, [r['paired']['diff'] for r in rr], yerr=[[r['paired']['diff'] - r['paired']['lo'] for r in rr], [r['paired']['hi'] - r['paired']['diff'] for r in rr]],
                   color=OR, marker='o', ms=2.5, lw=1, capsize=1.5)
    ax[2].axhline(0, color='#999', lw=0.7)
    ax[2].set_xlabel('generation')
    ax[2].set_ylabel('self-training minus control')
    ax[2].set_title('(c) paired difference, 128 prompts')
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_recursive.png')
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    cols = {'Tulu-3.1 8B': '#2a78d6', 'Olmo-3 7B Instruct': '#1baf7a'}
    for k, v in S.items():
        xs = np.arange(len(v['sft_length']))
        ax[0].plot(xs, v['sft_length'], color=cols[k], lw=1.2, label='%s SFT' % k)
        ax[0].plot(xs, v['dpo_length'], color=cols[k], lw=1.0, ls='--', label='%s DPO' % k)
        ax[1].plot(xs, np.abs(np.array(v['sft_length']) - np.array(v['dpo_length'])), color=cols[k], lw=1.2, label=k)
    ax[0].set_xlabel('layer')
    ax[0].set_ylabel('length (scaled within pair)')
    ax[0].set_title('(a) SFT and DPO length profiles')
    ax[0].legend(fontsize=5.5)
    ax[1].set_xlabel('layer')
    ax[1].set_ylabel('|SFT $-$ DPO|')
    ax[1].set_title('(b) difference by layer')
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_sftdpo.png')
    plt.close(fig)


# public configuration constants (config.json of each checkpoint): layers, hidden width, vocabulary size, tokenizer
META = {'EleutherAI_pythia_410m_deduped': (24, 1024, 50304, 'neox'), 'Qwen_Qwen1.5_7B': (32, 4096, 151936, 'qwen'),
        'Qwen_Qwen1.5_7B_Chat': (32, 4096, 151936, 'qwen'), 'TinyLlama_TinyLlama_1.1B_Chat_v1.0': (22, 2048, 32000, 'llama2'),
        'allenai_Llama_3.1_Tulu_3_8B': (32, 4096, 128256, 'llama3'), 'deepseek_ai_deepseek_llm_7b_base': (30, 4096, 102400, 'deepseek'),
        'deepseek_ai_deepseek_llm_7b_chat': (30, 4096, 102400, 'deepseek'), 'google_gemma_7b': (28, 3072, 256000, 'gemma'),
        'google_gemma_7b_it': (28, 3072, 256000, 'gemma'), 'lmsys_vicuna_13b_v1.5': (40, 5120, 32000, 'llama2'),
        'lmsys_vicuna_7b_v1.5': (32, 4096, 32000, 'llama2'), 'meta_llama_Llama_2_7b_hf': (32, 4096, 32000, 'llama2'),
        'meta_llama_Meta_Llama_3_8B': (32, 4096, 128256, 'llama3'), 'meta_llama_Meta_Llama_3_8B_Instruct': (32, 4096, 128256, 'llama3'),
        'microsoft_Orca_2_13b': (40, 5120, 32003, 'llama2'), 'microsoft_Orca_2_7b': (32, 4096, 32003, 'llama2'),
        'microsoft_phi_2': (32, 2560, 51200, 'phi'), 'mistralai_Mistral_7B_v0.1': (32, 4096, 32000, 'mistral1'),
        'tiiuae_falcon_7b': (32, 4544, 65024, 'falcon'), 'Qwen_Qwen3_4B': (36, 2560, 151936, 'qwen'),
        'Qwen_Qwen3_4B_Instruct_2507': (36, 2560, 151936, 'qwen'), 'Qwen_Qwen3_4B_Thinking_2507': (36, 2560, 151936, 'qwen'),
        'allenai_Olmo_3_1025_7B': (32, 4096, 100278, 'olmo3'), 'allenai_Olmo_3_7B_Instruct': (32, 4096, 100278, 'olmo3'),
        'allenai_Olmo_3_7B_Think': (32, 4096, 100278, 'olmo3'), 'deepseek_ai_DeepSeek_R1_Distill_Llama_8B': (32, 4096, 128256, 'llama3'),
        'google_gemma_3_1b_it': (26, 1152, 262144, 'gemma3'), 'google_gemma_3_1b_pt': (26, 1152, 262144, 'gemma3'),
        'mistralai_Mistral_7B_Instruct_v0.3': (32, 4096, 32768, 'mistral3'), 'mistralai_Mistral_7B_v0.3': (32, 4096, 32768, 'mistral3')}


def lineage_of(n):
    n = n.lower()
    for keys, lab in ((['deepseek_llm'], 'DeepSeek-LLM'), (['llama_2', 'vicuna', 'orca'], 'Llama-2'), (['tulu', 'meta_llama_3', 'r1_distill'], 'Llama-3'),
                      (['qwen1.5'], 'Qwen-1.5'), (['qwen3'], 'Qwen-3'), (['gemma_3'], 'Gemma-3'), (['gemma'], 'Gemma'), (['mistral'], 'Mistral'), (['olmo'], 'Olmo-3')):
        if any(k in n for k in keys):
            return lab
    return n


def _cut(x, part):
    x = np.asarray(x, float)
    n = len(x)
    return x if part == 'full' else (x[n // 3:] if part == 'last2' else x[2 * n // 3:])


def _mm(x, lo=None, hi=None):
    lo = x.min() if lo is None else lo
    hi = x.max() if hi is None else hi
    return 0.005 + 0.995 * (x - lo) / (hi - lo) if hi > lo else np.full_like(x, 0.5)


def part_depth_robustness(P):
    """A2: decoder/task decomposition, identification and retrieval on the full depth, the last two thirds and the last third."""
    raw = OUT['_dec_raw']
    lm = np.array([r[0] for r in raw])
    lt = np.array([r[1] for r in raw])
    rng = np.random.default_rng(7)

    def strat_p(M, lab, strata, nperm=1000):
        obs = anova_stats(M, lab)[1]
        c = 0
        for _ in range(nperm):
            pl = lab.copy()
            for s_ in np.unique(strata):
                ii = np.where(strata == s_)[0]
                pl[ii] = lab[rng.permutation(ii)]
            c += anova_stats(M, pl)[1] >= obs - 1e-12
        return (c + 1) / (nperm + 1)
    parts =[('full', 'all layers'), ('last2', 'last two thirds'), ('last3', 'last third')]
    res, rows = {}, ''
    for part, plab in parts:
        # rescale jointly over each decoder's ten tasks, per channel, after cutting
        cut = {}
        for d in np.unique(lm):
            idx = np.where(lm == d)[0]
            for ch, k in (('L', 2), ('V', 3)):
                xs = [_cut(raw[i][k], part) for i in idx]
                lo, hi = min(x.min() for x in xs), max(x.max() for x in xs)
                for i, x in zip(idx, xs):
                    cut[(i, ch)] = _mm(x, lo, hi)
        res[part] = {}
        for clab, chs in (('$\\mathcal{L}$', ['L']), ('$\\lVert\\mathbf{v}\\rVert$', ['V']), ('$\\mathcal{L},\\lVert\\mathbf{v}\\rVert$', ['L', 'V'])):
            M, _ = dmat([np.stack([cut[(i, c)] for c in chs], 1) for i in range(len(raw))])
            _, _, om = anova_stats(M, lm)
            _, _, ot = anova_stats(M, lt)
            pm, pt_ = strat_p(M, lm, lt), strat_p(M, lt, lm)
            nnm = 0
            for i in range(len(raw)):
                d = M[i].copy()
                d[i] = np.inf
                nnm += lm[i] == lm[int(np.argmin(d))]
            res[part][clab] = dict(w2m=float(om), pm=float(pm), w2t=float(ot), pt=float(pt_), nnm=int(nnm))
            rows += row([plab if clab == '$\\mathcal{L}$' else '', clab, fmt(om, 2), _pf(pm), fmt(ot, 2), _pf(pt_), '%d/40' % nnm])
    macro('tabDepthDecomp', rows)

    # identification across sample size and lineage retrieval, per-profile rescaling after cutting
    g1 = {short(k): v for k, v in P.items() if k.startswith('1000/')}
    g2 = {short(k): v for k, v in P.items() if k.startswith('2500/')}
    common = sorted(n for n in g1 if n in g2)
    names = [n for n in sorted(g2) if len(g2[n]['Spectral']) >= 15]
    groups = [lineage_of(n) for n in names]
    rows2, idres = '', {}
    for part, plab in parts:
        def prof(d, chs):
            return np.stack([_mm(_cut(d[c], part)) for c in chs], 1)
        idres[part] = {}
        for clab, chs in (('$\\mathcal{L}$', ['Thermo']), ('$\\mathcal{L},\\lVert\\mathbf{v}\\rVert$', LV)):
            top, mg = 0, []
            for n in common:
                a = prof(g1[n], chs)
                ds = {m: dtw(a, prof(g2[m], chs))[0] for m in common if abs(len(g1[n]['Thermo']) - len(g2[m]['Thermo'])) <= 0.5 * max(len(g1[n]['Thermo']), len(g2[m]['Thermo']))}
                o = min(v for m, v in ds.items() if m != n)
                top += ds[n] < o
                mg.append(o / max(ds[n], 1e-12))
            M, _ = dmat([prof(g2[n], chs) for n in names])
            lr = nn_lineage(M, groups, nperm=1000)
            idres[part][clab] = dict(top1=int(top), n=len(common), med=float(np.median(mg)), nn=lr['hit'], k=lr['k'], auc=lr['auc'], mrr=lr['mrr'])
            rows2 += row([plab if clab == '$\\mathcal{L}$' else '', clab, '%d/%d' % (top, len(common)), fmt(np.median(mg), 1) + '$\\times$',
                          '%d/%d' % (lr['hit'], lr['k']), fmt(lr['auc'], 2), fmt(lr['mrr'], 2)])
    macro('tabDepthIdent', rows2)
    OUT['depth_robustness'] = dict(decomposition=res, identification=idres)


def part_metadata(P):
    """A3: retrieval from architecture metadata alone, and profile retrieval inside a stratum of identical architecture."""
    R = jload(os.path.join(RES, 'squad_report_matrices.json'))
    rn = [n.replace('method5_', '') for n in R['method_5_generic/squad/LLM_curve_similarity_report.html'][0]['names']]
    cohorts = {'$n{=}2500$': {short(k): v for k, v in P.items() if k.startswith('2500/') and len(v['Spectral']) >= 15},
               '19-model set': {short(k): v for k, v in P.items() if k.startswith('method_5_generic/squad/') and short(k) in rn}}
    rng = np.random.default_rng(3)

    def meta_dist(names, with_tok):
        X = np.log(np.array([META[n][:3] for n in names], float))
        X = (X - X.mean(0)) / X.std(0).clip(1e-9)
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        if with_tok:
            tok = np.array([META[n][3] for n in names])
            D = D + 100.0 * (tok[:, None] != tok[None])
        return D

    def tie_avg(D, groups, draws=300):
        """Average NN hit, AUC and MRR over random tie-breaks (exactly equal metadata gives ties)."""
        out = []
        for _ in range(draws):
            J = D + rng.uniform(0, 1e-9, D.shape)
            J = (J + J.T) / 2
            out.append(nn_lineage(J, groups, nperm=0))
        return dict(hit=float(np.mean([o['hit'] for o in out])), k=out[0]['k'], auc=float(np.mean([o['auc'] for o in out])), mrr=float(np.mean([o['mrr'] for o in out])))
    res, rows = {}, ''
    for cname, g in cohorts.items():
        names = sorted(g)
        groups = [lineage_of(n) for n in names]
        Mp, _ = dmat([curve(g[n], LV) for n in names])
        prof = nn_lineage(Mp, groups, nperm=0)
        mnum = tie_avg(meta_dist(names, False), groups)
        mtok = tie_avg(meta_dist(names, True), groups)
        toks = [META[n][3] for n in names]
        # can lineage be told apart from tokenizer in this cohort? count tokenizers shared by more than one lineage
        mixed = sorted({t for t in toks if len({groups[i] for i in range(len(names)) if toks[i] == t}) > 1})
        res[cname] = dict(profile=prof, meta_numeric=mnum, meta_tokenizer=mtok, tokenizers_with_two_lineages=mixed)
        for lab, v in (('profile ($\\mathcal L,\\lVert\\mathbf v\\rVert$)', prof), ('metadata: depth, width, vocabulary', mnum), ('metadata + tokenizer identity', mtok)):
            rows += row([cname if lab.startswith('profile') else '', lab, ('%d' % v['hit'] if float(v['hit']).is_integer() else fmt(v['hit'], 1)) + '/%d' % v['k'], fmt(v['auc'], 2), fmt(v['mrr'], 2)])
    macro('tabMeta', rows)
    # stratum of identical depth and width with near-identical vocabulary (Llama-2 7B fine-tunes against Mistral 7B)
    srows = ''
    for cname, g in cohorts.items():
        names = sorted(n for n in g if META[n][0] == 32 and META[n][1] == 4096 and 32000 <= META[n][2] <= 32768)
        groups = [lineage_of(n) for n in names]
        Mp, _ = dmat([curve(g[n], LV) for n in names])
        same = [Mp[i, j] for i in range(len(names)) for j in range(i + 1, len(names)) if groups[i] == groups[j]]
        diff = [Mp[i, j] for i in range(len(names)) for j in range(i + 1, len(names)) if groups[i] != groups[j]]
        auc = float(np.mean([(s < d) + 0.5 * (s == d) for s in same for d in diff])) if same and diff else float('nan')
        hits, k = 0, 0
        for i in range(len(names)):
            if sum(gg == groups[i] for gg in groups) < 2:
                continue
            d = Mp[i].copy()
            d[i] = np.inf
            k += 1
            hits += groups[int(np.argmin(d))] == groups[i]
        res[cname]['stratum'] = dict(models=[pn(n) for n in names], groups=groups, nn=int(hits), k=k, auc=auc, n_same=len(same), n_diff=len(diff))
        srows += row([cname, ', '.join(pn(n) for n in names), '%d/%d' % (hits, k), fmt(auc, 2), '%d / %d' % (len(same), len(diff))])
    macro('tabMetaStratum', srows)
    # identification across sample size from metadata alone: candidates with identical configuration tie
    c = sorted({short(k) for k in P if k.startswith('1000/')} & {short(k) for k in P if k.startswith('2500/')})
    res['ident_expected'] = dict(top1=float(sum(1 / sum(META[m] == META[n] for m in c) for n in c)), n=len(c))
    OUT['metadata'] = res


def part_combine(P):
    """Why channels are stacked: per-layer and whole-profile combinations of length and field against stacking."""
    from scipy.stats import spearmanr
    g1 = {short(k): v for k, v in P.items() if k.startswith('1000/')}
    g2 = {short(k): v for k, v in P.items() if k.startswith('2500/')}
    common = sorted(n for n in g1 if n in g2)
    names = [n for n in sorted(g2) if len(g2[n]['Spectral']) >= 15]
    groups = [lineage_of(n) for n in names]
    Lf = lambda d: np.array(d['Thermo'], float)
    Vf = lambda d: np.array(d['Belief'], float)
    per_layer = [('stacked $(\\mathcal L_\\ell,\\lVert\\mathbf v_\\ell\\rVert)$ (protocol)', lambda d: np.stack([Lf(d), Vf(d)], 1)),
                 ('$\\mathcal L_\\ell$ alone', lambda d: Lf(d)[:, None]),
                 ('product $\\mathcal L_\\ell\\lVert\\mathbf v_\\ell\\rVert$', lambda d: _mm(Lf(d) * Vf(d))[:, None]),
                 ('geometric mean', lambda d: _mm(np.sqrt(Lf(d) * Vf(d)))[:, None]),
                 ('sum $\\mathcal L_\\ell+\\lVert\\mathbf v_\\ell\\rVert$', lambda d: _mm(Lf(d) + Vf(d))[:, None]),
                 ('difference $\\mathcal L_\\ell-\\lVert\\mathbf v_\\ell\\rVert$', lambda d: _mm(Lf(d) - Vf(d))[:, None])]
    eps = 1e-12
    whole = [('dashboard composite (product of three)', lambda d: d['score']),
             ('$\\sum_\\ell\\mathcal L_\\ell\\lVert\\mathbf v_\\ell\\rVert$', lambda d: float((Lf(d) * Vf(d)).sum())),
             ('$\\sum_\\ell(\\log\\mathcal L_\\ell+\\log\\lVert\\mathbf v_\\ell\\rVert)$', lambda d: float((np.log(Lf(d) + eps) + np.log(Vf(d) + eps)).sum())),
             ('$\\sum_\\ell(\\mathcal L_\\ell+\\lVert\\mathbf v_\\ell\\rVert)$', lambda d: float((Lf(d) + Vf(d)).sum())),
             ('$\\bar{\\mathcal L}-\\overline{\\lVert\\mathbf v\\rVert}$', lambda d: float(Lf(d).mean() - Vf(d).mean()))]
    res, rows = dict(per_layer={}, whole={}), ''
    ok = lambda a, b: abs(len(g1[a]['Thermo']) - len(g2[b]['Thermo'])) <= 0.5 * max(len(g1[a]['Thermo']), len(g2[b]['Thermo']))
    for lab, f in per_layer:
        top = 0
        for n in common:
            a = f(g1[n])
            ds = {m: dtw(a, f(g2[m]))[0] for m in common if ok(n, m)}
            top += min(ds, key=ds.get) == n
        M, _ = dmat([f(g2[n]) for n in names])
        lr = nn_lineage(M, groups, nperm=0)
        res['per_layer'][lab] = dict(ident=int(top), n=len(common), nn=lr['hit'], k=lr['k'], auc=lr['auc'])
        rows += row(['per layer' if lab.startswith('stacked') else '', lab, '%d/%d' % (top, len(common)), '%d/%d' % (lr['hit'], lr['k']), fmt(lr['auc'], 2), ''])
    for lab, f in whole:
        s1 = {n: f(g1[n]) for n in common}
        s2 = {n: f(g2[n]) for n in common}
        top = sum(min(common, key=lambda m: abs(s1[n] - s2[m])) == n for n in common)
        sv = np.array([f(g2[n]) for n in names], float)
        M = np.abs(sv[:, None] - sv[None])
        lr = nn_lineage(M, groups, nperm=0)
        rho = float(spearmanr([s1[n] for n in common], [s2[n] for n in common])[0])
        res['whole'][lab] = dict(ident=int(top), n=len(common), nn=lr['hit'], k=lr['k'], auc=lr['auc'], rho=rho)
        rows += row(['whole profile' if lab.startswith('dashboard') else '', lab, '%d/%d' % (top, len(common)), '%d/%d' % (lr['hit'], lr['k']), fmt(lr['auc'], 2), fmt(rho, 2)])
    macro('tabCombine', rows)
    res['zero_scores'] = [int(sum(v.get('score') == 0 for v in P.values())), int(sum(v.get('score') is not None for v in P.values()))]
    OUT['combine'] = res


def part_amplitude():
    """Close sibling pair without rescaling (results/closepair_raw.json): does amplitude separate the two models?"""
    R = jload(os.path.join(RES, 'closepair_raw.json'))['runs']
    lab = [(r['model'], r['dataset']) for r in R]
    D = [np.array(r['delta']) for r in R]
    V = [np.array(r['vnorm']) for r in R]
    n = len(R)
    th = np.array_split(np.arange(len(D[0])), 3)
    sc = lambda x: 0.005 + 0.995 * (x - x.min()) / (x.max() - x.min())
    res, rows = dict(runs=[], channels={}), ''
    for lab_, curves in [('$\\mathcal{L}$, unscaled', D), ('$\\mathcal{L}$, per profile', [sc(d) for d in D]),
                         ('$\\lVert\\mathbf v\\rVert$, unscaled', [100 * v for v in V]), ('$\\lVert\\mathbf v\\rVert$, per profile', [sc(v) for v in V])]:
        M, _ = dmat(curves)
        wm = np.mean([M[i, j] for i in range(n) for j in range(i + 1, n) if lab[i][0] == lab[j][0]])
        wd = np.mean([M[i, j] for i in range(n) for j in range(i + 1, n) if lab[i][1] == lab[j][1]])
        nnm = nnd = 0
        for i in range(n):
            d = M[i].copy()
            d[i] = np.inf
            j = int(np.argmin(d))
            nnm += lab[i][0] == lab[j][0]
            nnd += lab[i][1] == lab[j][1]
        res['channels'][lab_] = dict(within_model=float(wm), within_dataset=float(wd), ratio=float(wd / wm), nn_model=nnm, nn_dataset=nnd)
        rows += row([lab_, sci(wm) if wm < 1e-3 else fmt(wm, 4), sci(wd) if wd < 1e-3 else fmt(wd, 4), fmt(wd / wm, 2), '%d/%d' % (nnm, n), '%d/%d' % (nnd, n)])
    macro('tabAmpDist', rows)
    rows = ''
    for ds in ['ag-news', 'automathtext', 'stanford_plato']:
        i, j = lab.index(('Llama-3 8B', ds)), lab.index(('R1-Distill-Llama 8B', ds))
        l, q = D[i], D[j]
        al, aq = np.mean(R[i]['alpha']), np.mean(R[j]['alpha'])
        rec = dict(dataset=ds, n=R[i]['n_examples'], llama=float(l.mean()), r1=float(q.mean()), ratio=float(q.mean() / l.mean()),
                   thirds=[float(q[t].mean() / l[t].mean()) for t in th], layers_lower=int((q < l).sum()), alpha=[float(al), float(aq)],
                   vnorm=[float(V[i].mean()), float(V[j].mean())])
        res['runs'].append(rec)
        rows += row([ds.replace('_', '-'), R[i]['n_examples'], fmt(l.mean(), 3), fmt(q.mean(), 3), fmt(rec['ratio'], 3)] + [fmt(x, 2) for x in rec['thirds']] +
                    ['%d/31' % rec['layers_lower'], fmt(al, 3), fmt(aq, 3)])
    macro('tabAmpRuns', rows)
    SL = np.array([D[i] for i in range(n) if lab[i][0] == 'Llama-3 8B'])
    SR = np.array([D[i] for i in range(n) if lab[i][0] != 'Llama-3 8B'])
    res['layers_separated'] = int((SL.min(0) > SR.max(0)).sum())
    res['dataset_range'] = dict(llama=float(SL.mean(1).max() - SL.mean(1).min()), r1=float(SR.mean(1).max() - SR.mean(1).min()))
    res['share_first_third'] = [float(d[th[0]].sum() / d.sum()) for d in D]
    OUT['amplitude'] = res
    return R


def part_settle(E):
    """Where predictions settle: tail length and commitment depth of the four exported decoders (exp08)."""
    rows, res = '', {}
    for m in MODELS:
        d = E[m]['e08']['arrays']
        L, tl = arr(E[m]['e08'], 'length'), arr(E[m]['e08'], 'tail_length')
        tot = float(arr(E[m]['e08'], 'length_total'))
        eps, cf = arr(E[m]['e08'], 'eps_fraction'), arr(E[m]['e08'], 'commitment_depth_frac')
        th = np.array_split(np.arange(len(L)), 3)
        frac = tl / tot
        half = float(np.argmax(frac <= 0.5) / len(L))
        res[m] = dict(length=L.tolist(), total=tot, share=[float(L[t].sum() / L.sum()) for t in th], half=half, eps=eps.tolist(), commit=cf.tolist(), tail=frac.tolist())
        c = lambda e: fmt(min(cf[list(eps).index(e)], 1.0), 2) if cf[list(eps).index(e)] < 1 else 'none'
        rows += row([MNAME[m], len(L), fmt(tot, 1)] + [fmt(x, 2) for x in res[m]['share']] + [fmt(half, 2), c(0.2), c(0.1), c(0.05)])
    macro('tabSettle', rows)
    OUT['settle'] = res


def part_recursive_effort():
    """Parameter-space effort E and the original interior-angle score across recursive generations."""
    from scipy.stats import spearmanr
    R = jload(os.path.join(RES, 'recursive_runs.json'))['arms']
    res, rows = {}, ''
    for name, a in R.items():
        G = {int(g): v for g, v in a['generations'].items()}
        gg = sorted(G)
        Em = np.array([np.mean(G[g]['E']) for g in gg])
        Ed = np.array([np.median(G[g]['E']) for g in gg])
        k = np.array([np.mean(G[g]['kappa_legacy']) for g in gg])
        res[name] = dict(gens=gg, E_mean=Em.tolist(), E_median=Ed.tolist(), change=float(Em[-1] / Em[0] - 1), rho=float(spearmanr(gg, Em)[0]),
                         kappa_legacy_range=[float(k.min()), float(k.max())],
                         rho_len_E=float(spearmanr([np.mean(G[g]['delta']) for g in gg], Em)[0]))
    OUT['recursive_effort'] = res


def figures_more(arms, P):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    plt.rcParams.update({'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 200, 'savefig.dpi': 300, 'legend.frameon': False})
    BL, OR, GN, PU = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
    DSL = {'ag-news': '-', 'automathtext': '--', 'stanford_plato': ':'}
    MC = {'Llama-3 8B': BL, 'R1-Distill-Llama 8B': OR}

    # ---- amplitude: close sibling pair, unscaled (2D)
    R = jload(os.path.join(RES, 'closepair_raw.json'))['runs']
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.0))
    for r in R:
        x = np.arange(1, len(r['delta']) + 1)
        ls = DSL[r['dataset']]
        ax[0].plot(x, r['delta'], color=MC[r['model']], ls=ls, lw=1.0)
        ax[2].plot(x, 1e3 * np.array(r['vnorm']), color=MC[r['model']], ls=ls, lw=1.0)
        ax[3].plot(x, r['alpha'], color=MC[r['model']], ls=ls, lw=1.0)
    for ds, ls in DSL.items():
        l = [r for r in R if r['dataset'] == ds and r['model'] == 'Llama-3 8B'][0]
        q = [r for r in R if r['dataset'] == ds and r['model'] != 'Llama-3 8B'][0]
        ax[1].plot(np.arange(1, 32), np.array(q['delta']) / np.array(l['delta']), color='#555555', ls=ls, lw=1.0, label=ds.replace('_', '-'))
    ax[1].axhline(1, color='#999', lw=0.6)
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], color=BL, lw=1.2, label='Llama-3 8B'), Line2D([], [], color=OR, lw=1.2, label='R1-Distill-Llama 8B')], fontsize=5.5, loc='lower left')
    ax[1].legend(fontsize=5.5)
    for a, t, yl in zip(ax, ['(a) length, unscaled', '(b) R1-Distill / Llama', '(c) field norm, unscaled', '(d) alignment cosine'],
                        [r'$\mathcal{L}_\ell$', 'ratio of lengths', r'$\Vert\bar{\mathbf{v}}_\ell\Vert\times10^3$', r'$\alpha_\ell$']):
        a.set_title(t, fontsize=7)
        a.set_xlabel('layer')
        a.set_ylabel(yl)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_amplitude.png')
    plt.close(fig)

    # ---- amplitude: 3D trajectories (layer, unscaled length, unscaled field)
    fig = plt.figure(figsize=(3.3, 2.9))
    ax = fig.add_subplot(111, projection='3d')
    for r in R:
        x = np.arange(1, len(r['delta']) + 1)
        ax.plot(x, r['delta'], 1e3 * np.array(r['vnorm']), color=MC[r['model']], ls=DSL[r['dataset']], lw=1.1, marker='o', ms=1.2)
    ax.set_xlabel('layer', labelpad=-7, fontsize=6.5)
    ax.set_ylabel(r'length $\mathcal{L}_\ell$', labelpad=-7, fontsize=6.5)
    ax.set_zlabel(r'field $\times10^3$', labelpad=-7, fontsize=6.5)
    ax.tick_params(labelsize=5, pad=-3)
    ax.view_init(elev=20, azim=-58)
    ax.legend(handles=[Line2D([], [], color=BL, lw=1.2, label='Llama-3 8B'), Line2D([], [], color=OR, lw=1.2, label='R1-Distill-Llama 8B')] +
              [Line2D([], [], color='#555', ls=ls, lw=1, label=ds.replace('_', '-')) for ds, ls in DSL.items()], fontsize=5.3, loc='upper left', bbox_to_anchor=(0.0, 1.0))
    fig.subplots_adjust(left=0.0, right=0.95, top=1.0, bottom=0.04)
    fig.savefig(f'{FIG}/fig_amplitude_3d.png')
    plt.close(fig)

    # ---- recursive: 3D surfaces (layer x generation x length) and parameter effort
    st, ct = arms['self-training'], arms['control']
    fig = plt.figure(figsize=(7.6, 2.5))
    common = [g for g in sorted(st) if g in ct]
    x = np.arange(1, 32)
    zmax = max(max(max(A[g]['delta']) for g in A) for A in (st, ct))
    for i, (A, ttl, cmap) in enumerate(((st, '(a) self-training', 'Oranges'), (ct, '(b) control (human data)', 'Blues'))):
        ax = fig.add_axes([0.245 * i, 0.02, 0.26, 0.92], projection='3d')
        gg = sorted(A)
        X, Y = np.meshgrid(x, gg)
        Z = np.array([A[g]['delta'] for g in gg])
        ax.plot_surface(X, Y, Z, cmap=cmap, vmin=0, vmax=zmax * 1.1, linewidth=0.15, edgecolor='#ffffff55', antialiased=True)
        ax.set_zlim(0.3, zmax)
        ax.set_title(ttl, fontsize=7, pad=-2)
    ax = fig.add_axes([0.49, 0.02, 0.26, 0.92], projection='3d')
    X, Y = np.meshgrid(x, common)
    Z = np.array([np.array(st[g]['delta']) - np.array(ct[g]['delta']) for g in common])
    ax.plot_surface(X, Y, Z, cmap='RdBu_r', vmin=-0.2, vmax=0.2, linewidth=0.15, edgecolor='#ffffff55')
    ax.set_title('(c) self-training minus control', fontsize=7, pad=-2)
    for a in fig.axes[:3]:
        a.set_xlabel('layer', labelpad=-8, fontsize=6)
        a.set_ylabel('generation', labelpad=-8, fontsize=6)
        a.set_zlabel('length' if a is not fig.axes[2] else '', labelpad=-9, fontsize=6)
        a.tick_params(labelsize=4.5, pad=-3)
        a.view_init(elev=25, azim=-50)
    ax = fig.add_axes([0.855, 0.2, 0.14, 0.66])
    for A, c, labx in ((st, OR, 'self-training'), (ct, BL, 'control')):
        gg = sorted(A)
        ax.plot(gg, [np.mean(A[g]['E']) / 1e6 for g in gg], color=c, marker='o', ms=2.2, lw=1.1, label=labx)
    ax.set_xlabel('generation')
    ax.set_ylabel(r'mean $E_\ell$ ($10^6$)')
    ax.set_title('(d) parameter effort', fontsize=7)
    ax.legend(fontsize=5.5)
    fig.savefig(f'{FIG}/fig_recursive_3d.png')
    plt.close(fig)

    # ---- where predictions settle (exp08): remaining length and cumulative share
    fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.8))
    cols = {'gemma3_1b': GN, 'llama31_8b': BL, 'qwen3_4b': OR, 'qwen3_8b': PU}
    for m in MODELS:
        tail = np.array(OUT['settle'][m]['tail'])
        rel = np.arange(len(tail)) / (len(tail) - 1)
        ax[0].plot(rel, tail, color=cols[m], lw=1.1, label=MNAME[m])
        steps = np.array(OUT['settle'][m]['length'])
        ax[1].plot(np.arange(1, len(steps) + 1) / len(steps), steps / steps.sum() * len(steps), color=cols[m], lw=1.0)
    for e in (0.2, 0.1):
        ax[0].axhline(e, color='#aaa', lw=0.5, ls=':')
    ax[0].set_xlabel('relative depth')
    ax[0].set_ylabel('share of length still ahead')
    ax[0].set_title('(a) remaining length', fontsize=7)
    ax[0].legend(fontsize=5)
    ax[1].set_xlabel('relative depth')
    ax[1].set_ylabel('step / mean step')
    ax[1].set_title('(b) relative step size', fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_settle.png')
    plt.close(fig)

    # ---- ten tasks x four decoders as 3D waterfalls of length
    raw = OUT['_dec_raw']
    dec = list(dict.fromkeys(r[0] for r in raw))
    tasks = list(dict.fromkeys(r[1] for r in raw))
    fig = plt.figure(figsize=(6.2, 4.8))
    tcol = plt.get_cmap('viridis')(np.linspace(0, 0.95, len(tasks)))
    for i, dn in enumerate(dec):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        for j, t in enumerate(tasks):
            r = [r for r in raw if r[0] == dn and r[1] == t][0]
            y = np.array(r[2])
            xr = np.linspace(0, 1, len(y))
            ax.plot(xr, np.full_like(xr, j), y, color=tcol[j], lw=0.9)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([t.replace('_', ' ') for t in tasks], fontsize=4.6, rotation=0, va='center', ha='left')
        ax.set_xlabel('relative depth', labelpad=-8, fontsize=6)
        ax.set_zlabel(r'$\mathcal{L}_\ell$', labelpad=-9, fontsize=6)
        ax.tick_params(axis='x', labelsize=4.5, pad=-4)
        ax.tick_params(axis='z', labelsize=4.5, pad=-2)
        ax.tick_params(axis='y', pad=-3)
        ax.view_init(elev=24, azim=-62)
        ax.set_title(dn, fontsize=7, pad=-4)
    fig.subplots_adjust(left=0.0, right=0.93, top=0.96, bottom=0.02, wspace=0.12, hspace=0.08)
    fig.savefig(f'{FIG}/fig_tasks_3d.png')
    plt.close(fig)


def part_zoo():
    p = os.path.join(RES, 'model_zoo.json')
    if os.path.exists(p):
        OUT['zoo'] = jload(p)


# ----------------------------------------------------------------------------- figures
def figures(E, P):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                         'figure.dpi': 200, 'savefig.dpi': 300, 'axes.titlesize': 8.5, 'legend.frameon': False})
    COL = {'gemma3_1b': '#2a78d6', 'llama31_8b': '#eb6834', 'qwen3_4b': '#1baf7a', 'qwen3_8b': '#9a6b00'}
    MK = {'gemma3_1b': 'o', 'llama31_8b': 's', 'qwen3_4b': '^', 'qwen3_8b': 'D'}
    INK = '#52514e'

    # --- F1 absolute profiles
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1))
    for m in MODELS:
        ln = arr(E[m]['e08'], 'length')
        bl = arr(E[m]['e08'], 'belief')[1:]
        x = np.arange(1, len(ln) + 1) / len(ln)
        ax[0].plot(x, ln, color=COL[m], marker=MK[m], ms=2.5, label=MNAME[m])
        ax[1].plot(np.arange(1, len(bl) + 1) / len(bl), bl, color=COL[m], marker=MK[m], ms=2.5)
        ax[2].plot(x, np.cumsum(ln) / ln.sum(), color=COL[m])
    ax[0].set_title('(a) thermodynamic length $\\mathcal{L}_\\ell$')
    ax[1].set_title('(b) belief field norm $\\Vert\\mathbf{v}_\\ell\\Vert$')
    ax[2].set_title('(c) cumulative share of total length')
    for a in ax:
        a.set_xlabel('relative depth $\\ell/L$')
    ax[2].plot([0, 1], [0, 1], color='#999', lw=0.7, ls='--')
    ax[0].legend(fontsize=6.5, loc='upper center')
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_profiles.png')
    plt.close(fig)

    # --- F2 sample depth
    sd = OUT['sample_depth']
    g1 = {short(k): v for k, v in P.items() if k.startswith('1000/')}
    g2 = {short(k): v for k, v in P.items() if k.startswith('2500/')}
    show = ['meta_llama_Meta_Llama_3_8B', 'google_gemma_7b', 'deepseek_ai_deepseek_llm_7b_base', 'microsoft_phi_2']
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.7), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for i, n in enumerate(show[:2]):
        for k, ch in enumerate(['Thermo', 'Belief']):
            a = ax[i, k]
            a.plot(g1[n][ch], color='#2a78d6', label='n=1000')
            a.plot(g2[n][ch], color='#eb6834', ls='--', label='n=2500')
            a.set_title('%s: %s' % (pn(n), '$\\mathcal{L}_\\ell$' if k == 0 else '$\\Vert\\mathbf{v}_\\ell\\Vert$'), fontsize=7.5)
    ax[0, 0].legend(fontsize=6.5)
    ax[1, 0].set_xlabel('layer')
    ax[1, 1].set_xlabel('layer')
    rows = sorted(sd['rows'], key=lambda r: r['margin'])
    gs = ax[0, 2].get_gridspec()
    for a in ax[:, 2]:
        a.remove()
    big = fig.add_subplot(gs[:, 2])
    for y, r in enumerate(rows):
        big.plot([r['d_self'], r['d_other']], [y, y], color='#c9c8c0', lw=1)
        big.plot(r['d_self'], y, 'o', color='#2a78d6', ms=3.5)
        big.plot(r['d_other'], y, 's', color='#eb6834', ms=3.5)
    big.set_yticks(range(len(rows)))
    big.set_yticklabels([pn(r['model']) for r in rows], fontsize=5.4)
    big.set_xscale('log')
    big.set_xlabel('DTW distance (log)')
    big.set_title('(c) self vs nearest other model', fontsize=7.5)
    big.plot([], [], 'o', color='#2a78d6', label='same model')
    big.plot([], [], 's', color='#eb6834', label='nearest other')
    big.legend(fontsize=6.2, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_sample_depth.png')
    plt.close(fig)

    # --- F3 breadth
    br = OUT['breadth_theta']
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    bs = list(range(3, 11))
    for m in MODELS:
        ax[0].plot(bs, [br[m][b]['med'] for b in bs], color=COL[m], marker=MK[m], ms=2.5, label=MNAME[m])
        ax[0].fill_between(bs, [br[m][b]['lo'] for b in bs], [br[m][b]['hi'] for b in bs], color=COL[m], alpha=0.10, lw=0)
        ax[1].plot(bs, [br[m][b]['margin'] for b in bs], color=COL[m], marker=MK[m], ms=2.5)
    ax[0].axhline(1, color='#999', lw=0.7, ls='--')
    ax[0].set_title('(a) threshold $\\theta_b/\\theta_{10}$, subsets of $b$ tasks')
    ax[0].set_xlabel('number of tasks $b$')
    ax[1].axhline(1, color='#999', lw=0.7, ls='--')
    ax[1].set_title('(b) prune-40 / worst-case $\\theta_b$')
    ax[1].set_xlabel('number of tasks $b$')
    ax[0].legend(fontsize=6.2)
    bi = OUT['breadth_id']
    ax[2].plot([r['b'] for r in bi], [100 * r['acc'] for r in bi], color='#2a78d6', marker='o', ms=3, label='vs ten-task reference')
    ax[2].plot([r['b'] for r in bi], [100 * r['heldout'] for r in bi], color='#eb6834', marker='s', ms=3, ls='--', label='vs held-out tasks')
    ax[2].set_ylim(min(90, min(100 * r['heldout'] for r in bi) - 3), 101)
    ax[2].legend(fontsize=6, loc='lower right')
    ax[2].set_title('(c) model identity from $b$ tasks (%)')
    ax[2].set_xlabel('number of tasks $b$')
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_breadth.png')
    plt.close(fig)

    # --- F4 decoder vs dataset (MDS of L-only and v-only distances)
    lm, lt, ML, MV = OUT['_dec_items']
    ML, MV = np.array(ML), np.array(MV)

    def mds(M):
        n = len(M)
        J = np.eye(n) - np.ones((n, n)) / n
        Bm = -0.5 * J @ (M ** 2) @ J
        w, v = np.linalg.eigh(Bm)
        o = np.argsort(w)[::-1][:2]
        return v[:, o] * np.sqrt(np.maximum(w[o], 0))
    models = sorted(set(lm), key=lm.index)
    colm = dict(zip(models, ['#2a78d6', '#7fb3ee', '#eb6834', '#1baf7a']))
    tasks = sorted(set(lt), key=lt.index)
    mk = dict(zip(tasks, 'o s ^ v D P X * < >'.split()))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    # direct profiles: one thin line per (decoder, task); colour is the decoder
    for mname, task, th, bl in OUT['_dec_raw']:
        x = np.linspace(0, 1, len(th))
        ax[0].plot(x, th, color=colm[mname], lw=0.7, alpha=0.75)
        ax[1].plot(np.linspace(0, 1, len(bl)), bl, color=colm[mname], lw=0.7, alpha=0.75)
    ax[0].set_title('(a) length, ten tasks each')
    ax[1].set_title('(b) field norm, ten tasks each')
    for a in ax[:2]:
        a.set_xlabel('relative depth')
        a.set_ylabel('scaled value', fontsize=7)
    for mname in models:
        ax[0].plot([], [], color=colm[mname], lw=1.4, label=mname)
    ax[0].legend(fontsize=5.5, loc='upper right')
    dv = OUT['decoder_vs_task']
    keep = list(dv)[:3]
    labs = ['$\\mathcal{L}$', '$\\Vert\\mathbf{v}\\Vert$', '$\\mathcal{L},\\Vert\\mathbf{v}\\Vert$']
    x = np.arange(len(labs))
    ax[2].bar(x - 0.18, [max(dv[k]['w2m'], 0) for k in keep], 0.36, color='#2a78d6', label='model')
    ax[2].bar(x + 0.18, [max(dv[k]['w2t'], 0) for k in keep], 0.36, color='#eb6834', label='task')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labs, fontsize=6.5)
    ax[2].set_ylabel('association ($\\omega^2$, separate fits)')
    ax[2].set_title('(c) decoder and task')
    ax[2].set_xticklabels(labs, fontsize=6.5)
    ax[2].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_decoder_task.png')
    plt.close(fig)

    # --- F5 lineage heatmap
    names, groups, M = OUT['_M2500']
    M = np.array(M)
    cnt = {g: groups.count(g) for g in groups}
    gkey = [g if cnt[g] > 1 else 'zz-none' for g in groups]          # singletons last, as one block of unrelated models
    order = sorted(range(len(names)), key=lambda i: (gkey[i], names[i]))
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(M[np.ix_(order, order)], cmap='Blues_r', vmin=0, vmax=np.percentile(M, 90))
    # boundaries between known lineage groups (the ordering itself uses the labels)
    og = [gkey[i] for i in order]
    for k in range(1, len(og)):
        if og[k] != og[k - 1]:
            ax.axhline(k - 0.5, color='#d03b3b', lw=0.6)
            ax.axvline(k - 0.5, color='#d03b3b', lw=0.6)
    lab = [pn(names[i]) + ('$^\\dagger$' if cnt[groups[i]] == 1 else '') for i in order]
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(lab, rotation=90, fontsize=5)
    ax.set_yticklabels(lab, fontsize=5)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label('DTW distance (length and field)', fontsize=6)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_lineage.png')
    plt.close(fig)

    # --- F6 alternative metrics
    am = OUT['alt_metrics']
    keys = list(am)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    xx = np.arange(len(keys))
    ax[0].bar(xx - 0.2, [am[k]['two']['hit'] / am[k]['two']['k'] for k in keys], 0.4, color='#2a78d6', label='$(\\kappa,\\mathcal{L})$ view')
    ax[0].bar(xx + 0.2, [(am[k]['three']['hit'] / am[k]['three']['k']) if am[k]['three'] else 0 for k in keys], 0.4, color='#eb6834', label='all three channels')
    ax[0].axhline(OUT['lineage']['squad-19 report DTW 3ch']['chance'], color='#999', lw=0.7, ls='--')
    ax[0].set_ylabel('same-lineage nearest neighbour', fontsize=7)
    ax[0].set_ylim(0, 1.22)
    ax[1].bar(xx - 0.2, [am[k]['two']['auc'] for k in keys], 0.4, color='#2a78d6')
    ax[1].bar(xx + 0.2, [am[k]['three']['auc'] if am[k]['three'] else 0 for k in keys], 0.4, color='#eb6834')
    ax[1].set_ylim(0.5, 1.0)
    ax[1].set_ylabel('AUC (same-lineage pairs closer)')
    for a in ax:
        a.set_xticks(xx)
        a.set_xticklabels([k.replace('Discrete Fr\\\'echet', 'Frechet').replace('Area between curves', 'Area').replace(' (resampled)', '').replace('Curve length', 'CurveLen') for k in keys], rotation=30, ha='right', fontsize=6.5)
    ax[0].legend(fontsize=6.2, loc='upper left', ncol=2)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_metrics.png')
    plt.close(fig)

    # --- F7 behaviour vs geometry
    e3 = OUT['e03']
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    opm = {'quant_4bit': ('o', 'quant-4bit'), 'quant_2bit': ('s', 'quant-2bit'), 'prune_30': ('^', 'prune-30'), 'prune_60': ('v', 'prune-60'),
           'sibling_instruct': ('*', 'sibling'), 'sibling_think': ('*', 'sibling'), 'sibling_base': ('*', 'sibling')}
    done = set()
    for m in MODELS:
        for r in e3[m]['rows']:
            mk_, nm = opm[r['label']]
            lbl = nm if nm not in done else None
            done.add(nm)
            ax[0].scatter(r['ratio'], r['dtw'], c=COL[m], marker=mk_, s=22 if mk_ != '*' else 36, label=lbl, linewidths=0)
            ax[1].scatter(r['warp'], r['ratio'], c=COL[m], marker=mk_, s=22 if mk_ != '*' else 36, linewidths=0)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('perplexity ratio vs base')
    ax[0].set_ylabel('DTW from base')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('warping extent (depth points)')
    ax[1].set_ylabel('perplexity ratio vs base')
    ax[0].legend(fontsize=5.8, ncol=1, loc='lower right')
    for m in MODELS:
        ax[1].scatter([], [], c=COL[m], s=14, label=MNAME[m])
    ax[1].legend(fontsize=5.8, loc='upper left')
    ax[0].set_title('(a) geometry against behaviour')
    ax[1].set_title('(b) warp separates the failed models')
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_behaviour.png')
    plt.close(fig)

    # --- F8 patching raw vs partial
    pt = OUT['patching']
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    labs = ['$\\mathcal{L}_\\ell$', '$\\Vert\\mathbf{v}_\\ell\\Vert$', 'tail length']
    for k in range(3):
        for mi, m in enumerate(MODELS):
            x = k + (mi - 1.5) * 0.12
            ax.plot(x, pt[m]['raw'][k], 'o', mfc='none', mec=COL[m], ms=4)
            if not math.isnan(pt[m]['partial'][k][0]):
                ax.plot([x, x], [pt[m]['raw'][k], pt[m]['partial'][k][0]], color='#c9c8c0', lw=0.8)
                ax.plot(x, pt[m]['partial'][k][0], 'o', color=COL[m], ms=4)
    ax.text(2, -0.55, 'partial correlation\nundefined\n(tail is monotone in depth)', ha='center', fontsize=5.5, color='#52514e')
    ax.axhline(0, color='#999', lw=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labs)
    ax.set_ylabel('Spearman $\\rho$ with patch effect')
    ax.plot([], [], 'o', mfc='none', mec=INK, label='raw')
    ax.plot([], [], 'o', color=INK, label='controlling for layer')
    ax.legend(fontsize=6, loc='lower left')
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_patching.png')
    plt.close(fig)

    # --- F9 merging
    mg = OUT['merge']['rows']
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
    na = [('NA' in r['child'].split('+')) for r in mg]
    ax[0].scatter([r['pd'] for r, f in zip(mg, na) if not f], [r['ratio'] for r, f in zip(mg, na) if not f], c='#2a78d6', s=14, label='no NA parent', linewidths=0)
    ax[0].scatter([r['pd'] for r, f in zip(mg, na) if f], [r['ratio'] for r, f in zip(mg, na) if f], c='#eb6834', s=14, marker='s', label='NA parent', linewidths=0)
    ax[0].axhline(1, color='#999', lw=0.7, ls='--')
    ax[0].set_xlabel('distance between the two parents')
    ax[0].set_ylabel('residual / parent distance')
    ax[0].legend(fontsize=6.2)
    ax[0].set_title('(a) how far a child lies from the parental mixing line')
    ax[1].hist([r['rank'] for r in mg], bins=np.arange(0.5, 29.5, 1), color='#2a78d6')
    ax[1].set_xlabel('rank of the true parent pair among 28 pairs')
    ax[1].set_ylabel('children')
    ax[1].set_title('(b) parent recovery (chance: uniform)')
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_merge.png')
    plt.close(fig)


def figures_alignment(P):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # alignment profiles on Litmus and HarmBench (dashboards, scaled jointly within each probe group)
    grp = lambda sub: {os.path.basename(k).replace('_dashboard.html', ''): v for k, v in P.items() if f'/{sub}/' in k}
    gl, gh = grp('alignment_litmus'), grp('harmbench_plots')
    fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.3), sharex=True)
    stc = {'base': '#2a78d6', 'SFT': '#eb6834', 'DPO': '#1baf7a'}
    for r_, (fam, a_, b_) in enumerate((('Llama-3 8B', 'llama', 'Llama'), ('Qwen-3 4B', 'Qwen', 'Qwen'))):
        for c_, (probe, g, pre) in enumerate((('Litmus', gl, a_), ('HarmBench', gh, b_))):
            for k_, ch in enumerate(('Thermo', 'Belief')):
                a = ax[r_, 2 * c_ + k_]
                for st in ('base', 'SFT', 'DPO'):
                    y = np.array(g[f'{pre}_{st}'][ch], float)
                    a.plot(np.linspace(0, 1, len(y)), y, color=stc[st], lw=1.0, ls='-' if st != 'DPO' else '--', label=st)
                a.set_title('%s, %s: %s' % (fam, probe, 'length' if ch == 'Thermo' else 'field norm'), fontsize=6)
                a.tick_params(labelsize=5)
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)
    for a in ax[1]:
        a.set_xlabel('relative depth', fontsize=6)
    ax[0, 0].legend(fontsize=5.5)
    fig.tight_layout()
    fig.savefig(f'{FIG}/fig_alignment_profiles.png')
    plt.close(fig)


def figures3d(P):
    """3D depth profiles (relative depth, scaled length, scaled field norm) of measured dashboards, one panel per operation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    plt.rcParams.update({'font.size': 7.5, 'axes.linewidth': 0.5, 'figure.dpi': 200, 'savefig.dpi': 300, 'legend.frameon': False})

    def draw(ax, series, title):
        for lab, key, col, ls, lw in series:
            v = P[key]
            L, B = np.array(v['Thermo'], float), np.array(v['Belief'], float)
            x = np.linspace(0, 1, len(L))
            ax.plot(x, L, B, color=col, ls=ls, lw=lw, marker='o', ms=1.5, label=lab)
        ax.set_xlabel('relative depth', labelpad=-7, fontsize=6.5)
        ax.set_ylabel(r'length $\mathcal{L}_\ell$', labelpad=-7, fontsize=6.5)
        ax.set_zlabel(r'field $\Vert\mathbf{v}_\ell\Vert$', labelpad=-7, fontsize=6.5)
        ax.tick_params(labelsize=5, pad=-3)
        ax.view_init(elev=22, azim=-62)
        ax.set_title(title, fontsize=7.5, pad=-4)
        ax.legend(fontsize=5.6, loc='upper left', bbox_to_anchor=(0.0, 0.96))

    BL, OR, GN, PU, GY = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7', '#8a8a85'
    k25 = lambda n: f'2500/method5_squad_{n}_dashboard.html'
    mg = lambda n: f'merging_lora_qwen3_squad_all/squad__method5_lora_{n}_dashboard.html'
    al = lambda n: f'method_5_generic/harmbench_plots/{n}_dashboard.html'
    ds = lambda n: f'method_5_generic/distillation_plots/{n}_dashboard.html'
    sq = lambda n: f'method_5_generic/squad/method5_{n}_dashboard.html'
    main_panels = [
        ('(a) fine-tuning keeps the inherited path', [('Llama-2-7B (base)', k25('meta_llama_Llama_2_7b_hf'), BL, '-', 1.6), ('Vicuna-7B', k25('lmsys_vicuna_7b_v1.5'), OR, '-', 1.2),
                                                       ('Orca-2-7B', k25('microsoft_Orca_2_7b'), GN, '-', 1.2), ('Falcon-7B (unrelated)', k25('tiiuae_falcon_7b'), GY, ':', 1.2)]),
        ('(b) merging: child between two parents', [('parent EU', mg('finetuned_EU'), BL, '-', 1.4), ('parent LA', mg('finetuned_LA'), OR, '-', 1.4), ('merged EU+LA', mg('merged_EU_LA'), PU, '--', 1.8)]),
        ('(c) alignment: base, SFT, DPO', [('base', al('Llama_base'), BL, '-', 1.6), ('SFT', al('Llama_SFT'), OR, '-', 1.2), ('DPO', al('Llama_DPO'), GN, '--', 1.2)]),
        ('(d) distillation: student keeps its path', [('Llama-3 (student init)', ds('Llama_3'), BL, '-', 1.6), ('Llama-3.1 (teacher)', ds('Llama_3.1'), OR, '-', 1.4),
                                                       ('distilled student', ds('Llama_3_distilled'), PU, '--', 1.8)]),
    ]
    more_panels = [
        ('(a) Qwen-3 4B: base, instruct, thinking', [('base', sq('Qwen_Qwen3_4B'), BL, '-', 1.5), ('instruct', sq('Qwen_Qwen3_4B_Instruct_2507'), OR, '-', 1.2),
                                                     ('thinking', sq('Qwen_Qwen3_4B_Thinking_2507'), GN, '--', 1.2)]),
        ('(b) Gemma-3 1B: PT and IT', [('PT', sq('google_gemma_3_1b_pt'), BL, '-', 1.5), ('IT', sq('google_gemma_3_1b_it'), OR, '--', 1.3)]),
        ('(c) Llama-3 8B: base, instruct, Tulu-3.1', [('base', k25('meta_llama_Meta_Llama_3_8B'), BL, '-', 1.5), ('instruct', k25('meta_llama_Meta_Llama_3_8B_Instruct'), OR, '-', 1.2),
                                                     ('Tulu-3.1', k25('allenai_Llama_3.1_Tulu_3_8B'), GN, '--', 1.2)]),
        ('(d) alignment on Qwen-3 4B', [('base', al('Qwen_base'), BL, '-', 1.5), ('SFT', al('Qwen_SFT'), OR, '-', 1.2), ('DPO', al('Qwen_DPO'), GN, '--', 1.2)]),
        ('(e) merging: parents AF, ME and child', [('parent AF', mg('finetuned_AF'), BL, '-', 1.4), ('parent ME', mg('finetuned_ME'), OR, '-', 1.4), ('merged AF+ME', mg('merged_AF_ME'), PU, '--', 1.8)]),
        ('(f) distillation: Qwen2 into Qwen2.5 teacher', [('Qwen2 (student init)', ds('Qwen_2'), BL, '-', 1.5), ('Qwen2.5 (teacher)', ds('Qwen_2.5'), OR, '-', 1.3),
                                                          ('distilled student', ds('Qwen_2_distilled'), PU, '--', 1.8)]),
    ]
    for name, panels, rows, h, wd in (('fig_3d_ops', main_panels, 2, 5.0, 6.4), ('fig_3d_more', more_panels, 2, 4.6, 7.2)):
        cols = len(panels) // rows
        fig = plt.figure(figsize=(wd, h))
        for i, (ttl, ser) in enumerate(panels):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            draw(ax, ser, ttl)
        fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.02, wspace=0.0, hspace=0.08)
        fig.savefig(f'{FIG}/{name}.png')
        plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ng', default=None)
    ap.add_argument('--nofig', action='store_true')
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    E = exports()
    part_exports(E)
    P, R = load_all(a.ng)
    part_dash(P, R, a.ng)
    part_offline(E)
    part_depth_robustness(P)
    part_metadata(P)
    part_combine(P)
    RA = part_recursive()
    SD = part_sftdpo()
    part_amplitude()
    part_settle(E)
    part_recursive_effort()
    part_zoo()
    if not a.nofig:
        figures(E, P)
        figures3d(P)
        figures_alignment(P)
        figures_recursive(RA, SD)
        figures_more(RA, P)
    with open(os.path.join(HERE, 'tables.tex'), 'w', encoding='utf8') as f:
        f.write('% generated by analysis.py, do not edit\n' + '\n'.join(TEX))
    slim = {k: v for k, v in OUT.items() if not k.startswith('_')}
    json.dump(slim, open(os.path.join(RES, 'derived.json'), 'w'), indent=1, default=float)
    print('ok:', len(TEX), 'macros;', len(slim), 'result blocks')


if __name__ == '__main__':
    main()
