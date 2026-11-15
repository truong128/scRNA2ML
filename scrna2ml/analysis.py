"""Secondary analyses: paired random-effects meta-analysis, batch (cohort) predictability, feature-set
ablations, subgroup performance and feature-selection stability."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

from . import ml


def _bh(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty_like(q)
    res[order] = np.minimum(ranked, 1)
    out[ok] = res
    return out


def random_effects(effects, ses):
    """DerSimonian-Laird pooling over cohorts (rows) for many features (columns)."""
    w = 1 / ses ** 2
    fixed = np.nansum(w * effects, axis=0) / np.nansum(w, axis=0)
    k = np.sum(np.isfinite(effects), axis=0)
    q = np.nansum(w * (effects - fixed) ** 2, axis=0)
    c = np.nansum(w, axis=0) - np.nansum(w ** 2, axis=0) / np.nansum(w, axis=0)
    tau2 = np.maximum(0, (q - (k - 1)) / c)
    wr = 1 / (ses ** 2 + tau2)
    pooled = np.nansum(wr * effects, axis=0) / np.nansum(wr, axis=0)
    se = np.sqrt(1 / np.nansum(wr, axis=0))
    z = pooled / se
    i2 = np.where(q > 0, np.maximum(0, (q - (k - 1)) / np.where(q > 0, q, 1)), 0)
    return pooled, se, z, 2 * stats.norm.sf(np.abs(z)), tau2, i2, k


def paired_meta(cfg, compartment, min_donors=3, min_mean_logcpm=1.0):
    units, X, _ = ml.load(cfg, compartment)
    genes = [c for c in X if c.startswith('gene::')]
    keep_genes = [g for g in genes if X[g].mean() >= min_mean_logcpm]
    cols = keep_genes + [c for c in X if c.startswith('prop::')]
    tum = X[cols][units.tissue.eq('Tumor').to_numpy()].set_axis(units.donor_id[units.tissue.eq('Tumor')].to_numpy())
    nor = X[cols][units.tissue.eq('Normal').to_numpy()].set_axis(units.donor_id[units.tissue.eq('Normal')].to_numpy())
    diff = tum - nor.loc[tum.index]
    cohort = units.drop_duplicates('donor_id').set_index('donor_id').cohort.loc[diff.index]
    eff, se, per = [], [], []
    for c, d in diff.groupby(cohort.to_numpy()):
        if len(d) < min_donors:
            continue
        m, s = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        s = s.where(s > 0, np.nan)
        eff.append(m.to_numpy())
        se.append(s.to_numpy())
        per.append(pd.DataFrame({'feature': cols, 'cohort': c, 'n_donors': len(d), 'mean_paired_difference': m.to_numpy(), 'se': s.to_numpy(),
                                 'p_wilcoxon': [stats.wilcoxon(d[f]).pvalue if (d[f] != 0).any() else 1.0 for f in cols]}))
    eff, se = np.array(eff), np.array(se)
    eff[~np.isfinite(se)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pooled, pse, z, p, tau2, i2, k = random_effects(eff, se)
    signs = np.sign(eff)
    res = pd.DataFrame({'feature': cols, 'pooled_difference': pooled, 'se': pse, 'z': z, 'p': p, 'tau2': tau2, 'I2': i2, 'cohorts': k,
                        'cohorts_up': np.nansum(signs > 0, axis=0), 'cohorts_down': np.nansum(signs < 0, axis=0)})
    res['type'] = np.where(res.feature.str.startswith('prop::'), 'proportion', 'gene')
    res['fdr'] = np.nan
    for t in ('gene', 'proportion'):
        m = res.type == t
        res.loc[m, 'fdr'] = _bh(res.loc[m, 'p'])
    res['log2_fold_change'] = np.where(res.type == 'gene', res.pooled_difference / np.log(2), np.nan)
    out = Path(cfg['study']['results_dir']) / 'analysis' / compartment
    out.mkdir(parents=True, exist_ok=True)
    res.sort_values('p').to_csv(out / 'paired_meta_analysis.tsv', sep='\t', index=False)
    pd.concat(per).to_csv(out / 'paired_differences_per_cohort.tsv', sep='\t', index=False)
    return res


def cohort_predictability(cfg, compartment):
    """How well can the source cohort be predicted from the same features? (batch-effect magnitude)"""
    units, X, _ = ml.load(cfg, compartment)
    groups = units.donor_id.to_numpy()
    folds = ml.donor_folds(groups, 5, cfg['study']['seed'])
    cohort = units.cohort.to_numpy()
    out = {}
    for name, cols in [('all_features', X.columns), ('proportions_only', [c for c in X if c.startswith('prop::')])]:
        k = min(50, len(cols) - 1) if len(cols) > 50 else 'all'
        pipe = ml.pipeline(LogisticRegression(max_iter=5000, class_weight='balanced'), k=k)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pred = cross_val_predict(pipe, X[cols], cohort, cv=folds)
        out[name] = {'balanced_accuracy': balanced_accuracy_score(cohort, pred), 'chance': 1 / len(set(cohort)), 'n_cohorts': len(set(cohort))}
    path = Path(cfg['study']['results_dir']) / 'analysis' / compartment
    path.mkdir(parents=True, exist_ok=True)
    (path / 'cohort_predictability.json').write_text(json.dumps(out, indent=2))
    return out


def feature_set_ablation(cfg, compartment, scheme='loco'):
    """Primary model under the same nested scheme on proportions only, genes only and all features;
    plus each single proportion as a direction-fixed (training-cohort) univariate score."""
    units, X, y = ml.load(cfg, compartment)
    outer, inner = ml.make_splits(scheme, units, cfg)
    model = ml.registry()[0][ml.PRIMARY]
    rows = []
    sets = {'proportions_only': [c for c in X if c.startswith('prop::')], 'genes_only': [c for c in X if c.startswith('gene::')], 'all_features': list(X.columns)}
    for name, cols in sets.items():
        ks = ml.k_grid(cfg, len(cols))
        pred, score = np.empty(len(y)), np.empty(len(y))
        for (tr, te), inn in zip(outer, inner):
            f = ml.fit_one(ml.PRIMARY, model, X[cols], y, tr, te, inn, ks)
            pred[te], score[te] = f['pred'], f['score']
        frame = pd.DataFrame({'donor': units.donor_id, 'truth': y, 'prediction': pred.astype(int), 'score': score, 'fold': 0, 'probability': True})
        for i, (_, te) in enumerate(outer):
            frame.loc[te, 'fold'] = i
        r = ml.summarize(frame, cfg['ml']['bootstrap'], cfg['study']['seed'])
        rows.append(dict(feature_set=name, n_features=len(cols), **r))
    for col in sets['proportions_only']:
        score = np.empty(len(y))
        for tr, te in outer:
            sign = np.sign(X[col].iloc[tr][y[tr] == 1].mean() - X[col].iloc[tr][y[tr] == 0].mean()) or 1
            score[te] = sign * X[col].iloc[te]
        frame = pd.DataFrame({'donor': units.donor_id, 'truth': y, 'score': score, 'fold': 0})
        for i, (_, te) in enumerate(outer):
            frame.loc[te, 'fold'] = i
        rows.append({'feature_set': f'single::{col}', 'n_features': 1, 'n_donors': units.donor_id.nunique(),
                     'auroc': roc_auc_score(y, score), 'auroc_mean_over_folds': np.mean([roc_auc_score(f.truth, f.score) for _, f in frame.groupby('fold')]),
                     'paired_concordance': ml.paired_concordance(frame)})
    res = pd.DataFrame(rows)
    out = Path(cfg['study']['results_dir']) / 'analysis' / compartment
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f'feature_set_ablation_{scheme}.tsv', sep='\t', index=False)
    return res


def subgroups(cfg, compartment, scheme='loco', models=(ml.PRIMARY, 'nested_algorithm_selection')):
    units = pd.read_csv(Path(cfg['study']['results_dir']) / 'features' / compartment / 'units.tsv', sep='\t')
    oof = pd.read_csv(Path(cfg['study']['results_dir']) / 'ml' / compartment / scheme / 'oof_predictions.tsv', sep='\t')
    oof = oof.merge(units[['unit_id', 'histology', 'enrichment']], on='unit_id', validate='many_to_one')
    rows = []
    for model in models:
        d = oof[oof.model == model]
        for var in ('histology', 'enrichment', 'cohort'):
            for level, s in d.groupby(var):
                r = {'model': model, 'variable': var, 'level': level, 'n_donors': s.donor.nunique(),
                     'balanced_accuracy': balanced_accuracy_score(s.truth, s.prediction)}
                if s.score.notna().all() and s.truth.nunique() == 2:
                    if model != 'nested_algorithm_selection' or var == 'cohort':
                        r['auroc'] = roc_auc_score(s.truth, s.score)
                    r['paired_concordance'] = ml.paired_concordance(s)
                rows.append(r)
    res = pd.DataFrame(rows)
    out = Path(cfg['study']['results_dir']) / 'analysis' / compartment
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f'subgroups_{scheme}.tsv', sep='\t', index=False)
    return res


def stability(cfg, compartment, scheme='loco', model=ml.PRIMARY):
    sel = json.loads((Path(cfg['study']['results_dir']) / 'ml' / compartment / scheme / 'selected_features.json').read_text())[model]
    counts = pd.Series([f for fold in sel.values() for f in fold]).value_counts()
    res = pd.DataFrame({'feature': counts.index, 'folds_selected': counts.to_numpy(), 'fraction_of_folds': counts.to_numpy() / len(sel)})
    out = Path(cfg['study']['results_dir']) / 'analysis' / compartment
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f'selection_stability_{scheme}.tsv', sep='\t', index=False)
    return res
