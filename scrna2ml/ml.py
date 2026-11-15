"""Nested, donor-safe and cohort-safe evaluation of tumour vs adjacent-normal classifiers.

Schemes
  loco            outer: leave one cohort out; inner: leave one of the remaining cohorts out (model + k selection)
  loco_centered   as loco, features centred within each cohort (label-free, transductive batch adjustment)
  pooled          outer: 5-fold donor-grouped CV over all cohorts; inner: 3-fold donor-grouped CV
  within::<GSE>   outer: 5-fold donor-grouped CV inside one cohort; inner: 3-fold donor-grouped CV
All feature filtering, selection, scaling and model fitting happen inside training partitions.
Both specimens of a donor always fall in the same partition.
"""
import json
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import registry

PRIMARY = 'logistic_regression'
CLASSES = ['Normal', 'Tumor']


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def load(cfg, compartment):
    root = Path(cfg['study']['results_dir']) / 'features' / compartment
    units = pd.read_csv(root / 'units.tsv', sep='\t')
    X = pd.read_csv(root / 'X.tsv.gz', sep='\t')
    if len(units) != len(X):
        raise ValueError('units/X misaligned')
    y = (units.label == 'Tumor').astype(int).to_numpy()
    return units, X, y


def pipeline(model, k=10):
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('variance', VarianceThreshold()),
                     ('select', SelectKBest(f_classif, k=k)), ('scale', StandardScaler()), ('model', clone(model))])


def donor_folds(donors, n_splits, seed):
    """Donor-grouped folds. Each donor contributes one tumour and one normal unit, so every fold is class-balanced."""
    unique = np.array(sorted(set(donors)))
    n_splits = min(n_splits, len(unique))
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(unique), n_splits)
    folds = []
    for part in parts:
        te = np.flatnonzero(np.isin(donors, part))
        tr = np.flatnonzero(~np.isin(donors, part))
        folds.append((tr, te))
    return folds


def cohort_folds(cohorts):
    return [(np.flatnonzero(cohorts != c), np.flatnonzero(cohorts == c)) for c in sorted(set(cohorts))]


def make_splits(scheme, units, cfg):
    donors = units.donor_id.to_numpy()
    cohorts = units.cohort.to_numpy()
    seed = cfg['study']['seed']
    if scheme.startswith('loco'):
        if len(set(cohorts)) < 3:
            raise ValueError('Leave-one-cohort-out with inner cohort CV needs at least 3 cohorts')
        outer = cohort_folds(cohorts)
        inner = []
        for tr, _ in outer:
            inner.append(cohort_folds(cohorts[tr]))
    else:
        outer = donor_folds(donors, cfg['ml']['outer_max_splits'], seed)
        inner = [donor_folds(donors[tr], cfg['ml']['inner_max_splits'], seed + 1) for tr, _ in outer]
    y = (units.label == 'Tumor').astype(int).to_numpy()
    for (tr, te), inn in zip(outer, inner):
        assert not set(donors[tr]) & set(donors[te])
        assert set(y[tr]) == {0, 1} and set(y[te]) == {0, 1}
        for a, b in inn:
            assert not set(donors[tr][a]) & set(donors[tr][b])
            assert set(y[tr][a]) == {0, 1} and set(y[tr][b]) == {0, 1}
    return outer, inner


def k_grid(cfg, n_features):
    ks = [k for k in cfg['ml']['select_k'] if k < n_features]
    return ks or ['all']


def score_of(est, X):
    if hasattr(est, 'predict_proba'):
        try:
            return est.predict_proba(X)[:, 1], True
        except Exception:
            pass
    if hasattr(est, 'decision_function'):
        return np.asarray(est.decision_function(X)).reshape(-1), False
    return None, False


def fit_one(name, model, X, y, tr, te, inner, ks):
    t0 = time.time()
    search = GridSearchCV(pipeline(model), {'select__k': ks}, cv=inner, scoring='balanced_accuracy', refit=True, n_jobs=1, error_score='raise')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.fit(X.iloc[tr], y[tr])
        pred = np.asarray(search.predict(X.iloc[te])).reshape(-1).astype(int)
        score, prob = score_of(search, X.iloc[te])
    best = search.best_estimator_
    selected = X.columns[best['variance'].get_support()][best['select'].get_support()].tolist()
    return {'model': name, 'pred': pred, 'score': score, 'probability': prob, 'inner_score': float(search.best_score_),
            'k': search.best_params_['select__k'], 'features': selected, 'seconds': time.time() - t0}


def _safe_fit(name, model, X, y, tr, te, inner, ks):
    try:
        return fit_one(name, model, X, y, tr, te, inner, ks)
    except Exception as e:
        return {'model': name, 'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()}


def bootstrap_ci(frame, n_boot, seed, with_auc):
    rng = np.random.default_rng(seed)
    donors = frame.donor.unique()
    rows = {d: np.flatnonzero(frame.donor.to_numpy() == d) for d in donors}
    truth, pred = frame.truth.to_numpy(), frame.prediction.to_numpy()
    score = frame.score.to_numpy() if with_auc else None
    ba, auc = [], []
    for _ in range(n_boot):
        ix = np.concatenate([rows[d] for d in rng.choice(donors, len(donors), replace=True)])
        ba.append(balanced_accuracy_score(truth[ix], pred[ix]))
        if with_auc:
            auc.append(roc_auc_score(truth[ix], score[ix]))
    out = {'balanced_accuracy_ci_low': np.quantile(ba, .025), 'balanced_accuracy_ci_high': np.quantile(ba, .975)}
    if with_auc:
        out.update(auroc_ci_low=np.quantile(auc, .025), auroc_ci_high=np.quantile(auc, .975))
    return out


def paired_concordance(frame):
    wide = frame.pivot_table(index='donor', columns='truth', values='score', aggfunc='first')
    diff = wide[1] - wide[0]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def summarize(frame, n_boot, seed, pooled_auc=True):
    truth, pred = frame.truth.to_numpy(), frame.prediction.to_numpy()
    row = {'n_units': len(frame), 'n_donors': frame.donor.nunique(),
           'balanced_accuracy': balanced_accuracy_score(truth, pred), 'macro_f1': f1_score(truth, pred, average='macro'),
           'mcc': matthews_corrcoef(truth, pred), 'sensitivity_tumor': float((pred[truth == 1] == 1).mean()),
           'specificity_tumor': float((pred[truth == 0] == 0).mean())}
    has_score = frame.score.notna().all()
    if has_score:
        row['paired_concordance'] = paired_concordance(frame)
        per_fold = [roc_auc_score(d.truth, d.score) for _, d in frame.groupby('fold') if d.truth.nunique() == 2]
        row['auroc_mean_over_folds'] = float(np.mean(per_fold))
        if pooled_auc:
            row['auroc'] = roc_auc_score(truth, frame.score)
        if frame.probability.all():
            row['brier'] = brier_score_loss(truth, frame.score.clip(0, 1))
    row.update(bootstrap_ci(frame, n_boot, seed, has_score and pooled_auc))
    return row


def run_scheme(cfg, compartment, scheme, models=None, n_jobs=8):
    units, X, y = load(cfg, compartment)
    if scheme.startswith('within::'):
        keep = (units.cohort == scheme.split('::', 1)[1]).to_numpy()
        units, X, y = units[keep].reset_index(drop=True), X[keep].reset_index(drop=True), y[keep]
    if scheme == 'loco_centered':
        X = X - X.groupby(units.cohort.to_numpy()).transform('mean')
    out = Path(cfg['study']['results_dir']) / 'ml' / compartment / scheme.replace('::', '_')
    out.mkdir(parents=True, exist_ok=True)
    outer, inner = make_splits(scheme, units, cfg)
    ks = k_grid(cfg, X.shape[1])
    all_models, unavailable = registry()
    models = {k: v for k, v in all_models.items() if models is None or k in models}
    dump(out / 'splits.json', [{'fold': i, 'test_cohorts': sorted(set(units.cohort[te])), 'train_donors': units.donor_id[tr].unique().tolist(),
                                'test_donors': units.donor_id[te].unique().tolist(),
                                'inner': [units.donor_id[tr].iloc[b].unique().tolist() for a, b in inner[i]]} for i, (tr, te) in enumerate(outer)])
    jobs = [(fold, name, model) for fold in range(len(outer)) for name, model in models.items()]
    print(f'[ml] {compartment}/{scheme}: {len(units)} units, {units.donor_id.nunique()} donors, {len(outer)} outer folds, {len(models)} models, k={ks}', flush=True)
    fits = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_safe_fit)(name, model, X, y, outer[fold][0], outer[fold][1], inner[fold], ks) for fold, name, model in jobs)
    rows, errors, selected, winners = [], [], {}, []
    for (fold, name, _), fit in zip(jobs, fits):
        if 'error' in fit:
            errors.append({'fold': fold, 'model': name, 'error': fit['error'], 'traceback': fit['traceback']})
            continue
        te = outer[fold][1]
        selected.setdefault(name, {})[fold] = fit['features']
        for j, ix in enumerate(te):
            rows.append({'model': name, 'fold': fold, 'row': int(ix), 'unit_id': units.unit_id[ix], 'donor': units.donor_id[ix],
                         'cohort': units.cohort[ix], 'truth': int(y[ix]), 'prediction': int(fit['pred'][j]),
                         'score': float(fit['score'][j]) if fit['score'] is not None else np.nan, 'probability': fit['probability'],
                         'inner_score': fit['inner_score'], 'k': fit['k']})
    oof = pd.DataFrame(rows)
    for fold in range(len(outer)):
        cand = oof[(oof.fold == fold) & (oof.model != 'dummy_prior')].drop_duplicates('model')
        order = {m: i for i, m in enumerate(models)}
        cand = cand.assign(order=cand.model.map(order)).sort_values(['inner_score', 'order'], ascending=[False, True])
        if len(cand):
            win = cand.iloc[0].model
            winners.append({'fold': fold, 'model': win, 'inner_score': float(cand.iloc[0].inner_score)})
            oof = pd.concat([oof, oof[(oof.fold == fold) & (oof.model == win)].assign(model='nested_algorithm_selection')], ignore_index=True)
    oof.to_csv(out / 'oof_predictions.tsv', sep='\t', index=False)
    metrics, per_fold = [], []
    n_boot, seed = cfg['ml']['bootstrap'], cfg['study']['seed']
    for name, d in oof.groupby('model', sort=False):
        complete = d.row.nunique() == len(units) and len(d) == len(units)
        row = {'model': name, 'complete_oof': complete, 'failed_folds': sum(e['model'] == name for e in errors)}
        if complete:
            row.update(summarize(d, n_boot, seed, pooled_auc=name != 'nested_algorithm_selection'))
        metrics.append(row)
        for fold, f in d.groupby('fold'):
            r = {'model': name, 'fold': fold, 'test': '|'.join(sorted(f.cohort.unique())), 'n_donors': f.donor.nunique(),
                 'balanced_accuracy': balanced_accuracy_score(f.truth, f.prediction)}
            if f.score.notna().all():
                r['auroc'] = roc_auc_score(f.truth, f.score)
                r['paired_concordance'] = paired_concordance(f)
            per_fold.append(r)
    for name in models:
        if name not in oof.model.values:
            metrics.append({'model': name, 'complete_oof': False, 'failed_folds': len(outer)})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / 'metrics.tsv', sep='\t', index=False)
    pd.DataFrame(per_fold).to_csv(out / 'metrics_per_fold.tsv', sep='\t', index=False)
    dump(out / 'selected_features.json', selected)
    dump(out / 'nested_selection.json', winners)
    dump(out / 'errors.json', errors)
    dump(out / 'status.json', {'status': 'complete_with_errors' if errors else 'complete', 'scheme': scheme, 'compartment': compartment,
                               'units': len(units), 'donors': int(units.donor_id.nunique()), 'cohorts': sorted(units.cohort.unique()),
                               'outer_folds': len(outer), 'k_grid': ks, 'models': len(models), 'unavailable': unavailable,
                               'failed_fits': len(errors), 'primary_prespecified': PRIMARY})
    return metrics


def permutation_test(cfg, compartment, scheme, n_perm=None, n_jobs=8):
    """Null distribution of the prespecified primary model: tumour/normal labels swapped at random within each donor,
    then the complete nested procedure is repeated."""
    units, X, y = load(cfg, compartment)
    outer, inner = make_splits(scheme, units, cfg)
    ks = k_grid(cfg, X.shape[1])
    model = registry()[0][PRIMARY]
    n_perm = n_perm or cfg['ml']['permutations']
    donors = units.donor_id.to_numpy()

    def one(perm):
        yp = y.copy()
        if perm is not None:
            rng = np.random.default_rng([cfg['study']['seed'], perm])
            for d in np.unique(donors):
                if rng.random() < .5:
                    ix = np.flatnonzero(donors == d)
                    yp[ix] = 1 - yp[ix]
        pred, score = np.empty(len(y)), np.empty(len(y))
        for (tr, te), inn in zip(outer, inner):
            f = fit_one(PRIMARY, model, X, yp, tr, te, inn, ks)
            pred[te], score[te] = f['pred'], f['score']
        return balanced_accuracy_score(yp, pred), roc_auc_score(yp, score)

    observed = one(None)
    null = Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in range(n_perm))
    null = np.array(null)
    res = {'scheme': scheme, 'compartment': compartment, 'n_permutations': n_perm,
           'observed_balanced_accuracy': observed[0], 'observed_auroc': observed[1],
           'p_balanced_accuracy': float((1 + (null[:, 0] >= observed[0]).sum()) / (n_perm + 1)),
           'p_auroc': float((1 + (null[:, 1] >= observed[1]).sum()) / (n_perm + 1)),
           'null_balanced_accuracy_mean': float(null[:, 0].mean()), 'null_auroc_mean': float(null[:, 1].mean())}
    out = Path(cfg['study']['results_dir']) / 'ml' / compartment / scheme
    pd.DataFrame(null, columns=['balanced_accuracy', 'auroc']).to_csv(out / 'permutation_null.tsv', sep='\t', index=False)
    dump(out / 'permutation_test.json', res)
    return res


def final_model(cfg, compartment):
    """Refit the prespecified primary model on all cohorts (inner leave-cohort-out tuning of k) for interpretation and reuse."""
    units, X, y = load(cfg, compartment)
    inner = cohort_folds(units.cohort.to_numpy())
    search = GridSearchCV(pipeline(registry()[0][PRIMARY]), {'select__k': k_grid(cfg, X.shape[1])}, cv=inner, scoring='balanced_accuracy', refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.fit(X, y)
    best = search.best_estimator_
    feats = X.columns[best['variance'].get_support()][best['select'].get_support()]
    coef = pd.DataFrame({'feature': feats, 'coefficient': best['model'].coef_.ravel()}).sort_values('coefficient', key=np.abs, ascending=False)
    out = Path(cfg['study']['results_dir']) / 'ml' / compartment
    coef.to_csv(out / 'final_model_coefficients.tsv', sep='\t', index=False)
    joblib.dump({'pipeline': best, 'features': X.columns.tolist(), 'classes': CLASSES, 'positive_class': 'Tumor',
                 'compartment': compartment, 'k': search.best_params_['select__k'], 'research_only': True}, out / 'final_model.joblib', compress=3)
    return coef
