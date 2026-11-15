"""Model explanation and association structure of the tumour-normal signal.

shap_loco: out-of-fold SHAP values under leave-one-cohort-out. Each outer fold is refitted exactly as in the
benchmark and explained only on its held-out cohort, so no explanation uses a specimen the model was trained on.

knowledge_graph: an association graph linking features to the tumour outcome node. The data-only layer uses
within-donor paired differences; the integrated layer adds out-of-fold SHAP and LOCO selection frequency to the
feature-outcome edges. Edges are associations, not causal relations.
"""
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from scipy.stats import t as tdist
from sklearn.model_selection import GridSearchCV

from . import ml
from .analysis import _bh

SHAP_MODELS = ('logistic_regression', 'extra_trees')
OUTCOME = 'Tumour'
COMPOSITE = {'pagerank': .25, 'personalized_pagerank': .15, 'proximity': .15, 'direct': .35, 'betweenness': .10}


def _fit_fold(model, X, y, tr, inner, ks):
    search = GridSearchCV(ml.pipeline(model), {'select__k': ks}, cv=inner, scoring='balanced_accuracy', refit=True, n_jobs=1, error_score='raise')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        search.fit(X.iloc[tr], y[tr])
    return search.best_estimator_


def _explain(name, est, Ztr, Zte):
    model = est['model']
    if name == 'logistic_regression':
        explainer = shap.LinearExplainer(model, Ztr)
        vals = np.asarray(explainer.shap_values(Zte))
        output = model.decision_function(Zte)
        base = float(np.ravel(explainer.expected_value)[0])
    else:
        explainer = shap.TreeExplainer(model)
        vals = np.asarray(explainer.shap_values(Zte, check_additivity=False))
        vals = vals[..., 1] if vals.ndim == 3 else vals
        output = model.predict_proba(Zte)[:, 1]
        base = float(np.ravel(explainer.expected_value)[-1])
    return vals, float(np.abs(base + vals.sum(axis=1) - output).max())


def shap_loco(cfg, compartment):
    results = Path(cfg['study']['results_dir'])
    units, X, y = ml.load(cfg, compartment)
    outer, inner = ml.make_splits('loco', units, cfg)
    registry = ml.registry()[0]
    ks = ml.k_grid(cfg, X.shape[1])
    saved = pd.read_csv(results / 'ml' / compartment / 'loco' / 'oof_predictions.tsv', sep='\t')
    out = results / 'analysis' / compartment
    out.mkdir(parents=True, exist_ok=True)
    summary, checks = [], {}
    for name in SHAP_MODELS:
        S = np.zeros(X.shape)
        Z = np.full(X.shape, np.nan)
        selected = np.zeros(X.shape[1], int)
        additivity, reproduction = [], []
        ref = saved[saved.model == name].set_index('unit_id').score
        for (tr, te), inn in zip(outer, inner):
            est = _fit_fold(registry[name], X, y, tr, inn, ks)
            pre = est[:-1]
            cols = X.columns.get_indexer(X.columns[est['variance'].get_support()][est['select'].get_support()])
            Ztr, Zte = pre.transform(X.iloc[tr]), pre.transform(X.iloc[te])
            vals, err = _explain(name, est, Ztr, Zte)
            S[np.ix_(te, cols)] = vals
            Z[np.ix_(te, cols)] = Zte
            selected[cols] += 1
            additivity.append(err)
            score = est.predict_proba(X.iloc[te])[:, 1]
            reproduction.append(float(np.abs(score - ref.loc[units.unit_id.iloc[te]].to_numpy()).max()))
        checks[name] = {'max_additivity_error': max(additivity), 'max_score_difference_vs_saved': max(reproduction)}
        if checks[name]['max_score_difference_vs_saved'] > 1e-6:
            raise RuntimeError(f'{name}: refitted LOCO scores do not reproduce saved predictions')
        pd.DataFrame(S, columns=X.columns).assign(unit_id=units.unit_id).to_csv(out / f'shap_loco_{name}.tsv.gz', sep='\t', index=False)
        pd.DataFrame(Z, columns=X.columns).assign(unit_id=units.unit_id).to_csv(out / f'shap_loco_{name}_values.tsv.gz', sep='\t', index=False)
        direction = []
        for j in range(X.shape[1]):
            m = ~np.isnan(Z[:, j])
            direction.append(spearmanr(Z[m, j], S[m, j])[0] if m.sum() > 2 and np.ptp(Z[m, j]) > 0 else np.nan)
        summary.append(pd.DataFrame({'feature': X.columns, 'model': name, 'mean_abs_shap': np.abs(S).mean(axis=0),
                                     'folds_selected': selected, 'value_shap_correlation': direction}))
    summary = pd.concat(summary)
    summary['rank'] = summary.groupby('model').mean_abs_shap.rank(ascending=False, method='min').astype(int)
    summary.sort_values(['model', 'rank']).to_csv(out / 'shap_loco_summary.tsv', sep='\t', index=False)
    wide = summary.pivot(index='feature', columns='model', values='mean_abs_shap')
    top = wide[(wide > 0).all(axis=1)]
    checks['rank_correlation_between_models'] = float(spearmanr(top[SHAP_MODELS[0]], top[SHAP_MODELS[1]])[0])
    checks['features_with_nonzero_shap_in_both'] = int(len(top))
    checks['top20_overlap'] = int(len(set(wide[SHAP_MODELS[0]].nlargest(20).index) & set(wide[SHAP_MODELS[1]].nlargest(20).index)))
    ml.dump(out / 'shap_loco_checks.json', checks)
    return summary, checks


def _paired_differences(units, X, y):
    key = units.donor_id.to_numpy()
    T = X[y == 1].set_axis(key[y == 1])
    N = X[y == 0].set_axis(key[y == 0])
    D = T - N.loc[T.index]
    cohort = units.drop_duplicates('donor_id').set_index('donor_id').cohort.loc[D.index]
    return D, cohort


def _minmax(s):
    s = pd.Series(s, dtype=float)
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0


def _graph_scores(G):
    feats = [n for n in G if n != OUTCOME]
    for _, _, d in G.edges(data=True):
        d['distance'] = 1 / (d['weight'] + 1e-6)
    pr = nx.pagerank(G, weight='weight')
    ppr = nx.pagerank(G, weight='weight', personalization={OUTCOME: 1})
    dist = nx.single_source_dijkstra_path_length(G, OUTCOME, weight='distance')
    btw = nx.betweenness_centrality(G, weight='distance')
    direct = {n: G[n][OUTCOME]['weight'] if G.has_edge(n, OUTCOME) else 0 for n in feats}
    parts = pd.DataFrame({'pagerank': [pr[n] for n in feats], 'personalized_pagerank': [ppr[n] for n in feats],
                          'proximity': [1 / dist[n] if n in dist else 0 for n in feats], 'direct': [direct[n] for n in feats],
                          'betweenness': [btw[n] for n in feats], 'path_distance': [dist.get(n, np.inf) for n in feats]}, index=feats)
    composite = sum(w * _minmax(parts[c]) for c, w in COMPOSITE.items())
    return parts.assign(composite=composite)


def knowledge_graph(cfg, compartment, min_rho=0.5, max_q=0.01):
    results = Path(cfg['study']['results_dir'])
    out = results / 'analysis' / compartment
    units, X, y = ml.load(cfg, compartment)
    D, cohort = _paired_differences(units, X, y)
    meta = pd.read_csv(out / 'paired_meta_analysis.tsv', sep='\t').set_index('feature')
    concordant = (meta.cohorts_up == meta.cohorts) | (meta.cohorts_down == meta.cohorts)
    significant = (meta.fdr < 0.05) & concordant
    props = [c for c in X if c.startswith('prop::')]
    nodes = sorted(set(meta.index[significant]) | set(props))
    dz = D[nodes].mean() / D[nodes].std(ddof=1)
    # Removing cohort means of the paired differences keeps feature-feature edges from reflecting cohort-level shifts.
    Dc = (D[nodes] - D[nodes].groupby(cohort.to_numpy()).transform('mean')).to_numpy()
    rho = spearmanr(Dc).statistic
    iu = np.triu_indices(len(nodes), 1)
    n = len(D)
    t = rho[iu] * np.sqrt((n - 2) / np.clip(1 - rho[iu] ** 2, 1e-12, None))
    q = _bh(2 * tdist.sf(np.abs(t), n - 2))
    keep = (np.abs(rho[iu]) >= min_rho) & (q < max_q)
    pairs = [(nodes[i], nodes[j], float(rho[i, j])) for i, j, k in zip(*iu, keep) if k]

    shap_summary = pd.read_csv(out / 'shap_loco_summary.tsv', sep='\t')
    shap_mean = shap_summary.pivot(index='feature', columns='model', values='mean_abs_shap')
    shap_norm = (shap_mean / shap_mean.max()).mean(axis=1).reindex(nodes).fillna(0)
    stability = pd.read_csv(out / 'selection_stability_loco.tsv', sep='\t').set_index('feature').fraction_of_folds.reindex(nodes).fillna(0)

    direct_data = (dz.abs() / dz.abs().max()).where(significant.reindex(nodes, fill_value=False), 0)
    layers = {'data': direct_data, 'integrated': (direct_data + shap_norm + stability) / 3}
    edges, scores = [], {}
    for layer, direct in layers.items():
        G = nx.Graph()
        G.add_nodes_from(nodes + [OUTCOME])
        for f, w in direct.items():
            if w > 0:
                G.add_edge(f, OUTCOME, weight=float(w))
                edges.append({'layer': layer, 'source': f, 'target': OUTCOME, 'kind': 'feature-outcome', 'weight': float(w), 'sign': int(np.sign(dz[f]))})
        for a, b, r in pairs:
            G.add_edge(a, b, weight=abs(r))
            edges.append({'layer': layer, 'source': a, 'target': b, 'kind': 'feature-feature', 'weight': abs(r), 'sign': int(np.sign(r))})
        G.remove_nodes_from([v for v in list(G) if G.degree(v) == 0])
        scores[layer] = _graph_scores(G)
    table = pd.DataFrame({'type': ['proportion' if f.startswith('prop::') else 'gene' for f in nodes], 'paired_dz': dz.reindex(nodes),
                          'meta_fdr': meta.fdr.reindex(nodes), 'concordant_significant': significant.reindex(nodes, fill_value=False),
                          'shap_normalised': shap_norm, 'selection_frequency': stability}, index=nodes)
    # The cell-type proportion whose paired shift a gene tracks most closely separates composition-linked genes from state changes.
    R = pd.DataFrame(rho, index=nodes, columns=nodes)[props]
    best = R.abs().drop(index=props).idxmax(axis=1)
    table['linked_proportion'] = best.str.replace('prop::', '')
    table['rho_with_linked_proportion'] = pd.Series({g: R.loc[g, best[g]] for g in best.index})
    for layer, s in scores.items():
        s = s.reindex(nodes)
        table[f'composite_{layer}'] = s.composite
        table[f'rank_{layer}'] = s.composite.rank(ascending=False, method='min')
        table[f'path_distance_{layer}'] = s.path_distance
        for c in COMPOSITE:
            table[f'{c}_{layer}'] = s[c]
    table = table.sort_values('composite_integrated', ascending=False).rename_axis('feature').reset_index()
    table.to_csv(out / 'knowledge_graph_nodes.tsv', sep='\t', index=False)
    pd.DataFrame(edges).to_csv(out / 'knowledge_graph_edges.tsv', sep='\t', index=False)
    info = {'candidate_nodes': len(nodes), 'feature_feature_edges': len(pairs), 'min_abs_rho': min_rho, 'max_q': max_q,
            'outcome_edges_data': int((direct_data > 0).sum()), 'outcome_edges_integrated': int((layers['integrated'] > 0).sum()),
            'genes_composition_linked': int((table.rho_with_linked_proportion.abs() >= min_rho).sum()),
            'top10_overlap_data_vs_integrated': int(len(set(table.nsmallest(10, 'rank_data').feature) & set(table.nsmallest(10, 'rank_integrated').feature)))}
    ml.dump(out / 'knowledge_graph_summary.json', info)
    return table, info
