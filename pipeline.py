# стекинг: линейные модели на tfidf + cnn -> catboost
# на выходе submission_v2.csv
import os
import time

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from features import handcrafted, split_url

SEED = 42
N_FOLDS = 5
os.makedirs("cache", exist_ok=True)

tr = pd.read_csv("train_6dd2c989.csv")
te = pd.read_csv("test_5013633d.csv")
y = tr.result.values.astype(int)
ntr = len(tr)
urls_all = pd.concat([tr.url, te.url], ignore_index=True)
parts = pd.DataFrame([split_url(u) for u in urls_all])
parts["sld"] = parts["dom"] + "." + parts["suffix"]
parts["pathq"] = parts["path"] + "?" + parts["query"]
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(tr, y))


def best_f1(y, p):
    ts = np.linspace(0.1, 0.9, 161)
    fs = np.array([f1_score(y, p > t) for t in ts])
    i = int(np.argmax(fs))
    return fs[i], ts[i]


def cached(name, fn):
    path = f"cache/{name}.npz"
    if os.path.exists(path):
        d = np.load(path)
        return d["oof"], d["test"]
    t = time.time()
    oof, test = fn()
    np.savez(path, oof=oof, test=test)
    f, th = best_f1(y, oof)
    print(f"  {name}: oof F1@.5={f1_score(y, oof > .5):.4f} best={f:.4f}@{th:.2f} ({time.time() - t:.0f}s)")
    return oof, test


def linear_model(texts, vec_factory, C):
    texts = np.asarray(texts)
    a, b = texts[:ntr], texts[ntr:]

    def run():
        oof, test = np.zeros(ntr), np.zeros(len(b))
        for trn, val in folds:
            vecs = vec_factory()
            Xtr = hstack([v.fit_transform(a[trn]) for v in vecs]).tocsr()
            m = LogisticRegression(C=C, max_iter=5000, solver="liblinear").fit(Xtr, y[trn])
            oof[val] = m.predict_proba(hstack([v.transform(a[val]) for v in vecs]).tocsr())[:, 1]
            test += m.predict_proba(hstack([v.transform(b) for v in vecs]).tocsr())[:, 1] / N_FOLDS
        return oof, test

    return run


def char_vec(lo, hi, lower=False):
    return TfidfVectorizer(analyzer="char", ngram_range=(lo, hi), min_df=2, sublinear_tf=True, lowercase=lower, dtype=np.float32)


def word_vec():
    return TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z0-9]+", ngram_range=(1, 2), min_df=1, sublinear_tf=True, dtype=np.float32)


def target_encoding(keys, prior_w=1.0):
    # target encoding по фолдам, для теста считаем по всему train
    keys = np.asarray(keys)
    ktr, kte = keys[:ntr], keys[ntr:]
    prior = y.mean()

    def enc(fit_idx, apply_keys):
        g = pd.DataFrame({"k": ktr[fit_idx], "y": y[fit_idx]}).groupby("k").y.agg(["sum", "size"])
        s = g["sum"].reindex(apply_keys).fillna(0).values
        n = g["size"].reindex(apply_keys).fillna(0).values
        return (s + prior * prior_w) / (n + prior_w), n

    oof_m, oof_n = np.zeros(ntr), np.zeros(ntr)
    for trn, val in folds:
        oof_m[val], oof_n[val] = enc(trn, ktr[val])
    te_m, te_n = enc(np.arange(ntr), kte)
    return oof_m, oof_n, te_m, te_n


print("level 1")
L1 = {
    "lr_full": linear_model(urls_all, lambda: [char_vec(1, 5), word_vec()], 32.0),
    "lr_full_lower": linear_model(urls_all.str.lower(), lambda: [char_vec(2, 6, True)], 16.0),
    "lr_host": linear_model(parts.host, lambda: [char_vec(1, 5)], 16.0),
    "lr_path": linear_model(parts.pathq, lambda: [char_vec(1, 5), word_vec()], 16.0),
}
USE_CONTENT = os.path.exists("cache/content_train.parquet") and os.path.exists("cache/content_test.parquet")
if USE_CONTENT:
    content = pd.concat([pd.read_parquet("cache/content_train.parquet"), pd.read_parquet("cache/content_test.parquet")],
                        ignore_index=True)
    assert (content.url.values == urls_all.values).all(), "content rows misaligned with train/test"
    ctext = content.ctext.fillna("").str.lower()
    L1["lr_content"] = linear_model(ctext, lambda: [
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=500_000, sublinear_tf=True, dtype=np.float32),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=500_000, sublinear_tf=True, dtype=np.float32)], 8.0)
    L1["lr_title"] = linear_model(content.title.fillna("").str.lower(), lambda: [char_vec(1, 4, True), word_vec()], 4.0)
l1_oof, l1_test = {}, {}
for name, fn in L1.items():
    l1_oof[name], l1_test[name] = cached(name, fn)
if os.path.exists("cache/cnn_url.npz"):  # из cnn_url.py, фолды те же
    d = np.load("cache/cnn_url.npz")
    L1["cnn_url"] = None
    l1_oof["cnn_url"], l1_test["cnn_url"] = d["oof"], d["test"]

print("level 2")
hc = handcrafted(urls_all)
for k in ["host", "reg", "sld", "suffix", "dom"]:
    om, on, tm, tn = target_encoding(parts[k])
    hc[f"te_{k}"] = np.concatenate([om, tm])
    hc[f"te_cnt_{k}"] = np.concatenate([on, tn])
    # частоты по train+test
    hc[f"freq_{k}"] = parts[k].map(parts[k].value_counts()).values
for name in L1:
    hc[name] = np.concatenate([l1_oof[name], l1_test[name]])
if USE_CONTENT:
    num = content.drop(columns=["url", "ctext", "title"]).astype(np.float32)
    hc = pd.concat([hc, num.add_prefix("c_")], axis=1)
    hc["te_title"] = np.concatenate(target_encoding(content.title.fillna("").str.lower().str[:100])[0::2])
    hc["freq_title"] = content.title.map(content.title.value_counts()).values
hc["suffix_cat"] = parts["suffix"].astype("category").cat.codes

X, Xte = hc.iloc[:ntr].reset_index(drop=True), hc.iloc[ntr:].reset_index(drop=True)
oof, test = np.zeros(ntr), np.zeros(len(Xte))
best_iters = []
for trn, val in folds:
    m = CatBoostClassifier(iterations=4000, learning_rate=0.05, depth=7, l2_leaf_reg=3, random_seed=SEED,
                           thread_count=8, verbose=0, early_stopping_rounds=200, eval_metric="Logloss")
    m.fit(X.iloc[trn], y[trn], eval_set=(X.iloc[val], y[val]))
    best_iters.append(m.get_best_iteration())
    oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    test += m.predict_proba(Xte)[:, 1] / N_FOLDS
f, th = best_f1(y, oof)
print(f"CatBoost stack: oof F1@.5={f1_score(y, oof > .5):.4f} best={f:.4f}@{th:.3f}")
imp = pd.Series(m.get_feature_importance(), X.columns).sort_values(ascending=False)
print(imp.head(20).round(2).to_string())
np.savez("cache/stack.npz", oof=oof, test=test)

pd.DataFrame({"Id": np.arange(len(te)), "Predicted": (test > th).astype(int)}).to_csv("submission_v1.csv", index=False)
print("submission_v1.csv written, positive rate:", (test > th).mean().round(4))

# v2: без early stopping на валидации, фикс. число итераций, 3 сида
n_iter = int(np.mean(best_iters) * 1.1)
print("v2: fixed iterations", n_iter)
oof2, test2 = np.zeros(ntr), np.zeros(len(Xte))
SEEDS = [SEED, SEED + 1, SEED + 2]
for s in SEEDS:
    for trn, val in folds:
        m = CatBoostClassifier(iterations=n_iter, learning_rate=0.05, depth=7, l2_leaf_reg=3, random_seed=s,
                               thread_count=8, verbose=0)
        m.fit(X.iloc[trn], y[trn])
        oof2[val] += m.predict_proba(X.iloc[val])[:, 1] / len(SEEDS)
        test2 += m.predict_proba(Xte)[:, 1] / (N_FOLDS * len(SEEDS))
f2, th2 = best_f1(y, oof2)
print(f"v2 CatBoost x{len(SEEDS)} seeds: oof F1@.5={f1_score(y, oof2 > .5):.4f} best={f2:.4f}@{th2:.3f}")
np.savez("cache/stack_v2.npz", oof=oof2, test=test2)
pd.DataFrame({"Id": np.arange(len(te)), "Predicted": (test2 > th2).astype(int)}).to_csv("submission_v2.csv", index=False)
print("submission_v2.csv written, positive rate:", (test2 > th2).mean().round(4),
      "| disagreement with v1:", ((test2 > th2) != (test > th)).sum())
