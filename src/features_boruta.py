"""
Step 03 -- Boruta feature selection on the TRAINING rounds only (2011-2017).

The 2022 test survey is never seen here, so selection cannot leak future data.
Outputs:
  * results/selected_features.json  (confirmed / tentative / rejected + ranking)
  * figures/fig2_boruta.png         (RF importance bar, coloured by Boruta decision)

Run: python -m src.features_boruta
"""
from __future__ import annotations

import json

import numpy as np

# --- numpy 2.x shim: Boruta 0.4.3 still references the removed np.int/np.float ---
for _alias, _py in {"int": int, "float": float, "bool": bool}.items():
    if _alias not in np.__dict__:
        setattr(np, _alias, _py)

from boruta import BorutaPy                       # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

import config                                      # noqa: E402
from src import viz                                # noqa: E402
from src.preprocess import build_design            # noqa: E402

MIN_FEATURES = 5   # if Boruta confirms fewer, widen the set (logged, never silent)


def run(force: bool = False) -> dict:
    out_path = config.RESULTS / "selected_features.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())

    d = build_design()
    X, y = d["X_train"], d["y_train"]
    names = np.array(d["feature_names"])

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=7, class_weight="balanced",
        random_state=config.SEED, n_jobs=-1,
    )
    boruta = BorutaPy(rf, n_estimators="auto", max_iter=100,
                      random_state=config.SEED, verbose=0)
    boruta.fit(X, y)

    confirmed = names[boruta.support_].tolist()
    tentative = names[boruta.support_weak_].tolist()
    rejected = names[~(boruta.support_ | boruta.support_weak_)].tolist()

    # decide the final set (no silent truncation -- log any widening)
    selected = list(confirmed)
    if len(selected) < MIN_FEATURES:
        print(f"  Boruta confirmed only {len(confirmed)} (<{MIN_FEATURES}); "
              f"adding {len(tentative)} tentative feature(s).")
        selected += [f for f in tentative if f not in selected]
    if len(selected) < MIN_FEATURES:
        rf.fit(X, y)
        top = names[np.argsort(rf.feature_importances_)[::-1][:MIN_FEATURES]]
        print(f"  still <{MIN_FEATURES}; falling back to top-{MIN_FEATURES} RF importance.")
        selected = list(dict.fromkeys(selected + top.tolist()))

    result = {
        "selected": selected,
        "confirmed": confirmed,
        "tentative": tentative,
        "rejected": rejected,
        "ranking": dict(zip(names.tolist(), boruta.ranking_.astype(int).tolist())),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  selected {len(selected)} features -> {out_path}")

    _plot(X, y, names, boruta, rf)
    return result


def _plot(X, y, names, boruta, rf):
    rf.fit(X, y)
    imp = rf.feature_importances_
    order = np.argsort(imp)
    decision = np.where(boruta.support_, "confirmed",
                        np.where(boruta.support_weak_, "tentative", "rejected"))
    colour = {"confirmed": viz.PALETTE[3], "tentative": viz.PALETTE[1],
              "rejected": viz.PALETTE[7]}

    fig, ax = viz.plt.subplots(figsize=(7, 9))
    ax.barh(range(len(order)), imp[order],
            color=[colour[decision[i]] for i in order])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i].replace("num__", "").replace("cat__", "")
                        for i in order], fontsize=8)
    ax.set_xlabel("Random-forest importance")
    # in-image title removed for journal style; caption carries it
    handles = [viz.plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (colour["confirmed"], colour["tentative"], colour["rejected"])]
    ax.legend(handles, ["confirmed", "tentative", "rejected"], loc="lower right")
    viz.save(fig, "fig2_boruta.png")


if __name__ == "__main__":
    run(force=True)
