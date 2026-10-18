"""C1 - combine A1, B1 and B2 into the paper's headline numbers and figures.

Gap decomposition (PLAN.md, section 4). For any metric M where higher = worse:
    gap_total    = M(hist_raw)  - M(modern)
    orthography  = M(hist_raw)  - M(hist_norm)
    residual     = M(hist_norm) - M(modern)
    orth_share   = orthography / gap_total
CIs on the share come from a bootstrap over editions where unit-level data exist
(A1 perplexity, LID); B1/B2 report the point estimate with the component CIs.

Outputs (results/C1/):
  lid_threshold_sweep.csv    LID survival at thresholds 0.3/0.5/0.65/0.8
  gate_attribution.csv       per-gate independent removal + cascade survival
  gap_decomposition.csv      one row per metric
  headline.json              numbers quoted in the abstract
  figures/fig1_cascade.png, figures/fig2_gap.png

Run (after A1 raw + normalised (+ --xh-stopwords), B1, B2):
      python experiments/c1_summary.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from _common import PAPER_COLORS, config, outdir, plt, save_json

from src import stats

A1, A1N = config.RESULTS_DIR / "A1", config.RESULTS_DIR / "A1_norm"
THRESHOLDS = (0.3, 0.5, 0.65, 0.8)
GATES = ("lid_nllb", "gopher", "badwords", "perplexity")


def load_units() -> pd.DataFrame:
    raw = pd.read_csv(A1 / "units.csv.gz")
    norm = pd.read_csv(A1N / "units.csv.gz")
    norm = norm[norm.corpus == "hist_xh"].assign(corpus="hist_xh_norm")
    return pd.concat([raw, norm], ignore_index=True)


def lid_sweep(u: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for corp, g in u.groupby("corpus"):
        tgt = "eng" if corp == "hist_en" else "xho"
        for model in ("nllb", "glotlid"):
            if f"lid_{model}_label" not in g:
                continue
            right = g[f"lid_{model}_label"].str.split("_").str[0] == tgt
            for thr in THRESHOLDS:
                rows.append({"corpus": corp, "model": model, "threshold": thr,
                             "keep": float((right & (g[f"lid_{model}_prob"] >= thr)).mean())})
    return pd.DataFrame(rows).pivot_table(index=["corpus", "model"], columns="threshold", values="keep")


def attribution(u: pd.DataFrame, extra: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for label, g in [*u.groupby("corpus"), *extra.items()]:
        r = {"corpus": label, "n": len(g)}
        for gate in GATES:
            col = f"keep_{gate}"
            if col in g:
                r[f"removed_by_{gate}"] = 1 - g[col].mean()
        r["survives"] = g.survives.mean()
        est, lo, hi = stats.cluster_bootstrap(g, lambda d: d.survives.mean(), n_boot=500)
        r["survives_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(r)
    return pd.DataFrame(rows).set_index("corpus")


MIN_REL_GAP = 0.03  # below this the 1880s-vs-modern gap is too small for a meaningful share


def share(raw: float, norm: float, mod: float) -> float:
    """Orthographic share of the gap; undefined (NaN) when raw and modern differ by
    less than MIN_REL_GAP of the modern value (a ratio of two near-zero numbers)."""
    if mod == 0 or abs(raw - mod) / abs(mod) < MIN_REL_GAP:
        return np.nan
    return (raw - norm) / (raw - mod)


def unit_gap(u: pd.DataFrame, col: str, fn, n_boot: int = 500, seed: int = 0) -> dict:
    """Gap decomposition of a unit-level score with an edition bootstrap on all three corpora."""
    parts = {c: u[u.corpus == c].reset_index(drop=True) for c in ("hist_xh", "hist_xh_norm", "modern_xh")}
    point = {c: fn(d[col]) for c, d in parts.items()}
    rng = np.random.default_rng(seed)
    idx = {c: list(d.groupby("edition").indices.values()) for c, d in parts.items()}
    shares = []
    for _ in range(n_boot):
        v = {}
        for c, d in parts.items():
            pick = rng.integers(0, len(idx[c]), len(idx[c]))
            v[c] = fn(d[col].to_numpy()[np.concatenate([idx[c][k] for k in pick])])
        shares.append(share(v["hist_xh"], v["hist_xh_norm"], v["modern_xh"]))
    lo, hi = np.nanpercentile(shares, [2.5, 97.5])
    return {"hist_raw": point["hist_xh"], "hist_norm": point["hist_xh_norm"], "modern": point["modern_xh"],
            "orth_share": share(point["hist_xh"], point["hist_xh_norm"], point["modern_xh"]),
            "share_lo": lo, "share_hi": hi}


def main():
    out = outdir("C1")
    u = load_units()

    sweep = lid_sweep(u)
    sweep.to_csv(out / "lid_threshold_sweep.csv")
    print(sweep.round(3))

    extra = {}
    for name in ("A1_xhstop", "A1_norm_xhstop"):
        p = config.RESULTS_DIR / name / "units.csv.gz"
        if p.exists():
            x = pd.read_csv(p)
            for corp, g in x.groupby("corpus"):
                if corp != "hist_en" and not (name.startswith("A1_norm") and corp != "hist_xh"):
                    extra[f"{corp}{'_norm' if 'norm' in name else ''} (xh stop words)"] = g
    att = attribution(u, extra)
    att.to_csv(out / "gate_attribution.csv")
    print(att.round(3))

    # ---- gap decomposition -------------------------------------------------
    rows = []
    # LID: 1 - P(top label is isiXhosa) (higher = less recognisably isiXhosa)
    for model in ("nllb", "glotlid"):
        u[f"lid_{model}_xho_p"] = np.where(u[f"lid_{model}_label"].str.startswith("xho"), u[f"lid_{model}_prob"], 0.0)
        g = unit_gap(u, f"lid_{model}_xho_p", lambda s: 1 - np.mean(s))
        rows.append({"metric": f"LID {model}: 1 - P(xho)", "source": "A1", **g})
    rows.append({"metric": "CCNet perplexity (median bits/char)", "source": "A1",
                 **unit_gap(u, "ppl_bpc", np.median)})

    b1 = config.RESULTS_DIR / "B1" / "fertility.csv"
    if b1.exists():
        f = pd.read_csv(b1)
        for tok, g in f[f.metric == "tokens_per_char"].groupby("tokenizer"):
            v = g.set_index("condition").value
            rows.append({"metric": f"tokens/letter ({tok})", "source": "B1", "hist_raw": v["hist_raw"],
                         "hist_norm": v["hist_norm"], "modern": v["modern"],
                         "orth_share": share(v["hist_raw"], v["hist_norm"], v["modern"])})
    b2 = config.RESULTS_DIR / "B2" / "lm_2x2.csv"
    if b2.exists():
        f = pd.read_csv(b2)
        m = f[f.modern_frac == 1.0].set_index("spelling")
        if len(m) == 2:  # modern-only model: historical test (raw / norm) vs held-out modern
            rows.append({"metric": "modern char-LM bits/raw letter", "source": "B2",
                         "hist_raw": m.loc["raw", "hist_bpl"], "hist_norm": m.loc["norm", "hist_bpl"],
                         "modern": m.loc["raw", "modern_bpl"],
                         "orth_share": share(m.loc["raw", "hist_bpl"], m.loc["norm", "hist_bpl"], m.loc["raw", "modern_bpl"])})
    gap = pd.DataFrame(rows)
    gap.to_csv(out / "gap_decomposition.csv", index=False)
    print(gap.round(3).to_string())

    # ---- headline numbers ---------------------------------------------------
    head = {"survival": att["survives"].round(4).to_dict(),
            "removed_by_gate": att.filter(like="removed_by").round(4).to_dict(orient="index"),
            "gap": gap.set_index("metric")[["orth_share"]].round(3).orth_share.to_dict()}
    if b2.exists():
        f = pd.read_csv(b2)
        head["B2"] = f[["spelling", "modern_frac", "hist_bpw", "hist_bpl"]].round(3).to_dict(orient="records")
    save_json(head, out / "headline.json")

    # ---- Figure 1: per-gate removal (independent) + end-to-end survival ------
    p = plt()
    show = [c for c in ("hist_en", "modern_xh", "hist_xh", "hist_xh_norm") if c in att.index]
    labels = {"hist_en": "1880s English (same pages)", "modern_xh": "modern isiXhosa",
              "hist_xh": "1880s isiXhosa, raw", "hist_xh_norm": "1880s isiXhosa, normalised"}
    colors = {"hist_en": PAPER_COLORS["EN"], "modern_xh": PAPER_COLORS["modern"],
              "hist_xh": PAPER_COLORS["IMVO"], "hist_xh_norm": PAPER_COLORS["ISIG"]}
    # Each bar group is (decision column, use the isiXhosa-stop-word run?). English text keeps
    # Gopher's English list in that run, so its bars are the same in both settings.
    groups = [("removed_by_lid_nllb", False, "language ID"),
              ("removed_by_gopher", False, "Gopher\nEN stops"),
              ("removed_by_gopher", True, "Gopher\nxh stops"),
              ("removed_by_badwords", False, "bad words"),
              ("removed_by_perplexity", False, "perplexity"),
              ("survives", False, "survives\nEN stops"),
              ("survives", True, "survives\nxh stops")]
    groups = [g for g in groups if g[0] in att]

    def val(c, col, xh):
        key = f"{c} (xh stop words)"
        return att.loc[key if xh and key in att.index else c, col]

    fig, ax = p.subplots(figsize=(5.4, 2.4))
    w = 0.8 / len(show)
    for i, c in enumerate(show):
        vals = [val(c, col, xh) for col, xh, _ in groups]
        x = np.arange(len(groups)) + i * w
        ax.bar(x, vals, w, color=colors[c], label=labels[c])
        for j in (len(groups) - 2, len(groups) - 1):
            ax.text(x[j], vals[j] + 0.02, f"{100 * vals[j]:.1f}", ha="center", fontsize=5.5, rotation=90)
    ax.axvline(len(groups) - 2.5 + w * (len(show) - 1) / 2 + 0.4, color="k", lw=0.6, ls=":")
    ax.set_xticks(np.arange(len(groups)) + w * (len(show) - 1) / 2, [g[2] for g in groups], fontsize=6)
    ax.set_ylabel("share of 150-word units\nremoved / surviving", fontsize=7)
    ax.set_ylim(0, 1.18)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
    fig.savefig(config.FIG_DIR / "fig1_cascade.png")

    # ---- Figure 2: gap decomposition ---------------------------------------
    # Tokenisers are left out: their 1880s-vs-modern gap is under MIN_REL_GAP, so a share is meaningless.
    keep = ["LID nllb: 1 - P(xho)", "CCNet perplexity (median bits/char)", "modern char-LM bits/raw letter"]
    nice = {keep[0]: "language ID\n(1 - P(isiXhosa), NLLB)", keep[1]: "CCNet perplexity\n(bits/char)",
            keep[2]: "modern char LM\n(bits/raw letter)"}
    gp = gap.set_index("metric").reindex([k for k in keep if k in set(gap.metric)])
    fig, ax = p.subplots(figsize=(4.2, 1.9))
    y = np.arange(len(gp))
    o = ((gp.hist_raw - gp.hist_norm) / (gp.hist_raw - gp.modern)).to_numpy()
    ax.barh(y, o, color=PAPER_COLORS["ISIG"], label="closed by normalising spelling")
    ax.barh(y, 1 - o, left=o, color="#c8c8c8", label="residual (lexis, register, OCR)")
    has_ci = gp.share_lo.notna().to_numpy() if "share_lo" in gp else np.zeros(len(gp), bool)
    if has_ci.any():
        ax.errorbar(o[has_ci], y[has_ci], xerr=[o[has_ci] - gp.share_lo[has_ci], gp.share_hi[has_ci] - o[has_ci]],
                    fmt="none", ecolor="k", capsize=2, lw=0.8)
    for yi, (m, r) in enumerate(gp.iterrows()):
        ax.text(1.02, yi, f"{r.hist_raw:.3g} → {r.hist_norm:.3g} | {r.modern:.3g}", va="center", fontsize=5.5)
    ax.set_yticks(y, [nice[m] for m in gp.index], fontsize=6.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the 1880s-vs-modern gap")
    ax.legend(frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.savefig(config.FIG_DIR / "fig2_gap.png")
    print("saved to", out)


if __name__ == "__main__":
    main()
