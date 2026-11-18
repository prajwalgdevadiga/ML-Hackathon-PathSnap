"""Build the comparison figure and checkpoint notes from the measured runs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NAMES={"logistic":"Logistic regression","gaussian_nb":"Gaussian Naive Bayes","tree":"Decision tree",
       "forest":"Random forest","hist_boost":"Histogram gradient boosting","logistic_no_heading":"Logistic regression without heading"}
MAIN=list(NAMES)[:5]
DETAIL={
    "logistic":"The linear model estimates candidate log odds with L2-regularised logistic loss. Standardisation is fitted inside each fold. C = 1 and the iteration limit is 250. This gives a fast, interpretable test of distance and direction, with limited ability to express interactions.",
    "gaussian_nb":"The generative model estimates class priors and class-conditional Gaussian means and variances for standardised features, with variance smoothing of 1e-8. It is inexpensive to fit. Correlation between distance and distance above the nearest candidate, and the binary availability flags, challenge its assumptions.",
    "tree":"A single decision tree uses Gini impurity, maximum depth 12 and at least 30 candidate rows per leaf. It can express different rules for moving and stationary observations. Its probability estimates depend on individual leaf counts.",
    "forest":"The forest averages positive-class probabilities from 48 Gini trees. Each uses a half-size bootstrap sample, square-root feature sampling, maximum depth 14 and at least 12 rows per leaf. Averaging can stabilise a single tree's rules, at a cost in prediction time.",
    "hist_boost":"The histogram-based ensemble fits additive trees to binary log loss. The fixed configuration uses 75 iterations, learning rate 0.1, up to 15 leaves, at least 30 rows per leaf and L2 penalty 1. Early stopping is disabled. It can learn how speed and direction interact, with more fitting choices than a linear model."
}


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    out=Path(__file__).parent
    raw=read(out/"comparison.csv")
    d={(r["algorithm"],r["split"]):{k:float(v) for k,v in r.items() if k not in ("algorithm","split")} for r in raw}
    folds=read(out/"fold-scores.csv")
    times=read(out/"fit-log.csv")
    winner,second,*_=sorted(MAIN,key=lambda n:d[n,"cv"]["macro_f1"],reverse=True)
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout="constrained")
    y=np.arange(len(MAIN))
    labels=["Logistic regression","Gaussian NB","Decision tree","Random forest","Histogram boosting"]
    for ax,(split,key,title) in zip(axes,[("cv","macro_f1","Cross-validation Macro F1"),
                ("student_test","accuracy","Student-test Top-1"),("student_test","log_loss","Student-test log loss")]):
        values=[d[n,split][key] for n in MAIN]
        ax.barh(y,values,color=["#23786c" if n==winner else "#7191a8" for n in MAIN])
        ax.set_yticks(y,labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title,fontsize=11,loc="left")
        ax.set_xlim(0,1.04 if key!="log_loss" else max(values)*1.2)
        for i,v in enumerate(values):
            ax.text(v+.006,i,f"{v:.3f}",va="center",fontsize=8)
        ax.spines[["top","right","left"]].set_visible(False)
    fig.suptitle("Five candidate classifiers on the same trips and road shortlists",fontweight="bold")
    fig.savefig(out/"model-comparison.png",dpi=155)
    plt.close(fig)
    # Include student-test Macro F1 in its own plot beside Top-1 for review.
    fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
    for off,key,col,label in [(-.17,"accuracy","#276b70","Top-1 accuracy"),(.17,"macro_f1","#ad7034","Macro F1")]:
        vals=[d[n,"student_test"][key] for n in MAIN]
        bars=ax.barh(y+off,vals,.32,color=col,label=label)
        ax.bar_label(bars,fmt="%.3f",padding=3,fontsize=8)
    ax.set(yticks=y,yticklabels=labels,xlim=(0,1.03),title="Development results on the ten student-test trips")
    ax.invert_yaxis()
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.08),ncol=2,frameon=False)
    fig.savefig(out/"student-test-comparison.png",dpi=155)
    plt.close(fig)
    lines=["# PathSnap Stage 4 model comparison","",
      f"**Recommendation for team review.** Take {NAMES[winner].lower()} into Bayesian tuning. It ranked first on the primary measure, the mean trip-level out-of-fold Macro F1 across all 80 training trips. Its score was {d[winner,'cv']['macro_f1']:.4f}, compared with {d[second,'cv']['macro_f1']:.4f} for {NAMES[second].lower()}. The primary measure gives each trip and its road classes weight. Student-test results are development comparisons. No evaluator data were available or used.","",
      "| Algorithm | CV Macro F1 | Student Top-1 | Student Macro F1 | Student log loss | Continuity | Sequence F1 | Inference, 10 trips |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in MAIN:
        r=d[n,"student_test"]
        lines.append(f"| {NAMES[n]} | {d[n,'cv']['macro_f1']:.4f} | {r['accuracy']:.2%} | {r['macro_f1']:.4f} | {r['log_loss']:.4f} | {r['continuity']:.2%} | {r['sequence_f1']:.4f} | {r['inference_s']:.3f} s |")
    full=d["logistic","student_test"]
    abl=d["logistic_no_heading","student_test"]
    aligned=d["logistic_no_heading","cv"]
    lines+=["","## Travel direction and baseline","",
     f"The paired logistic-regression experiment dropped heading cosine and its availability flag while retaining speed. Student-test Top-1 was {abl['accuracy']:.2%} without those features and {full['accuracy']:.2%} with them, a difference of {100*(full['accuracy']-abl['accuracy']):.2f} percentage points. Reverse-direction mistakes changed from {abl['reverse_errors']:,.0f} to {full['reverse_errors']:,.0f}. Out-of-fold Macro F1 changed from {aligned['macro_f1']:.4f} to {d['logistic','cv']['macro_f1']:.4f}. The earlier nearest-road baseline achieved 59.485% student-test Top-1, Macro F1 of 0.4542 and sequence F1 of 0.4800. It made 4,741 reverse-direction errors.","",
     "## Design and scoring","",
     "All five classifiers take the eight frozen features from Stage 3 and estimate whether a candidate is the derived target for an observation. Each eligible candidate row is one binary training example. All available candidates are included with equal row weight, including all-negative sets when the reference is missing. The full training fit uses 709,934 candidate rows and 134,600 positive rows. Uncertain observations are retained at prediction time. Trip IDs, arc IDs and reference-derived quantities are excluded from model features.","",
     "Five folds are formed by shuffling the 80 unique training trip IDs with seed 352, then assigning 16 entire trips to validation per fold. Every training trip is scored once out of fold. Each algorithm is also refitted on all 80 trips for the 10-trip student-test comparison. The logistic heading check is a controlled variant of the same algorithm. Configuration, fold membership, fit times, per-trip scores and predictions are included in the package.","",
     "At prediction time, positive-class probabilities are clipped to [1e-12, 1 − 1e-12] and converted to odds. Odds are normalised across valid candidates to produce a trip-local candidate distribution per point. Equal scores prefer the fixed Stage 3 candidate order. The probabilities are estimates for the shortlist. They have not undergone a separate calibration fit. Point log loss is mean negative log probability of the eligible reference with a floor of 1e-12, including candidate misses.","",
     "Point metrics use the frozen derived labels. Training has 134,669 eligible points out of 208,417 and student testing has 13,126 out of 19,160. Four student-test candidate misses and 69 training misses count against every algorithm. Pooled Top-1 weights observations. Macro F1, precision and recall average over the union of predicted and reference arcs within each trip, then weight trips equally. A training trip with only one eligible observation has the same trip weight and increases fold variability.","",
     "Route continuity uses every prediction and excludes intervals exceeding 30 seconds. It counts a stay or a directly connected arc change as continuous. Sequence F1 compares ordered predicted and reference arc lists using the Stage 3 longest-common-subsequence protocol. Change-only continuity is available in the CSV. Inference times cover candidate scoring and ranking on cached features, including probability checks. They exclude candidate generation, reference alignment, metrics and API transport. All timings are single local measurements.","",
     "## Reading the individual results",""]
    for n in MAIN:
        test=d[n,"student_test"]
        cv=d[n,"cv"]
        tr=d[n,"refit_train"]
        vals=[float(x["macro_f1"]) for x in folds if x["algorithm"]==n]
        cost=next(float(x["seconds"]) for x in times if x["algorithm"]==n and x["fit"]=="full")
        lines += [f"### {NAMES[n]}","",DETAIL[n],"",
            f"Full training took {cost:.2f} s. The refit scored {tr['macro_f1']:.4f} Macro F1 on its training trips and {cv['macro_f1']:.4f} out of fold. Fold scores ranged from {min(vals):.4f} to {max(vals):.4f}. The observed train-minus-validation difference was {tr['macro_f1']-cv['macro_f1']:+.4f}. On student testing it got {test['correct']:,.0f} of 13,126 eligible points correct, with macro precision {test['macro_precision']:.4f} and macro recall {test['macro_recall']:.4f}.",""]
    lines+=["The refit–validation differences describe this dataset and are not formal tests of overfitting. Variation across trips and ambiguous point labels limits fine-grained claims about algorithm rank. No epoch losses were recorded for these runs.","",
            "## Reproduction and provenance","",
            "Use Python 3.13.5, scikit-learn 1.6.1, NumPy 2.1.3, SciPy 1.15.3, joblib 1.4.2 and Matplotlib 3.10.0. From this directory, the first command rebuilds the approved Stage 3 cache if needed. Place generated intermediates in a local working directory.","",
            "```text",'python stage3/run_stage3.py "path/to/Student folder.zip" --out "path/to/baseline" --cache "path/to/cache"',
            'python compare.py "path/to/Student folder.zip" "path/to/cache"',
            'python verify.py "path/to/cache"',"python review.py","```","",
            "This review records AI-assisted implementation and execution. It is a development checkpoint for team review. Saved estimators, predictions, configurations and measured outputs support independent reproduction and the team's own interpretation. Hyperparameter tuning and route decoding follow a separate approval checkpoint."]
    (out/"Stage 4 review.md").write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
