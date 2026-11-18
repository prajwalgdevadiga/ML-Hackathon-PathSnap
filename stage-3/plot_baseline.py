"""Plot measured retrieval and direction errors from the baseline run."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(folder):
    summary = json.loads((folder/"summary.json").read_text())
    fig, (left, right) = plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    x = np.arange(4)
    for i,(key,name,color) in enumerate([("train","Training","#196681"),
                                         ("student_test","Student test","#c56a27")]):
        s = summary[key]
        values = [100*s[k]/s["eligible"] for k in ("correct","hits_at_2","hits_at_4","hits_at_8")]
        left.bar(x+(i-.5)*.35, values,.35,label=name,color=color)
        errors = s["eligible"]-s["correct"]
        share = 100*s["reverse_errors"]/errors
        right.barh(i,share,color="#654d91",height=.52)
        right.barh(i,100-share,left=share,color="#d4cddf",height=.52)
        right.text(share/2,i,f"{share:.1f}%",ha="center",va="center",color="white",weight="bold")
    left.set(xticks=x,xticklabels=["First","First 2","First 4","First 8"],ylim=(0,105),
             ylabel="Eligible points with reference arc found (%)",title="Reference coverage by shortlist depth")
    left.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.19),ncol=2,fontsize=9)
    right.set(yticks=[0,1],yticklabels=["Training","Student test"],xlim=(0,100),
              xlabel="Share of nearest-road errors (%)",title="Opposite-direction errors")
    right.invert_yaxis()
    right.legend(["Same road, reverse direction","Other errors"],frameon=False,
                 loc="upper center",bbox_to_anchor=(.5,-.19),ncol=2,fontsize=9)
    for ax in (left,right):
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(folder/"baseline-results.png",dpi=160)
    plt.close(fig)


if __name__=="__main__":
    plot(Path(__file__).parent)
