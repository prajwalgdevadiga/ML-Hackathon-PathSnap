"""Offline Folium comparisons and a plot of the measured Stage 5 trials."""
import argparse
import csv
from zipfile import ZipFile

from branca.element import Element
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tune import ROOT
from run_stage3 import read_array


def trip_map(z,k,saved):
    base=f"Student folder/student_test/{k}/{k}."
    track=read_array(z,base+"track",float,2)
    nodes=read_array(z,base+"nodes",float,2)
    arcs=read_array(z,base+"arcs",int,2)
    route=read_array(z,base+"route",int,1)
    points=track[:,1::-1]
    road=nodes[arcs][:,:,::-1]
    m=folium.Map(location=points.mean(axis=0),tiles=None,prefer_canvas=True,zoom_start=13)
    m.default_js=[]
    m.default_css=[]
    for name,tag in [("leaflet.js","script"),("leaflet.css","style")]:
        m.get_root().header.add_child(Element(f"<{tag}>"+(ROOT/"assets"/name).read_text(encoding="utf8")+f"</{tag}>"))
    m.get_root().header.add_child(Element("<style>.leaflet-container{background:#f8f7f1}.leaflet-control-layers{font:14px Arial}</style>"))
    pad=np.array([.002,.003])
    nearby=((road.max(axis=1)>=points.min(axis=0)-pad)&(road.min(axis=1)<=points.max(axis=0)+pad)).all(axis=1)
    def layer(name,indices,color,width,show=True,dash=None):
        group=folium.FeatureGroup(name,show=show).add_to(m)
        if len(indices):
            folium.PolyLine(road[np.unique(indices)].tolist(),color=color,weight=width,opacity=.86,dash_array=dash).add_to(group)
    layer("Road network · grey",np.flatnonzero(nearby),"#b5b6b3",1)
    layer("Development reference · green",route,"#36865d",8)
    layer("Tuned classifier · orange",saved["tuned_"+k],"#d98136",5,False)
    layer("Viterbi prediction · blue",saved["viterbi_"+k],"#2859a0",3)
    connector=saved["connector_"+k]
    layer("Inferred connecting roads · purple",connector[connector>=0],"#8c3584",3,True,"4 4")
    gps=folium.FeatureGroup("GPS observations · charcoal",show=True).add_to(m)
    for i,pt in enumerate(points):
        folium.CircleMarker(pt,radius=1.6,stroke=False,fill=True,fill_color="#222a32",fill_opacity=.55,
            tooltip=f"GPS row {i} · t={track[i,2]:g} s").add_to(gps)
    breaks=folium.FeatureGroup("Start of a decoded run · red",show=False).add_to(m)
    for i in np.flatnonzero(saved["start_"+k]):
        folium.CircleMarker(points[i],radius=5,color="#b82943",tooltip=f"Run starts at row {i}").add_to(breaks)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([points.min(axis=0).tolist(),points.max(axis=0).tolist()])
    m.get_root().html.add_child(Element(f'<div style="position:fixed;bottom:12px;left:12px;z-index:900;background:white;padding:9px 13px;font:14px Arial">Student-test trip <b>{k}</b> · Toggle layers to compare routes</div>'))
    m.save(ROOT/f"route-{k}.html")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive")
    args=p.parse_args()
    with (ROOT/"trip-results.csv").open() as f:
        rows=[r for r in csv.DictReader(f) if r["split"]=="student_test" and r["method"]=="viterbi"]
    typical=min(r["trip"] for r in rows)
    difficult=min(rows,key=lambda r:int(r["repaired"])-int(r["damaged"]))["trip"]
    with ZipFile(args.archive) as z,np.load(ROOT/"student-predictions.npz") as saved:
        for k in sorted({typical,difficult}):
            trip_map(z,k,saved)
    with (ROOT/"bayesian-trials.csv").open() as f:
        trials=list(csv.DictReader(f))
    with (ROOT/"comparison.csv").open() as f:
        methods=[r for r in csv.DictReader(f) if r["split"]=="student_test"]
    fig,(a,b)=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    val=np.array([float(r["cv_macro_f1"]) for r in trials])
    x=np.arange(1,len(val)+1)
    a.scatter(x,val,c=["#8a9095"]*3+["#28679d"]*7)
    a.step(x,np.maximum.accumulate(val),where="post",color="#21765f")
    a.set(xlabel="Trial number",ylabel="Trip-averaged Macro F1",title="Training-fold search")
    for dx,key,label,color in [(-.17,"accuracy","Point accuracy","#28679d"),(.17,"sequence_f1","Sequence F1","#21765f")]:
        values=[float(r[key]) for r in methods]
        bars=b.bar(np.arange(3)+dx,values,.32,label=label,color=color)
        b.bar_label(bars,fmt="%.3f",fontsize=8)
    b.set(xticks=np.arange(3),xticklabels=["Untuned","Tuned","Tuned + Viterbi"],ylim=(0,1.08),title="Ten student-test trips")
    b.legend(loc="upper center",bbox_to_anchor=(.5,-.12),frameon=False,ncol=2)
    fig.savefig(ROOT/"stage-5-results.png",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    main()
