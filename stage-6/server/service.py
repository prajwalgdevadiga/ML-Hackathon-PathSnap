"""FastAPI route prediction, with reference access confined to the scoring step."""
import argparse
import json
import logging
import re
from pathlib import Path, PurePosixPath
from threading import Lock
from time import perf_counter
from uuid import uuid4
from zipfile import ZipFile

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from threadpoolctl import threadpool_limits

from inference import predict_trip
from reference import inspect
from routescore import metrics

HERE=Path(__file__).resolve().parent
WEB=HERE.parent/"web"


class TripStore:
    """Index exactly one supplied partition. Never extract the archive or serve data files."""
    def __init__(self, dataset, partition, references=None):
        self.dataset=dataset
        self.references=references
        self.prefix={}
        if dataset.is_dir():
            for track in sorted(dataset.glob("*/*.track")):
                if track.parent.name==track.stem:
                    self.prefix[track.stem]=track.with_suffix("")
        else:
            with ZipFile(dataset) as archive:
                for name in archive.namelist():
                    p=PurePosixPath(name)
                    if p.suffix==".track" and len(p.parts)>=3 and p.parts[-3]==partition and p.parent.name==p.stem:
                        if p.stem in self.prefix:
                            raise ValueError("Duplicate trip ID in the selected partition.")
                        self.prefix[p.stem]=name[:-6]
        if len(self.prefix)!=10 or not all(re.fullmatch(r"[a-zA-Z0-9_-]+",s) for s in self.prefix):
            raise ValueError("Expected ten uniquely named trip folders. For a directory, pass the partition folder itself.")

    def read(self,trip,suffix,dtype=float,ndmin=2):
        if suffix=="route" and self.references:
            path=self.references/trip/(trip+".route")
            if not path.is_file():
                path=self.references/(trip+".route")
            return np.loadtxt(path,dtype=dtype,ndmin=ndmin)
        path=str(self.prefix[trip])+"."+suffix
        if self.dataset.is_dir():
            return np.loadtxt(path,dtype=dtype,ndmin=ndmin)
        with ZipFile(self.dataset) as archive,archive.open(path) as stream:
            return np.loadtxt(stream,dtype=dtype,ndmin=ndmin)


def presentation(track,nodes,arcs,result):
    gps=track[:,1::-1]
    latlon=nodes[:,::-1]
    nearby=latlon[arcs]
    pad=np.array([.002,.003])
    inside=((nearby.max(axis=1)>=gps.min(axis=0)-pad)&(nearby.min(axis=1)<=gps.max(axis=0)+pad)).all(axis=1)
    road_edges=np.unique(np.sort(arcs[inside],axis=1),axis=0)
    # Emit one directed polyline per continuous run. Keep inferred connectors explicit.
    runs=[]
    connector_edges=[]
    for i,arc in enumerate(result["decoded"]):
        if arc<0:
            continue
        a,b=arcs[arc]
        if result["starts"][i]:
            runs.append([latlon[a].tolist(),latlon[b].tolist()])
        elif arc!=result["decoded"][i-1] or result["connectors"][i]>=0:
            connector=result["connectors"][i]
            if connector>=0:
                ca,cb=arcs[connector]
                connector_edges.append(latlon[[ca,cb]].tolist())
                runs[-1].append(latlon[cb].tolist())
            runs[-1].append(latlon[b].tolist())
    return dict(gps=gps.tolist(),roads=latlon[road_edges].tolist(),runs=runs,connectors=connector_edges,
                predicted_arc_ids=result["decoded"].tolist())


def server_scores(store,trip,track,nodes,arcs,result):
    reference=store.read(trip,"route",int,1)
    _,(path,_,_,_,eligible,_)=inspect(trip,(track,nodes,arcs,reference))
    r=metrics(track,arcs,reference,reference[path],eligible,result["decoded"][:,None],result["fallback"])
    return dict(observations=r["gps"],eligible=r["eligible"],correct=r["correct"],
                accuracy=r["correct"]/r["eligible"] if r["eligible"] else None,
                macro_f1=r["macro_f1"],sequence_f1=r["sequence_f1"],
                continuity=r["continuous"]/r["steps"] if r["steps"] else None)


def create_app(store,mode):
    model=joblib.load(HERE/"tuned.joblib")
    scale=json.loads((HERE/"decoder-config.json").read_text())["scale_m"]
    gate=Lock()
    app=FastAPI(title="PathSnap",docs_url=None,redoc_url=None,openapi_url=None)
    app.add_middleware(GZipMiddleware,minimum_size=2048)
    app.mount("/assets",StaticFiles(directory=WEB/"assets"),name="assets")

    @app.get("/")
    def home():
        return FileResponse(WEB/"index.html")

    @app.get("/api/trips")
    def trips():
        return {"mode":mode,"trips":sorted(store.prefix)}

    @app.post("/api/trips/{trip}/predict")
    def run(trip:str,request:Request):
        if trip not in store.prefix:
            raise HTTPException(404,"Trip is outside the active partition.")
        req=uuid4().hex[:12]
        with gate,threadpool_limits(3):
            track=store.read(trip,"track")
            nodes=store.read(trip,"nodes")
            arcs=store.read(trip,"arcs",int)
            result=predict_trip(model,scale,track,nodes,arcs)
            output=presentation(track,nodes,arcs,result)
            try:
                scores=server_scores(store,trip,track,nodes,arcs,result)
            except (FileNotFoundError,KeyError):
                raise HTTPException(503,"The server-side reference for this trip is unavailable.")
        ms=round(1000*result["seconds"])
        logging.getLogger("uvicorn.error").info("RESULT id=%s client=%s trip=%s correct=%s/%s inference_ms=%d",
            req,request.client.host if request.client else "unknown",trip,scores["correct"],scores["eligible"],ms)
        return dict(request_id=req,mode=mode,trip=trip,metrics=scores,inference_ms=ms,map=output)
    return app


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",required=True,type=Path,help="Dataset ZIP or chosen partition directory")
    p.add_argument("--partition",default="student_test",help="ZIP partition folder name")
    p.add_argument("--references",type=Path,help="Optional private directory of reference routes")
    p.add_argument("--mode",choices=["student-test","evaluator"],default="student-test")
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=8765)
    p.add_argument("--log",type=Path,help="Also write receiving log to this local file")
    args=p.parse_args()
    store=TripStore(args.data,args.partition,args.references)
    config=dict(uvicorn.config.LOGGING_CONFIG)
    if args.log:
        config["handlers"]=dict(config["handlers"],file={"class":"logging.FileHandler",
            "filename":str(args.log),"formatter":"default","encoding":"utf8"})
        config["loggers"]={k:dict(v) for k,v in config["loggers"].items()}
        for k in ("uvicorn","uvicorn.access"):
            config["loggers"][k]["handlers"]=[*config["loggers"][k]["handlers"],"file"]
    print(f"PathSnap | {args.mode} | {len(store.prefix)} trips | http://{args.host}:{args.port}",flush=True)
    uvicorn.run(create_app(store,args.mode),host=args.host,port=args.port,log_config=config,
                proxy_headers=False)


if __name__=="__main__":
    main()
