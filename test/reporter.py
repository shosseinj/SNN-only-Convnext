#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import statistics


def flatten_record(d):
    keys=["experiment_name","dataset","seed","temporal_model_type","time_steps","residual_fusion","learnable_delay","force_positive_weights",
          "best_epoch","best_validation_accuracy","test_accuracy","test_loss","spikes_per_sample","synops_per_sample_estimate",
          "trainable_parameters","total_parameters","model_size_mb","dense_macs_per_sample","dense_flops_per_sample_estimate",
          "training_time_seconds","inference_ms_per_sample","peak_gpu_memory_mb","checkpoint","status"]
    return {k:d.get(k,"N/A") for k in keys}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--results_root",default="./results"); p.add_argument("--output",default="./reports/experiment_report"); a=p.parse_args()
    records=[]
    for exp in Path(a.results_root).glob("*"):
        merged={}
        for name in ["config.json","training_summary.json","evaluation_results.json"]:
            f=exp/name
            if f.exists(): merged.update(json.load(open(f,encoding="utf-8")))
        if merged: records.append(flatten_record(merged))
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    fields=list(records[0].keys()) if records else []
    with open(str(out)+".csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)
    json.dump(records,open(str(out)+".json","w",encoding="utf-8"),indent=2)
    lines=["# Experiment Report","",f"Experiments found: {len(records)}","", "| "+" | ".join(fields)+" |", "|"+"---|"*len(fields)]
    for r in records: lines.append("| "+" | ".join(str(r[k]) for k in fields)+" |")
    groups=defaultdict(list)
    for r in records:
        key=(r["dataset"],r["temporal_model_type"],r["time_steps"],r["residual_fusion"])
        if isinstance(r.get("test_accuracy"),(int,float)): groups[key].append(r)
    lines += ["","## Multi-seed summaries",""]
    for key,rs in groups.items():
        acc=[float(r["test_accuracy"]) for r in rs]
        mean=statistics.mean(acc); sd=statistics.stdev(acc) if len(acc)>1 else None
        lines.append(f"- {key}: accuracy = {mean:.4f}" + (f" ± {sd:.4f} over {len(acc)} seeds" if sd is not None else " (one seed)"))
    Path(str(out)+".md").write_text("\n".join(lines),encoding="utf-8")
    print(f"Wrote {out}.csv/.json/.md")
if __name__=="__main__": main()
