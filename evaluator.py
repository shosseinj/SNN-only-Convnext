#!/usr/bin/env python3
"""Evaluate a trained pure discrete-time TTFS SNN checkpoint."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.discrete_ttfs_convnext import build_discrete_ttfs_convnext
from experiment_utils import atomic_json_dump, count_parameters, model_size_mb
from reproducibility import seed_everything, seed_worker


def str2bool(v):
    if isinstance(v,bool): return v
    if str(v).lower() in {"1","true","yes","y"}: return True
    if str(v).lower() in {"0","false","no","n"}: return False
    raise argparse.ArgumentTypeError("expected boolean")


def parse_args():
    p=argparse.ArgumentParser("Pure discrete TTFS evaluator")
    p.add_argument("--checkpoint",required=True); p.add_argument("--config",default="")
    p.add_argument("--data_path",default="./cifar_data"); p.add_argument("--output_dir",default="")
    p.add_argument("--batch_size",type=int,default=150); p.add_argument("--num_workers",type=int,default=4)
    p.add_argument("--device",default="cuda"); p.add_argument("--download",type=str2bool,default=True)
    p.add_argument("--synthetic_data",action="store_true"); p.add_argument("--max_batches",type=int,default=0)
    return p.parse_args()


def dense_ops_hooks(model):
    totals={"macs":0.0}; handles=[]
    def hook(m, inp, out):
        x=inp[0]
        if isinstance(m,nn.Conv2d):
            batch=out.shape[0]; spatial=out.shape[2]*out.shape[3]
            kh,kw=m.kernel_size
            totals["macs"] += batch*spatial*m.out_channels*(m.in_channels/m.groups)*kh*kw
        elif isinstance(m,nn.Linear):
            rows=x.numel()/x.shape[-1]
            totals["macs"] += rows*m.in_features*m.out_features
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)): handles.append(m.register_forward_hook(hook))
    return totals,handles


def main(args):
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    cfg=ckpt.get("config",{})
    if args.config:
        cfg.update(json.load(open(args.config,encoding="utf-8")))
    seed=int(cfg.get("seed",42)); gen=seed_everything(seed)
    device=torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model=build_discrete_ttfs_convnext(model_size=cfg.get("model_size","tiny"),in_chans=3,
      num_classes=int(cfg.get("num_classes",10)),time_steps=int(cfg.get("time_steps",5)),
      threshold=float(cfg.get("threshold",1.0)),learnable_delay=bool(cfg.get("learnable_delay",True)),
      init_delay=float(cfg.get("init_delay",0.0)),force_positive_weights=bool(cfg.get("force_positive_weights",True))).to(device)
    missing,unexpected=model.load_state_dict(ckpt["model"],strict=False)
    if missing or unexpected: raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    tf=transforms.ToTensor()
    if args.synthetic_data: ds=datasets.FakeData(size=32,image_size=(3,32,32),num_classes=10,transform=tf)
    else: ds=datasets.CIFAR10(args.data_path,train=False,download=args.download,transform=tf)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,
                      worker_init_fn=seed_worker,generator=gen,pin_memory=device.type=="cuda")
    criterion=nn.CrossEntropyLoss()
    trainable,total_params=count_parameters(model)
    correct=total=0; loss_sum=spikes=synops=0.0; elapsed=0.0
    spike_by_t=[0.0]*model.time_steps; synops_by_t=[0.0]*model.time_steps
    dense_macs=0.0
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for bi,(x,y) in enumerate(loader):
            if args.max_batches and bi>=args.max_batches: break
            x,y=x.to(device),y.to(device)
            op,handles=dense_ops_hooks(model)
            if device.type=="cuda": torch.cuda.synchronize()
            t0=time.perf_counter(); logits,st=model(x,return_stats=True)
            if device.type=="cuda": torch.cuda.synchronize()
            elapsed += time.perf_counter()-t0
            for h in handles: h.remove()
            dense_macs += op["macs"]
            n=y.numel(); total+=n; loss_sum+=criterion(logits,y).item()*n
            correct+=(logits.argmax(1)==y).sum().item(); spikes+=st["total_spikes"]; synops+=st["total_synops_estimate"]
            for i,v in enumerate(st["spike_count_per_timestep"]): spike_by_t[i]+=float(v)
            for i,v in enumerate(st["synops_per_timestep_estimate"]): synops_by_t[i]+=float(v)
    peak=(torch.cuda.max_memory_allocated(device)/(1024**2)) if device.type=="cuda" else None
    delays=torch.cat([m.values().detach().cpu().flatten() for m in model.modules() if hasattr(m,"values") and callable(m.values)])
    # Effective positive weights are ReLU(raw); evaluate all synaptic tensors because this new model constrains all of them.
    eff=[]
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)) and getattr(m,"force_positive_weights",False): eff.append(torch.relu(m.weight.detach()).cpu().flatten())
    ew=torch.cat(eff) if eff else torch.tensor([])
    result={
      "experiment_name":cfg.get("experiment_name",Path(args.checkpoint).parent.name),"status":"completed",
      "dataset":"FakeData" if args.synthetic_data else "CIFAR-10","seed":seed,
      "temporal_model_type":"DISCRETE_TIME_TTFS_SNN","time_steps":model.time_steps,"maximum_spikes_per_neuron":1,
      "residual_fusion":"earliest_spike_or","prediction_rule":"argmax_on_differentiable_first_spike_score",
      "checkpoint_epoch":ckpt.get("epoch"),"test_samples":total,"test_loss":loss_sum/max(total,1),"test_accuracy":100*correct/max(total,1),
      "spikes_per_sample":spikes/max(total,1),"spike_count_per_timestep_per_sample":[v/max(total,1) for v in spike_by_t],
      "synops_per_sample_estimate":synops/max(total,1),"synops_per_timestep_per_sample_estimate":[v/max(total,1) for v in synops_by_t],
      "synops_status":"approximate_event_fanout","trainable_parameters":trainable,"total_parameters":total_params,
      "model_size_mb":model_size_mb(model),"dense_macs_per_sample":dense_macs/max(total,1),
      "dense_flops_per_sample_estimate":2*dense_macs/max(total,1),"flops_definition":"2 FLOPs per multiply-accumulate; includes all T-step calls",
      "inference_time_seconds":elapsed,"inference_ms_per_sample":1000*elapsed/max(total,1),"peak_gpu_memory_mb":peak,
      "learnable_delay_parameters":int(delays.numel()),"mean_delay_steps":float(delays.mean()) if delays.numel() else None,
      "min_delay_steps":float(delays.min()) if delays.numel() else None,"max_delay_steps":float(delays.max()) if delays.numel() else None,
      "minimum_effective_constrained_weight":float(ew.min()) if ew.numel() else None,
      "fraction_effective_weights_near_zero":float((ew<1e-8).float().mean()) if ew.numel() else None,
    }
    out=Path(args.output_dir or Path(args.checkpoint).parent); out.mkdir(parents=True,exist_ok=True)
    atomic_json_dump(result,out/"evaluation_results.json")
    print(json.dumps(result,indent=2)); return result

if __name__=="__main__": main(parse_args())
