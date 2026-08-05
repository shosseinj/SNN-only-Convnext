هر block:
Depthwise Conv → Spike
Pointwise Conv → Spike
Pointwise Conv → Spike

- residual

جمع زمانی spikeهای آخرین Stage

# Training ANN version

```
python .\trainer_ann_convnext_relu.py `
  --experiment_name cifar10_ann_relu_tiny_seed42_fp32 `
  --model_size tiny `
  --batch_size 128 `
  --epochs 200 `
  --lr 0.001 `
  --min_lr 0.00001 `
  --warmup_epochs 5 `
  --weight_decay 0.05 `
  --label_smoothing 0.1 `
  --cifar_stem true `
  --current_norm true `
  --residual true `
  --force_positive_weights false `
  --amp false `
  --grad_clip 5 `
  --seed 42 `
  --download false

```
