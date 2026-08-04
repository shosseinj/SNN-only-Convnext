



## Ablation with 3 or 5 seeds:
Experiment 1;
- baseline
- +minimum residual merge
- +learnable delay
- +Non-negative weights

-----

Experiment 2 on residual merge:
- Sum
- minimum
- learnable gate

| روش          | Accuracy | Spike rate | SynOps | Parameters | زمان آموزش |
| ------------ | -------: | ---------: | -----: | ---------: | ---------: |
| Sum          |          |            |        |            |            |
| Learned Gate |          |            |        |            |            |
| Minimum      |          |            |        |            |            |

----- 


Experiment 3:
- Non-negative weight
- unlimited weight

Exoeriment 4:
- learnable delay
- static delay
- without delay


## Datasets:
- Cifar10
    - resolution 
    - training number:
    - validation number:

- Cifar100
    - resolution: 
    - training number:
    - valdiation number:

- ImageNet-tiny
    - 200 classes
    - 500 images for each class 
    - 100,000 images for training 
    - 10,000 for validation 
    - resolution 64*64 


## Seeds:

- 42
- 123
- 2024
- 3407
- 7777

```
import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

چه چیزهایی را برای هر Seed ذخیره کنیم؟

حداقل این موارد:

- Best validation accuracy
- Test accuracy مربوط به checkpoint انتخاب‌شده
- Training loss
- Validation loss
- Average spike rate یا activation sparsity
- SynOps
- تعداد پارامترها
- زمان آموزش
- در صورت امکان inference latency
- میانگین زمان اولین spike
- درصد neuronهایی که spike تولید نمی‌کنند

برای مقایسه روش‌ها بهتر است علاوه بر Mean ± Std، نتایج تک‌Seedها را نیز در supplementary material قرار دهیم. 
## ساختار پیشنهادی فایل نتیجه:

```
results/
├── cifar10/
│   ├── baseline_seed0.json
│   ├── baseline_seed1.json
│   ├── baseline_seed2.json
│   ├── baseline_seed3.json
│   └── baseline_seed4.json
└── cifar100/
    ├── baseline_seed0.json
    └── ...

```

```
{
  "dataset": "cifar100",
  "seed": 0,
  "best_epoch": 241,
  "test_accuracy": 81.42,
  "spike_rate": 0.137,
  "synops": 185000000,
  "parameters": 27800000
}
```
## محاسبه Mean ± Std
برای n=5 اجرا بهتر است sample standard deviation گزارش شود؛ یعنی در محاسبه انحراف معیار از ddof=1 استفاده شود.


```
import numpy as np

accuracies = np.array([81.42, 81.18, 81.53, 81.27, 81.46])

mean = accuracies.mean()
std = accuracies.std(ddof=1)

print(f"{mean:.2f} ± {std:.2f}")        # output will be like = 81.37 ± 0.14
```

##

