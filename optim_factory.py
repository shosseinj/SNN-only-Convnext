

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Optimiser factory with lazy / guarded imports so the script
runs with both old and new timm versions.
"""
import torch
from torch import optim as optim
import json

# -------------- timm optimisers – guard every optional import --------------
try:
    from timm.optim.adafactor import Adafactor
except ImportError:
    Adafactor = None
try:
    from timm.optim.adahessian import Adahessian
except ImportError:
    Adahessian = None
try:
    from timm.optim.adamp import AdamP
except ImportError:
    AdamP = None
try:
    from timm.optim.lookahead import Lookahead
except ImportError:
    Lookahead = None
try:
    from timm.optim.nadam import Nadam
except ImportError:
    Nadam = None
try:
    from timm.optim.novograd import NovoGrad
except ImportError:
    NovoGrad = None
try:
    from timm.optim.nvnovograd import NvNovoGrad
except ImportError:
    NvNovoGrad = None
try:
    from timm.optim.radam import RAdam
except ImportError:
    RAdam = None
try:
    from timm.optim.rmsprop_tf import RMSpropTF
except ImportError:
    RMSpropTF = None
try:
    from timm.optim.sgdp import SGDP
except ImportError:
    SGDP = None

# -------------- apex --------------
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


# ---------------------------------------------------------------------------
# ConvNeXT layer-id helper (unchanged)
# ---------------------------------------------------------------------------
def get_num_layer_for_convnext(var_name: str):
    num_max_layer = 12
    if var_name.startswith("downsample_layers"):
        stage_id = int(var_name.split('.')[1])
        return (0 if stage_id == 0 else
                stage_id + 1 if stage_id in (1, 2) else
                12)
    elif var_name.startswith("stages"):
        stage_id = int(var_name.split('.')[1])
        block_id = int(var_name.split('.')[2])
        if stage_id in (0, 1):
            return stage_id + 1
        if stage_id == 2:
            return 3 + block_id // 3
        return 12
    return num_max_layer + 1


class LayerDecayValueAssigner:
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_convnext(var_name)


# ---------------------------------------------------------------------------
# parameter grouping (unchanged)
# ---------------------------------------------------------------------------
def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None,
                         get_layer_scale=None):
    parameter_group_names, parameter_group_vars = {}, {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name, this_weight_decay = "no_decay", 0.
        else:
            group_name, this_weight_decay = "decay", weight_decay

        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = f"layer_{layer_id}_{group_name}"
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            scale = get_layer_scale(layer_id) if get_layer_scale is not None else 1.
            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


# ---------------------------------------------------------------------------
# optimiser factory
# ---------------------------------------------------------------------------
def create_optimizer(args, model, get_num_layer=None, get_layer_scale=None,
                     filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    if filter_bias_and_bn:
        skip = skip_list if skip_list is not None else (
            model.no_weight_decay() if hasattr(model, 'no_weight_decay') else {})
        parameters = get_parameter_groups(model, weight_decay, skip,
                                          get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), \
            'APEX and CUDA required for fused optimisers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]

    # ------------- SGD variants -------------------------------------------
    if opt_lower in ('sgd', 'nesterov'):
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)

    # ------------- Adam family --------------------------------------------
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        if Nadam is None:
            raise RuntimeError('Nadam not available in current timm version')
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        if RAdam is None:
            raise RuntimeError('RAdam not available in current timm version')
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':
        if AdamP is None:
            raise RuntimeError('AdamP not available in current timm version')
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)

    # ------------- others -------------------------------------------------
    elif opt_lower == 'sgdp':
        if SGDP is None:
            raise RuntimeError('SGDP not available in current timm version')
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        if Adafactor is None:
            raise RuntimeError('Adafactor not available in current timm version')
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        if Adahessian is None:
            raise RuntimeError('Adahessian not available in current timm version')
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        if RMSpropTF is None:
            raise RuntimeError('RMSpropTF not available in current timm version')
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        if NovoGrad is None:
            raise RuntimeError('NovoGrad not available in current timm version')
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        if NvNovoGrad is None:
            raise RuntimeError('NvNovoGrad not available in current timm version')
        optimizer = NvNovoGrad(parameters, **opt_args)

    # ------------- apex fused --------------------------------------------
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)

    else:
        raise ValueError(f'Invalid optimiser: {args.opt}')

    # ------------- wrapper ------------------------------------------------
    if len(opt_split) > 1 and opt_split[0] == 'lookahead':
        if Lookahead is None:
            raise RuntimeError('Lookahead not available in current timm version')
        optimizer = Lookahead(optimizer)

    return optimizer