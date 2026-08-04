# # Copyright (c) Meta Platforms, Inc. and affiliates.

# # All rights reserved.

# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.layers import trunc_normal_, DropPath
# from timm.models.registry import register_model


# def call_spiking_torch(tj, W, D_i, t_min_prev, t_min, t_max):
#     """
#     PyTorch version of the TF `call_spiking`.
#     Inputs:
#       - tj: input spike times tensor (shape [..., C_in])  # last dim is channels/features
#       - W: weight matrix (C_in, C_out) or (C_out, C_in) depending on matmul orientation
#       - D_i: per-output delay (broadcastable to output shape)
#       - t_min_prev: previous layer min spike time (unused for simplified form but kept for API)
#       - t_min, t_max: scalars or tensors
#     Returns:
#       - ti: output spike times clamped by t_max
#     Notes:
#       - Ensure matmul dimension ordering matches your usage (here we use `... @ W` with W shape (C_in, C_out))
#     """

#     # threshold Eq. 18
#     threshold = t_max - t_min - D_i

#     # ensure floating dtype
#     tj = tj.to(W.dtype)
#     threshold = threshold.to(W.dtype)

#     # compute ti = (tj - t_min) @ W + threshold + t_min
#     # if last dim of tj is C_in and W is (C_in, C_out)
#     delta = tj - t_min
#     ti = torch.matmul(delta, W) + threshold + t_min

#     # clamp to t_max
#     ti = torch.where(ti < t_max, ti, t_max)
#     return ti


# class Block(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
    
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
#         # self.norm = LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
#                                     requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
#         # x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         # if self.gamma is not None:
#         #     x = self.gamma * x
#         x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

#         x = input + self.drop_path(x)
#         return x

#     def get_sparsity(self):
#         """Calculate sparsity for each sub-layer in this block"""
#         sparsity_dict = {}
        
#         # Depthwise conv sparsity
#         dwconv_weights = self.dwconv.weight.data
#         dwconv_zeros = (dwconv_weights == 0).sum().item()
#         dwconv_total = dwconv_weights.numel()
#         sparsity_dict['dwconv'] = dwconv_zeros / dwconv_total if dwconv_total > 0 else 0.0
        
#         # First pointwise conv sparsity
#         pwconv1_weights = self.pwconv1.weight.data
#         pwconv1_zeros = (pwconv1_weights == 0).sum().item()
#         pwconv1_total = pwconv1_weights.numel()
#         sparsity_dict['pwconv1'] = pwconv1_zeros / pwconv1_total if pwconv1_total > 0 else 0.0
        
#         # Second pointwise conv sparsity
#         pwconv2_weights = self.pwconv2.weight.data
#         pwconv2_zeros = (pwconv2_weights == 0).sum().item()
#         pwconv2_total = pwconv2_weights.numel()
#         sparsity_dict['pwconv2'] = pwconv2_zeros / pwconv2_total if pwconv2_total > 0 else 0.0
        
#         # Average sparsity for this block
#         sparsity_dict['block_avg'] = (sparsity_dict['dwconv'] + sparsity_dict['pwconv1'] + sparsity_dict['pwconv2']) / 3
        
#         return sparsity_dict

   
# class ConvNeXt(nn.Module):
#     r""" ConvNeXt
#         A PyTorch impl of : `A ConvNet for the 2020s`  -
#           https://arxiv.org/pdf/2201.03545.pdf

#     Args:
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
#         dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
#         drop_path_rate (float): Stochastic depth rate. Default: 0.
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#         head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
#     """
#     # def __init__(self, in_chans=3, num_classes=1000, 
#     #              depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
#     #              layer_scale_init_value=1e-6, head_init_scale=1.,
#     #              ):
#     def __init__(self, in_chans=3, num_classes=1000, depths=(3, 3, 9, 3),
#                  dims=(96, 192, 384, 768), drop_path_rate=0.,
#                  head_init_scale=1., layer_scale_init_value=1e-6, **kwargs):
#         # ----- timm ≥ 0.9 compatibility -----
#         # ----- timm ≥ 0.9 passes this key; ignore it -----
#         kwargs.pop('pretrained_cfg', None)
#         super().__init__()

#         self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
#         stem = nn.Sequential(
#             nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#             # LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
#         )
#         self.downsample_layers.append(stem)
#         for i in range(3):
#             downsample_layer = nn.Sequential(
#                     # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
#                     nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
#             )
#             self.downsample_layers.append(downsample_layer)

#         self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
#         dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
#         cur = 0
#         for i in range(4):
#             stage = nn.Sequential(
#                 *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
#                 layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
#             )
#             self.stages.append(stage)
#             cur += depths[i]

#         # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
#         self.head = nn.Linear(dims[-1], num_classes)

#         self.apply(self._init_weights)
#         self.head.weight.data.mul_(head_init_scale)
#         self.head.bias.data.mul_(head_init_scale)

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             trunc_normal_(m.weight, std=.02)
#             nn.init.constant_(m.bias, 0)

#     def forward_features(self, x):
#         for i in range(4):
#             x = self.downsample_layers[i](x)
#             x = self.stages[i](x)
#         x = x.mean([-2, -1])
#             # x = self.norm(x)
#         return x # global average pooling, (N, C, H, W) -> (N, C)

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x


# class SpikingBlock(nn.Module):
#     """
#     Spiking variant of ConvNeXt Block using TTFS analytic mapping.
#     Reuses weights from an existing Block instance.
#     """
#     def __init__(self, orig_block: Block, t_min=0.0, t_max=1.0, force_positive_weights: bool = False, init_delay: float = 0.0):
#         super().__init__()
#         # reuse modules (share weights)
#         self.dwconv = orig_block.dwconv
#         self.pw1 = orig_block.pwconv1
#         self.pw2 = orig_block.pwconv2
#         self.force_positive_weights = force_positive_weights
#         self.gamma = getattr(orig_block, 'gamma', None)
#         self.drop_path = orig_block.drop_path if hasattr(orig_block, 'drop_path') else nn.Identity()
#         self.t_min = float(t_min)
#         self.t_max = float(t_max)

#         self.D_mid = nn.Parameter(torch.zeros(self.pw1.out_features))
#         self.D_out = nn.Parameter(torch.zeros(self.pw2.out_features))
        
#         # Store intermediate spike times for sparsity analysis
#         self.t_mid_spike = None  # After pw1 spiking
#         self.t_out_spike = None  # After pw2 spiking
        
#         # Optionally initialize delays to a small positive value
#         self._init_delay = float(init_delay)
#         if self._init_delay > 0.0:
#             with torch.no_grad():
#                 self.D_mid.data.fill_(self._init_delay)
#                 self.D_out.data.fill_(self._init_delay)

#     def get_sparsity(self):
#         """Calculate sparsity for each sub-layer in this spiking block"""
#         sparsity_dict = {}
        
#         # Depthwise conv sparsity
#         dwconv_weights = self.dwconv.weight.data
#         dwconv_zeros = (dwconv_weights == 0).sum().item()
#         dwconv_total = dwconv_weights.numel()
#         sparsity_dict['dwconv'] = dwconv_zeros / dwconv_total if dwconv_total > 0 else 0.0
        
#         # First pointwise conv sparsity
#         pw1_weights = self.pw1.weight.data
#         pw1_zeros = (pw1_weights == 0).sum().item()
#         pw1_total = pw1_weights.numel()
#         sparsity_dict['pw1'] = pw1_zeros / pw1_total if pw1_total > 0 else 0.0
        
#         # Second pointwise conv sparsity
#         pw2_weights = self.pw2.weight.data
#         pw2_zeros = (pw2_weights == 0).sum().item()
#         pw2_total = pw2_weights.numel()
#         sparsity_dict['pw2'] = pw2_zeros / pw2_total if pw2_total > 0 else 0.0
        
#         # Average sparsity for this block
#         sparsity_dict['block_avg'] = (sparsity_dict['dwconv'] + sparsity_dict['pw1'] + sparsity_dict['pw2']) / 3
        
#         return sparsity_dict

#     def get_activation_sparsity(self):
#         """Calculate activation sparsity from stored spike times."""
#         sparsity_dict = {}
        
#         # After pw1 spiking
#         if self.t_mid_spike is not None:
#             silent_mid = (self.t_mid_spike >= self.t_max - 1e-6)
#             sparsity_dict['mid'] = silent_mid.sum().item() / self.t_mid_spike.numel()
#         else:
#             sparsity_dict['mid'] = 0.0
        
#         # After pw2 spiking
#         if self.t_out_spike is not None:
#             silent_out = (self.t_out_spike >= self.t_max - 1e-6)
#             sparsity_dict['out'] = silent_out.sum().item() / self.t_out_spike.numel()
#         else:
#             sparsity_dict['out'] = 0.0
        
#         # Final output
#         if self.latest_spike is not None:
#             silent_final = (self.latest_spike >= self.t_max - 1e-6)
#             sparsity_dict['final'] = silent_final.sum().item() / self.latest_spike.numel()
#         else:
#             sparsity_dict['final'] = 0.0
        
#         # Average of both spiking operations
#         sparsity_dict['avg_spiking_ops'] = (sparsity_dict['mid'] + sparsity_dict['out']) / 2
        
#         return sparsity_dict

#     # def forward(self, tj):
#     #     # tj: spike times tensor (N, C, H, W)
#     #     x = tj
#     #     # depthwise conv: apply to the spike-time map (heuristic)
#     #     x = self.dwconv(x)

#     #     # prepare for pointwise linear mapping per spatial location
#     #     N, C, H, W = x.shape
#     #     x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C_in)

#     #     # pw1
#     #     W1 = (torch.relu(self.pw1.weight) if self.force_positive_weights else self.pw1.weight).t().contiguous()
#     #     device = x_flat.device
#     #     dtype = x_flat.dtype
#     #     D_mid = torch.clamp(torch.relu(self.D_mid), max=0.9 * (self.t_max - self.t_min)).to(device=device, dtype=dtype)
#     #     t_min = torch.tensor(self.t_min, device=device, dtype=dtype)
#     #     t_max = torch.tensor(self.t_max, device=device, dtype=dtype)
#     #     t_mid = call_spiking_torch(x_flat, W1, D_mid, None, t_min, t_max)
        
#     #     # Store first spiking output
#     #     self.t_mid_spike = t_mid.view(N, H, W, -1).permute(0, 3, 1, 2).detach()

#     #     # pw2
#     #     W2 = (torch.relu(self.pw2.weight) if self.force_positive_weights else self.pw2.weight).t().contiguous()
#     #     D_out = torch.clamp(torch.relu(self.D_out), max=0.9 * (self.t_max - self.t_min)).to(device=device, dtype=dtype)
#     #     t_out = call_spiking_torch(t_mid, W2, D_out, None, t_min, t_max)

#     #     # reshape back
#     #     t_out = t_out.view(N, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
#     #     # Store second spiking output
#     #     self.t_out_spike = t_out.detach()

#     #     # out = tj + self.drop_path(t_out)
#     #     out = torch.minimum(tj, self.drop_path(t_out))

#     #     # store latest output spike times for regularization/monitoring
#     #     try:
#     #         self.latest_spike = out
#     #         # self.latest_spike = out.detach()
#     #     except Exception:
#     #         self.latest_spike = None

#     #     return out
#     def forward(self, tj):
#         # tj: spike times tensor (N, C, H, W)
#         x = self.dwconv(tj)

#         # prepare for pointwise linear mapping per spatial location
#         N, C, H, W = x.shape
#         x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C_in)

#         device = x_flat.device
#         dtype = x_flat.dtype
#         t_min = torch.tensor(self.t_min, device=device, dtype=dtype)
#         t_max = torch.tensor(self.t_max, device=device, dtype=dtype)

#         # ---- pw1: SPIKING (with activation, like GELU in ANN) ----
#         W1 = (torch.relu(self.pw1.weight) if self.force_positive_weights else self.pw1.weight).t().contiguous()
#         D_mid = torch.clamp(torch.relu(self.D_mid), max=0.9 * (self.t_max - self.t_min)).to(device=device, dtype=dtype)
#         t_mid = call_spiking_torch(x_flat, W1, D_mid, None, t_min, t_max)
        
#         # Store first spiking output
#         self.t_mid_spike = t_mid.view(N, H, W, -1).permute(0, 3, 1, 2).detach()

#         # ---- pw2: LINEAR (no spiking, like ANN block has no activation after pw2) ----
#         # Convert spike times back to "voltage" domain, apply linear, convert back
#         # Use negative spike times as activation scores: s = -t (earlier spike → higher score)
#         s_mid = -t_mid  # (N*H*W, 4C)
        
#         W2 = (torch.relu(self.pw2.weight) if self.force_positive_weights else self.pw2.weight)
#         s_out = torch.matmul(s_mid, W2.t())  # (N*H*W, C)
        
#         # Convert back to spike times: t = -s, then clamp
#         t_out = -s_out
#         t_out = torch.clamp(t_out, self.t_min, self.t_max)
        
#         # reshape back
#         t_out = t_out.view(N, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
#         # Store second output (not spiking, but store for analysis)
#         self.t_out_spike = t_out.detach()

#         # Residual connection with minimum
#         out = torch.minimum(tj, self.drop_path(t_out))

#         # Store latest output
#         self.latest_spike = out.detach()
        
#         return out

# # ============================================================================
# # SPARSITY CALCULATION FUNCTIONS


# def calculate_model_sparsity(model):
#     """
#     Calculate activation sparsity for all SpikingBlock layers.
#     Returns both scenarios: block output and per-spiking-op.
#     """
#     stage_sparsities_final = []
#     stage_sparsities_mid = []
#     stage_sparsities_out = []
#     layer_names_final = []
#     layer_names_mid = []
#     layer_names_out = []
    
#     print("\n" + "="*100)
#     print("ACTIVATION SPARSITY ANALYSIS")
#     print("="*100)
    
#     # Collect sparsity from all blocks
#     for stage_idx, stage in enumerate(model.stages):
#         for block_idx, block in enumerate(stage):
#             if isinstance(block, SpikingBlock):
#                 name = f"stages.{stage_idx}.{block_idx}"
#                 sparsity_info = block.get_activation_sparsity()
                
#                 stage_sparsities_final.append(sparsity_info['final'] * 100)
#                 stage_sparsities_mid.append(sparsity_info['mid'] * 100)
#                 stage_sparsities_out.append(sparsity_info['out'] * 100)
#                 layer_names_final.append((name, sparsity_info['final'] * 100))
#                 layer_names_mid.append((name, sparsity_info['mid'] * 100))
#                 layer_names_out.append((name, sparsity_info['out'] * 100))
    
#     if not stage_sparsities_final:
#         print("No SpikingBlock layers found!")
#         return None
    
#     # Scenario 1: Final output sparsity
#     print(f"\n{'='*60}")
#     print("SCENARIO 1: BLOCK OUTPUT SPARSITY")
#     print(f"  (Final output after torch.minimum)")
#     print(f"{'='*60}")
#     print(f"{'Layer':<25} {'Sparsity %':>12}")
#     print(f"{'-'*40}")
    
#     for name, sp in layer_names_final:
#         print(f"{name:<25} {sp:11.2f}%")
    
#     avg_final = sum(stage_sparsities_final) / len(stage_sparsities_final)
#     print(f"{'-'*40}")
#     print(f"{'AVERAGE':<25} {avg_final:11.2f}%")
#     print(f"  Min: {min(stage_sparsities_final):.2f}%")
#     print(f"  Max: {max(stage_sparsities_final):.2f}%")
    
#     # Scenario 2: Per-spiking-op sparsity
#     print(f"\n{'='*60}")
#     print("SCENARIO 2: PER-SPIKING-OP SPARSITY")
#     print(f"  (All call_spiking_torch outputs)")
#     print(f"{'='*60}")
#     print(f"{'Layer':<25} {'After pw1':>12} {'After pw2':>12}")
#     print(f"{'-'*52}")
    
#     for i in range(len(layer_names_mid)):
#         name = layer_names_mid[i][0]
#         mid_sp = layer_names_mid[i][1]
#         out_sp = layer_names_out[i][1]
#         print(f"{name:<25} {mid_sp:11.2f}% {out_sp:11.2f}%")
    
#     avg_mid = sum(stage_sparsities_mid) / len(stage_sparsities_mid)
#     avg_out = sum(stage_sparsities_out) / len(stage_sparsities_out)
#     print(f"{'-'*52}")
#     print(f"{'AVERAGE':<25} {avg_mid:11.2f}% {avg_out:11.2f}%")
    
#     # Combined
#     all_spiking_ops = stage_sparsities_mid + stage_sparsities_out
#     overall_avg = sum(all_spiking_ops) / len(all_spiking_ops)
    
#     # Summary
#     print(f"\n{'='*60}")
#     print("COMPARISON SUMMARY")
#     print(f"{'='*60}")
#     print(f"  Total spiking blocks: {len(stage_sparsities_final)}")
#     print(f"  Total spiking ops: {len(all_spiking_ops)}")
#     print(f"")
#     print(f"  Scenario 1 (Block outputs):     {avg_final:.2f}%")
#     print(f"  Scenario 2 (Per-spiking-op):    {overall_avg:.2f}%")
#     print(f"    - After pw1:                  {avg_mid:.2f}%")
#     print(f"    - After pw2:                  {avg_out:.2f}%")
#     print(f"")
#     print(f"  VGG16-SNN (literature):         ~30%")
#     print(f"{'='*60}\n")
    
#     return {
#         'scenario1_avg': avg_final,
#         'scenario2_avg': overall_avg,
#         'pw1_avg': avg_mid,
#         'pw2_avg': avg_out,
#         'final_sparsities': stage_sparsities_final,
#         'mid_sparsities': stage_sparsities_mid,
#         'out_sparsities': stage_sparsities_out
#     }


# # ============================================================================
# # USAGE
# # ============================================================================
# # After loading your model:
# # model.eval()
# # with torch.no_grad():
# #     dummy_input = torch.randn(1, 3, 32, 32)
# #     _ = model(dummy_input)
# # 
# # results = calculate_model_sparsity(model)

# class ConvNeXtSpiking(ConvNeXt):
#     def __init__(self, *args, t_min=0.0, t_max=1.0, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.force_positive_weights = kwargs.get('force_positive_weights', False)
#         # accept init_delay forwarded from model constructor
#         self.init_delay = kwargs.get('init_delay', 0.0)
#         # stage_specific_delays: list of 4 delay values, one per stage (overrides init_delay if provided)
#         # Example: [0.3, 0.1, 0.05, 0.02] for higher sparsity in early stages
#         self.stage_delays = kwargs.get('stage_delays', None)
        
#         # replace blocks in stages with SpikingBlock wrappers preserving weights
#         for si, stage in enumerate(self.stages):
#             # Determine delay for this stage
#             if self.stage_delays is not None and si < len(self.stage_delays):
#                 stage_delay = self.stage_delays[si]
#             else:
#                 stage_delay = self.init_delay
            
#             new_blocks = []
#             for b in stage:
#                 spb = SpikingBlock(b, t_min=t_min, t_max=t_max, force_positive_weights=self.force_positive_weights, init_delay=stage_delay)
#                 new_blocks.append(spb)
#             self.stages[si] = nn.Sequential(*new_blocks)
#         # head: we need to map spike times to logits; we'll treat head as linear on features,
#         # but input into head should be scores, so we convert times -> negative times as activation.
#         # Keep self.head as-is but forward will convert spike-times to scores before head.
#         self.t_min = float(t_min)
#         self.t_max = float(t_max)

            
#     def forward_features(self, x_t):
#         # x_t: spike times tensor (N, C_in, H, W)
#         # propagate
#         for i in range(4):
#             x_t = self.downsample_layers[i](x_t)   # these are convs on time maps
#             x_t = self.stages[i](x_t)
#         # global pool: average over spatial dims of spike times
#         x_pool = x_t.mean([-2, -1])  # shape (N, C)
#         return x_pool

#     def forward(self, x_t):
#         # x_t: spike times in [t_min, t_max]
#         x_pool = self.forward_features(x_t)  # spike times per channel
#         # convert times to scores: earlier spike -> higher score; simple mapping s = -t
#         logits = self.head(-x_pool)
#         return logits

#     def get_layer_sparsity(self):
#         """
#         Calculate and return per-layer sparsity statistics for all blocks.
#         Returns a dictionary with sparsity info for each stage and block.
#         """
#         sparsity_report = {}
        
#         # Calculate sparsity for each stage and block
#         for stage_idx, stage in enumerate(self.stages):
#             sparsity_report[f'stage_{stage_idx}'] = {}
#             stage_sparsities = []
            
#             for block_idx, block in enumerate(stage):
#                 if isinstance(block, SpikingBlock):
#                     block_sparsity = block.get_sparsity()
#                     sparsity_report[f'stage_{stage_idx}'][f'block_{block_idx}'] = block_sparsity
#                     stage_sparsities.append(block_sparsity['block_avg'])
            
#             # Average sparsity for this stage
#             if stage_sparsities:
#                 sparsity_report[f'stage_{stage_idx}']['stage_avg'] = sum(stage_sparsities) / len(stage_sparsities)
        
#         # Calculate global average sparsity
#         all_block_avgs = []
#         for stage_idx, stage in enumerate(self.stages):
#             if f'stage_{stage_idx}' in sparsity_report and 'stage_avg' in sparsity_report[f'stage_{stage_idx}']:
#                 all_block_avgs.append(sparsity_report[f'stage_{stage_idx}']['stage_avg'])
        
#         sparsity_report['global_avg'] = sum(all_block_avgs) / len(all_block_avgs) if all_block_avgs else 0.0
        
#         return sparsity_report

#     def print_layer_sparsity(self):
#         """
#         Print formatted per-layer sparsity statistics for academic reporting.
#         This method displays weight sparsity (zero weights in parameters).
#         For activation sparsity, use evaluate_sparsity.py instead.
#         """
#         sparsity_report = self.get_layer_sparsity()
#         print("\n" + "="*80)
#         print("WEIGHT SPARSITY REPORT (Zero Weights in Model Parameters)")
#         print("="*80)
#         print("Note: Weight sparsity = 0 indicates model learned dense, non-zero weights")
#         print("For activation sparsity during inference, refer to evaluate_sparsity.py\n")
        
#         for stage_idx in range(4):
#             stage_key = f'stage_{stage_idx}'
#             if stage_key in sparsity_report:
#                 print(f"Stage {stage_idx}:")
#                 for block_idx in range(len(self.stages[stage_idx])):
#                     block_key = f'block_{block_idx}'
#                     if block_key in sparsity_report[stage_key]:
#                         block_sparsity = sparsity_report[stage_key][block_key]
#                         print(f"  Block {block_idx}:")
#                         print(f"    dwconv:    {block_sparsity['dwconv']:.4f}")
#                         print(f"    pw1:       {block_sparsity['pw1']:.4f}")
#                         print(f"    pw2:       {block_sparsity['pw2']:.4f}")
#                         print(f"    block_avg: {block_sparsity['block_avg']:.4f}")
                
#                 # Stage average
#                 if 'stage_avg' in sparsity_report[stage_key]:
#                     print(f"  Stage Average: {sparsity_report[stage_key]['stage_avg']:.4f}\n")
        
#         print(f"Global Average Weight Sparsity: {sparsity_report['global_avg']:.4f}")
#         print("="*80 + "\n")
    



# def compute_ann_macs_for_size(input_size):
#     """Quick ANN MACs for given input size"""
#     H0 = W0 = input_size // 4
#     H1, W1 = H0 // 2, W0 // 2
#     H2, W2 = H1 // 2, W1 // 2
#     H3, W3 = H2 // 2, W2 // 2
    
#     total = 0
#     # Stem
#     total += 4 * 4 * 3 * 96 * H0 * W0
#     # Downsample
#     total += 2 * 2 * 96 * 192 * H1 * W1
#     total += 2 * 2 * 192 * 384 * H2 * W2
#     total += 2 * 2 * 384 * 768 * H3 * W3
#     # Head
#     total += 768 * 10
    
#     # Blocks
#     configs = [(96, H0, W0, 3), (192, H1, W1, 3), (384, H2, W2, 9), (768, H3, W3, 3)]
#     for C, H, W, blocks in configs:
#         for _ in range(blocks):
#             total += 49 * C * H * W  # dwconv
#             total += 4 * C * C * H * W  # pw1
#             total += 4 * C * C * H * W  # pw2
    
#     return total



# def compute_energy_correctly(model, dataloader, device, t_max=1.0):
#     """
#     Correct energy accounting for TTFS SNN.
#     All spatial dimensions are detected dynamically from the model.
#     """
#     model.eval()
    
#     # Run one batch to populate spike times and get spatial dims
#     with torch.no_grad():
#         for images, _ in dataloader:
#             images = images.to(device)
#             _ = model(images)
#             break
    
#     total_macs = 0
#     total_synops = 0
#     all_layers = []
    
#     # Detect spatial dimensions from first SpikingBlock
#     H0 = W0 = 0
#     for name, module in model.named_modules():
#         if isinstance(module, SpikingBlock) and module.t_mid_spike is not None:
#             _, _, H0, W0 = module.t_mid_spike.shape
#             break
    
#     if H0 == 0:
#         print("ERROR: Could not detect spatial dimensions!")
#         return None
    
#     # Calculate all spatial sizes
#     H1, W1 = H0 // 2, W0 // 2
#     H2, W2 = H1 // 2, W1 // 2
#     H3, W3 = H2 // 2, W2 // 2
    
#     input_size = H0 * 4  # Stem has stride 4
#     print(f"\nDetected input size: {input_size}×{input_size}")
#     print(f"Spatial sizes: {H0}×{H0} -> {H1}×{H1} -> {H2}×{H2} -> {H3}×{H3}")
    
#     print(f"\n{'='*100}")
#     print(f"PER-LAYER OPERATION COUNT")
#     print(f"{'='*100}")
#     print(f"{'Layer':<30} {'Type':<15} {'Operation':<12} {'Count':>15} {'Details':>20}")
#     print(f"{'-'*100}")
    
#     # ---- Stem: Non-spiking (stride=4) ----
#     stem_macs = 4 * 4 * 3 * 96 * H0 * W0
#     total_macs += stem_macs
#     all_layers.append({'name': 'stem', 'type': 'Non-spiking', 'op': 'MACs', 'count': stem_macs})
#     print(f"{'stem':<30} {'Non-spiking':<15} {'MACs':<12} {stem_macs:>15,} {f'Conv 3->96, {H0}x{W0}':>20}")
    
#     # ---- Downsample layers: Non-spiking (stride=2) ----
#     ds_configs = [
#         ('downsample_1', 96, 192, H1, W1),
#         ('downsample_2', 192, 384, H2, W2),
#         ('downsample_3', 384, 768, H3, W3),
#     ]
    
#     for ds_name, in_ch, out_ch, h, w in ds_configs:
#         ds_macs = 2 * 2 * in_ch * out_ch * h * w
#         total_macs += ds_macs
#         all_layers.append({'name': ds_name, 'type': 'Non-spiking', 'op': 'MACs', 'count': ds_macs})
#         print(f"{ds_name:<30} {'Non-spiking':<15} {'MACs':<12} {ds_macs:>15,} {f'Conv {in_ch}->{out_ch}, {h}x{w}':>20}")
    
#     # ---- Spiking Blocks ----
#     for name, module in model.named_modules():
#         if isinstance(module, SpikingBlock):
#             H, W = 0, 0
#             dwconv_macs = 0
#             synops_pw1 = 0
#             synops_pw2 = 0
#             fired_pw1 = 0
#             fired_pw2 = 0
            
#             # ---- dwconv (Non-spiking) ----
#             if module.t_mid_spike is not None:
#                 _, _, H, W = module.t_mid_spike.shape
#                 dwconv_macs = 49 * module.dwconv.in_channels * H * W
#                 total_macs += dwconv_macs
            
#             # ---- pw1 (Spiking) ----
#             if module.t_mid_spike is not None:
#                 fired_pw1 = (module.t_mid_spike < t_max - 1e-6).sum().item()
#                 fan_in_pw1 = module.pw1.in_features
#                 synops_pw1 = fired_pw1 * fan_in_pw1
#                 total_synops += synops_pw1
            
#             # ---- pw2 (Spiking) ----
#             if module.t_out_spike is not None:
#                 if H == 0:
#                     _, _, H, W = module.t_out_spike.shape
#                 fired_pw2 = (module.t_out_spike < t_max - 1e-6).sum().item()
#                 fan_in_pw2 = module.pw2.in_features
#                 synops_pw2 = fired_pw2 * fan_in_pw2
#                 total_synops += synops_pw2
            
#             # Print
#             all_layers.append({'name': f'{name}.dwconv', 'type': 'Non-spiking', 'op': 'MACs', 'count': dwconv_macs})
#             print(f"{name+'.dwconv':<30} {'Non-spiking':<15} {'MACs':<12} {dwconv_macs:>15,} {f'7x7 Depthwise, {H}x{W}':>20}")
            
#             all_layers.append({'name': f'{name}.pw1', 'type': 'Spiking', 'op': 'SynOps', 'count': synops_pw1})
#             spike_rate_pw1 = (fired_pw1 / module.t_mid_spike.numel() * 100) if module.t_mid_spike is not None and module.t_mid_spike.numel() > 0 else 0
#             print(f"{name+'.pw1':<30} {'Spiking':<15} {'SynOps':<12} {synops_pw1:>15,} {f'Spikes: {fired_pw1:,} ({spike_rate_pw1:.1f}%)':>20}")
            
#             all_layers.append({'name': f'{name}.pw2', 'type': 'Spiking', 'op': 'SynOps', 'count': synops_pw2})
#             spike_rate_pw2 = (fired_pw2 / module.t_out_spike.numel() * 100) if module.t_out_spike is not None and module.t_out_spike.numel() > 0 else 0
#             print(f"{name+'.pw2':<30} {'Spiking':<15} {'SynOps':<12} {synops_pw2:>15,} {f'Spikes: {fired_pw2:,} ({spike_rate_pw2:.1f}%)':>20}")
            
#             print(f"{'-'*100}")
    
#     # ---- Head: Non-spiking ----
#     head_macs = 768 * 10
#     total_macs += head_macs
#     all_layers.append({'name': 'head', 'type': 'Non-spiking', 'op': 'MACs', 'count': head_macs})
#     print(f"{'head':<30} {'Non-spiking':<15} {'MACs':<12} {head_macs:>15,} {'Linear 768->10':>20}")
    
#     # ================================================================
#     # SUMMARY
#     # ================================================================
#     ENERGY_PER_MAC = 4.6e-12
#     ENERGY_PER_SOP = 0.9e-12
    
#     energy_macs_mj = total_macs * ENERGY_PER_MAC * 1000
#     energy_synops_mj = total_synops * ENERGY_PER_SOP * 1000
#     total_energy_mj = energy_macs_mj + energy_synops_mj
    
#     # Calculate ANN MACs for the same input size
#     ann_macs = compute_ann_macs_for_size(input_size)
#     ann_energy = ann_macs * ENERGY_PER_MAC * 1000
    
#     non_spiking_count = sum(1 for l in all_layers if l['type'] == 'Non-spiking')
#     spiking_count = sum(1 for l in all_layers if l['type'] == 'Spiking')
    
#     print(f"\n{'='*100}")
#     print(f"SUMMARY (Input: {input_size}×{input_size})")
#     print(f"{'='*100}")
#     print(f"  Total layers: {len(all_layers)}")
#     print(f"    Non-spiking: {non_spiking_count} layers")
#     print(f"    Spiking:     {spiking_count} layers")
#     print(f"")
#     print(f"  Non-spiking MACs:  {total_macs:>15,} ({total_macs/1e6:.2f} M)")
#     print(f"  Spiking SynOps:    {total_synops:>15,} ({total_synops/1e9:.3f} G)")
#     print(f"")
#     print(f"  MACs energy:       {energy_macs_mj:.4f} mJ")
#     print(f"  SynOps energy:     {energy_synops_mj:.4f} mJ")
#     print(f"  TOTAL SNN energy:  {total_energy_mj:.4f} mJ")
#     print(f"")
#     print(f"  ANN MACs:          {ann_macs/1e9:.3f} G")
#     print(f"  ANN Energy:        {ann_energy:.4f} mJ")
    
#     if total_energy_mj > 0:
#         ratio = ann_energy / total_energy_mj
#         if ratio >= 1:
#             print(f"  Energy Ratio:      {ratio:.1f}× LESS than ANN ✓")
#         else:
#             print(f"  Energy Ratio:      {ratio:.4f}× (SNN uses {1/ratio:.1f}× MORE energy)")
#     print(f"{'='*100}")
    
#     # Table row
#     print(f"\n{'='*100}")
#     print(f"PAPER-READY TABLE ROW:")
#     print(f"{'='*100}")
#     print(f"| ConvNeXt-T | ANN | -    | {ann_macs/1e9:.3f} | -         | {ann_energy:.4f} | 1.0× |")
#     print(f"| ConvNeXt-T | SNN | TTFS | {ann_macs/1e9:.3f} | {total_synops/1e9:.3f} | {total_energy_mj:.4f} | {ratio:.1f}× |")
#     print(f"{'='*100}")
    
#     return {
#         'input_size': input_size,
#         'total_macs': total_macs,
#         'total_synops': total_synops,
#         'energy_macs_mj': energy_macs_mj,
#         'energy_synops_mj': energy_synops_mj,
#         'total_energy_mj': total_energy_mj,
#         'ann_macs': ann_macs,
#         'ann_energy': ann_energy,
#         'ratio': ratio,
#         'all_layers': all_layers
#     }



# # class SparsityHook:
# #     """Measure activation sparsity: fraction of outputs == t_max (no spike)."""
# #     def __init__(self, layer_name, t_max=1.0):
# #         self.layer_name = layer_name
# #         self.t_max = t_max
# #         self.sparsity = 0.0

# #     def __call__(self, module, input, output):
# #         if not isinstance(output, torch.Tensor):
# #             return
# #         silent = (output >= self.t_max - 1e-6)
# #         num_silent = silent.sum().item()
# #         num_total = output.numel()
# #         self.sparsity = num_silent / num_total if num_total > 0 else 0.0


# # def register_sparsity_hooks(model, t_max):
# #     """Register hooks ONLY for SpikingBlock outputs."""
# #     hooks = {}
# #     for name, module in model.named_modules():
# #         if not name:
# #             continue
        
# #         # Only register for SpikingBlock
# #         if isinstance(module, SpikingBlock):
# #             hook = SparsityHook(name, t_max)
# #             module.register_forward_hook(hook)
# #             hooks[name] = hook
   
# #     print(f"Registered {len(hooks)} sparsity hooks")
# #     return hooks


# class DualSparsityHook:
#     """Measure activation sparsity at all spiking points in a SpikingBlock."""
#     def __init__(self, layer_name, t_max=1.0):
#         self.layer_name = layer_name
#         self.t_max = t_max
#         self.sparsity_mid = 0.0   # After pw1 spiking
#         self.sparsity_out = 0.0   # After pw2 spiking  
#         self.sparsity_final = 0.0 # Final block output

#     def __call__(self, module, input, output):
#         if not isinstance(output, torch.Tensor):
#             return
        
#         # Final output sparsity
#         silent_final = (output >= self.t_max - 1e-6)
#         num_silent_final = silent_final.sum().item()
#         num_total_final = output.numel()
#         self.sparsity_final = num_silent_final / num_total_final if num_total_final > 0 else 0.0
        
#         # Intermediate spiking outputs
#         if hasattr(module, 't_mid_spike') and module.t_mid_spike is not None:
#             silent_mid = (module.t_mid_spike >= self.t_max - 1e-6)
#             num_silent_mid = silent_mid.sum().item()
#             num_total_mid = module.t_mid_spike.numel()
#             self.sparsity_mid = num_silent_mid / num_total_mid if num_total_mid > 0 else 0.0
        
#         if hasattr(module, 't_out_spike') and module.t_out_spike is not None:
#             silent_out = (module.t_out_spike >= self.t_max - 1e-6)
#             num_silent_out = silent_out.sum().item()
#             num_total_out = module.t_out_spike.numel()
#             self.sparsity_out = num_silent_out / num_total_out if num_total_out > 0 else 0.0


# def register_sparsity_hooks(model, t_max):
#     """Register hooks for all SpikingBlocks."""
#     hooks = {}
#     for name, module in model.named_modules():
#         if not name:
#             continue
#         if isinstance(module, SpikingBlock):
#             hook = DualSparsityHook(name, t_max)
#             module.register_forward_hook(hook)
#             hooks[name] = hook
#     print(f"Registered {len(hooks)} sparsity hooks")
#     return hooks








# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


def call_spiking_torch(tj, W, D_i, t_min_prev, t_min, t_max):
    """
    PyTorch version of the TF `call_spiking`.
    Inputs:
      - tj: input spike times tensor (shape [..., C_in])  # last dim is channels/features
      - W: weight matrix (C_in, C_out) or (C_out, C_in) depending on matmul orientation
      - D_i: per-output delay (broadcastable to output shape)
      - t_min_prev: previous layer min spike time (unused for simplified form but kept for API)
      - t_min, t_max: scalars or tensors
    Returns:
      - ti: output spike times clamped by t_max
    Notes:
      - Ensure matmul dimension ordering matches your usage (here we use `... @ W` with W shape (C_in, C_out))
    """

    # threshold Eq. 18
    threshold = t_max - t_min - D_i

    # ensure floating dtype
    tj = tj.to(W.dtype)
    threshold = threshold.to(W.dtype)

    # compute ti = (tj - t_min) @ W + threshold + t_min
    # if last dim of tj is C_in and W is (C_in, C_out)
    delta = tj - t_min
    ti = torch.matmul(delta, W) + threshold + t_min

    # clamp to t_max
    ti = torch.where(ti < t_max, ti, t_max)
    return ti


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        # x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    
    

    
class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    # def __init__(self, in_chans=3, num_classes=1000, 
    #              depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
    #              layer_scale_init_value=1e-6, head_init_scale=1.,
    #              ):
    def __init__(self, in_chans=3, num_classes=1000, depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768), drop_path_rate=0.,
                 head_init_scale=1., layer_scale_init_value=1e-6, **kwargs):
        # ----- timm ≥ 0.9 compatibility -----
        # ----- timm ≥ 0.9 passes this key; ignore it -----
        kwargs.pop('pretrained_cfg', None)
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            # LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])
            # x = self.norm(x)
        return x # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class SpikingBlock(nn.Module):
    """
    Spiking variant of ConvNeXt Block using TTFS analytic mapping.
    Reuses weights from an existing Block instance.
    """
    def __init__(self, orig_block: Block, t_min=0.0, t_max=1.0, force_positive_weights: bool = False, init_delay: float = 0.0):
        super().__init__()
        # reuse modules (share weights)
        self.dwconv = orig_block.dwconv
        self.pw1 = orig_block.pwconv1
        self.pw2 = orig_block.pwconv2
        self.force_positive_weights = force_positive_weights
        self.gamma = getattr(orig_block, 'gamma', None)
        self.drop_path = orig_block.drop_path if hasattr(orig_block, 'drop_path') else nn.Identity()
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        # Optional read-only observer used by evaluation diagnostics. Keeping
        # this as None makes the submitted forward path the default behavior.
        self.temporal_diagnostics_observer = None

        self.D_mid = nn.Parameter(torch.zeros(self.pw1.out_features))
        self.D_out = nn.Parameter(torch.zeros(self.pw2.out_features))
        # Optionally initialize delays to a small positive value (helps push spikes later early)
        self._init_delay = float(init_delay)
        if self._init_delay > 0.0:
            with torch.no_grad():
                self.D_mid.data.fill_(self._init_delay)
                self.D_out.data.fill_(self._init_delay)


    def forward(self, tj):
        # tj: spike times tensor (N, C, H, W)
        x = tj
        # depthwise conv: apply to the spike-time map (heuristic)
        x = self.dwconv(x)

        # prepare for pointwise linear mapping per spatial location
        N, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C_in)

        # pw1
        # W1 = torch.relu(self.pw1.weight).t().contiguous()  # (C_in, C_mid)
        # optionally enforce non-negative pointwise weights (encourages sparsity / pruning)
        W1 = (torch.relu(self.pw1.weight) if self.force_positive_weights else self.pw1.weight).t().contiguous()  # (C_in, C_mid)
        device = x_flat.device
        dtype = x_flat.dtype
        # Use learned per-output delays (non-negative and bounded) instead of zeros
        # self.D_mid is a parameter created at init; clamp it to a sensible max
        D_mid = torch.clamp(torch.relu(self.D_mid), max=0.9 * (self.t_max - self.t_min)).to(device=device, dtype=dtype)
        t_min = torch.tensor(self.t_min, device=device, dtype=dtype)
        t_max = torch.tensor(self.t_max, device=device, dtype=dtype)
        t_mid = call_spiking_torch(x_flat, W1, D_mid, None, t_min, t_max)

        # pw2
        W2 = (torch.relu(self.pw2.weight) if self.force_positive_weights else self.pw2.weight).t().contiguous()  # (C_mid, C_out)
        D_out = torch.clamp(torch.relu(self.D_out), max=0.9 * (self.t_max - self.t_min)).to(device=device, dtype=dtype)
        t_out = call_spiking_torch(t_mid, W2, D_out, None, t_min, t_max)

        # reshape back
        t_out = t_out.view(N, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # out = tj + self.drop_path(t_out)
        out = torch.minimum(tj, self.drop_path(t_out))

        if self.temporal_diagnostics_observer is not None:
            self.temporal_diagnostics_observer(tj, t_out, out)

        # store latest output spike times for regularization/monitoring
        # Use the final output `out` (this is what hooks and forward actually return)
        try:
            self.latest_spike = out.detach()
        except Exception:
            self.latest_spike = None

        return out


class ConvNeXtSpiking(ConvNeXt):
    def __init__(self, *args, t_min=0.0, t_max=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_positive_weights = kwargs.get('force_positive_weights', False)
        # accept init_delay forwarded from model constructor
        self.init_delay = kwargs.get('init_delay', 0.0)
        # replace blocks in stages with SpikingBlock wrappers preserving weights
        for si, stage in enumerate(self.stages):
            new_blocks = []
            for b in stage:
                spb = SpikingBlock(b, t_min=t_min, t_max=t_max, force_positive_weights=self.force_positive_weights, init_delay=self.init_delay)
                new_blocks.append(spb)
            self.stages[si] = nn.Sequential(*new_blocks)
        # head: we need to map spike times to logits; we'll treat head as linear on features,
        # but input into head should be scores, so we convert times -> negative times as activation.
        # Keep self.head as-is but forward will convert spike-times to scores before head.
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def forward_features(self, x_t):
        # x_t: spike times tensor (N, C_in, H, W)
        # propagate
        for i in range(4):
            x_t = self.downsample_layers[i](x_t)   # these are convs on time maps
            x_t = self.stages[i](x_t)
        # global pool: average over spatial dims of spike times
        x_pool = x_t.mean([-2, -1])  # shape (N, C)
        return x_pool

    def forward(self, x_t):
        # x_t: spike times in [t_min, t_max]
        x_pool = self.forward_features(x_t)  # spike times per channel
        # convert times to scores: earlier spike -> higher score; simple mapping s = -t
        logits = self.head(-x_pool)
        return logits
    


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

