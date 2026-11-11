"""Contribution 4 - Parameter-efficient adaptation with Graph LoRA.

W' = W + (alpha/r) * B A, with A in R^{r x in}, B in R^{out x r}; B is zero-initialised so the adapted model
starts exactly equal to the pretrained one. The base weights are frozen; only A and B are trained.
Can be toggled on/off (set_lora_enabled) so the same network gives the zero-shot (off) or adapted (on) output.
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=8.0):
        super().__init__()
        self.base, self.r, self.scaling, self.enabled = base, r, alpha / r, True
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # keep adapters on the same device/dtype as the wrapped layer
        self.to(base.weight.device)

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            out = out + self.scaling * (x @ self.lora_A.t() @ self.lora_B.t())
        return out

    @torch.no_grad()
    def merge(self):
        """Fold the adapter into the base weight (inference-time, zero overhead)."""
        self.base.weight += self.scaling * (self.lora_B @ self.lora_A)
        self.enabled = False


def inject_graph_lora(model, scope=('gcn1', 'gcn2'), r=4, alpha=8.0):
    """Wrap every nn.Linear inside model.<scope> with LoRA. Returns the list of LoRALinear modules."""
    wrapped = []

    def _walk(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                lora = LoRALinear(child, r, alpha)
                setattr(parent, name, lora)
                wrapped.append(lora)
            else:
                _walk(child)

    for s in scope:
        _walk(getattr(model, s))
    return wrapped


def set_lora_enabled(model, enabled: bool):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = enabled


def freeze_base_train_lora(model):
    """Freeze everything except LoRA matrices; returns the trainable parameter list."""
    for p in model.parameters():
        p.requires_grad = False
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad = m.lora_B.requires_grad = True
            params += [m.lora_A, m.lora_B]
    return params


def count_trainable(params_or_module):
    ps = params_or_module.parameters() if isinstance(params_or_module, nn.Module) else params_or_module
    return sum(p.numel() for p in ps if p.requires_grad)
