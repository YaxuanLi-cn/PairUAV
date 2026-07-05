import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A minimal LoRA wrapper for nn.Linear.

    The original Linear layer is kept frozen. Only lora_down/lora_up are trainable.
    This keeps the pretrained diffusion weights stable while allowing small adapters
    to learn the UAV target-view transformation.
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Standard LoRA init: down random, up zero so the wrapped model starts identical.
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

        for p in self.base_layer.parameters():
            p.requires_grad = False
        for p in self.lora_down.parameters():
            p.requires_grad = True
        for p in self.lora_up.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.base_layer(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scaling


def _is_lora_module(module: nn.Module) -> bool:
    return isinstance(module, LoRALinear)


def inject_lora_into_linear(module: nn.Module, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0,
                            name_prefix: str = "") -> int:
    """Recursively replace nn.Linear layers with LoRALinear.

    Returns the number of Linear layers wrapped. It is intentionally generic so it
    can be applied to UNet output blocks, ControlNet, or other selected modules.
    """
    wrapped = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{name_prefix}.{child_name}" if name_prefix else child_name
        if isinstance(child, nn.Linear):
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            wrapped += 1
        elif not _is_lora_module(child):
            wrapped += inject_lora_into_linear(child, rank=rank, alpha=alpha, dropout=dropout,
                                               name_prefix=full_name)
    return wrapped


def mark_only_lora_trainable(module: nn.Module) -> int:
    """Set only LoRA adapter weights trainable inside a module.

    Returns number of trainable LoRA parameter tensors.
    """
    count = 0
    for name, p in module.named_parameters():
        if "lora_down" in name or "lora_up" in name:
            p.requires_grad = True
            count += 1
        else:
            p.requires_grad = False
    return count
