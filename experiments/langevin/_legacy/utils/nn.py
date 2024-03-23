from typing import Sequence
import flax.linen as nn

class MLP(nn.Module):
    layer_widths: Sequence[int]
    @nn.compact
    def __call__(self, x):
        for i, width in enumerate(self.layer_widths):
            x = nn.Dense(width, name=f'lin_{i}')(x)
            if i != len(self.layer_widths) - 1:
                x = nn.celu(x)
        return x