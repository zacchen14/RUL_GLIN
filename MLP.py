import torch
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable
from utils import exists, _get_clones


class MLPLayer(nn.Module):
    """

    """
    def __init__(self, n_inputs: int, n_hiddens: int, act=nn.ReLU, drop:float = 0.1):
        super(MLPLayer, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hiddens)
        self.act = act()
        self.fc2 = nn.Linear(n_hiddens, n_inputs)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x + self.drop2(x)


class MLP(nn.Module):
    """MLP is a stack of N MLP layers.

    """
    def __init__(self, n_inputs: int, n_hidden: int = 64, num_layers: int = 3, norm=None,
                 custom_MLPBlock: Optional[Any] = None):
        super(MLP, self).__init__()
        if exists(custom_MLPBlock):
            mlp_block = custom_MLPBlock
        else:
            mlp_block = MLPLayer(n_inputs, n_hidden)
        self.layers = _get_clones(mlp_block, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, emb: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the input through the encoder layers in turn.

        Args:
            emb: the sequence to the MLP (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = emb

        for mod in self.layers:
            output = mod(output)

        if self.norm is not None:
            output = self.norm(output)

        return output