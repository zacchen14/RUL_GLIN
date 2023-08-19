from typing import Optional, Any, Union, Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
from common.utils.utils import exists
from utils import FixedPositionalEncoding, prior_association, generate_prior
from torch.nn.init import xavier_uniform_
from model.Transformer import Transformer


class RULformer(nn.Module):
    r"""A transformer-based remaining useful life prediction model. User is able to modify the attributes as needed.

    Args:
        d_model: the number of feature dimension of inputs (default=14).
        d_emb: the number of expected features in the encoder/decoder inputs (default=128).
        nhead: the number of heads in the multi-head attention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=256).
        dim_flatten: the dimension of the regressor linear model (default=64).
        dropout: the dropout value (default=0.1).
        window_size: sliding time window size (default=30).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        norm: the normalization function, can be a string ("batch" or "layer"). Default: None
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        batch_norm_eps: the eps value in batch normalization components (default=1e-5).
        batch_first: If ``True``, then the input and save_path tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).

    Examples::
        >>> transformer_model = RULformer(d_model=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note:
        A full example to apply nn.Transformer module for the word language model is available in
        https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 14, d_emb: int = 128, max_life: Optional[int] = None,
                 nhead: int = 8, num_encoder_layers: int = 2, num_decoder_layers: int = 1,
                 dim_feedforward: int = 256, dim_flatten: int = 64, dropout: float = 0.1,
                 window_size: int = 30,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 norm: Optional[str] = None,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 batch_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RULformer, self).__init__()

        self.d_model = d_model
        self.max_life = max_life
        self.device = device
        self.window_size = window_size

        self.num_decoder_layers = num_decoder_layers

        self.pe = FixedPositionalEncoding(d_model=d_emb, dropout=dropout, max_len=1024, batch_first=batch_first)
        self.embedding = nn.Linear(d_model, d_emb)

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            Linear(dim_flatten, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            # nn.Sigmoid() # sigmoid is effectiveless, but at association discrepency, to avoid the output is out of
            # boarder, such that the Gaussian kernel generates NaN.
        )

        self._reset_parameters()
        self.batch_first = batch_first
        self.Transformer = Transformer(d_model=d_emb, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                       dropout=dropout, activation=activation, custom_encoder=custom_encoder,
                                       custom_decoder=custom_decoder, norm=norm, norm_eps=batch_norm_eps,
                                       batch_first=batch_first, norm_first=norm_first, **factory_kwargs)
        self.out_proj = Linear(d_emb, d_model)

    def forward(self, src: Tensor, tgt: Tensor, label: Tensor = None,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                need_ass_dis: bool = False):
        """Take in and process masked source/target sequences. From source code by Pytorch official.
           More Information: http://www. TODO: complete the url.

            Args:
                src: the sequence to the encoder (required).
                tgt: the sequence to the decoder (required).
                src_mask: the additive mask for the src sequence (optional).
                tgt_mask: the additive mask for the tgt sequence (optional).
                memory_mask: the additive mask for the encoder save_path (optional).
                src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
                tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
                memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).
                need_ass_dis: return cross-attention in decoder matrix in (B, N, N) to if `need_weights=True` or None (optional).

            Shape:
                - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
                  `(N, S, E)` if `batch_first=True`.
                - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
                  `(N, T, E)` if `batch_first=True`.
                - src_mask: :math:`(S, S)`.
                - tgt_mask: :math:`(T, T)`.
                - memory_mask: :math:`(T, S)`.
                - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
                - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
                - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
                - mha_attn: (T, T) for unbatched input otherwise :math:`(N, T, T)

                Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
                positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
                while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
                are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
                is provided, it will be added to the attention weight.
                [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
                the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
                positions will be unchanged. If a BoolTensor is provided, the positions with the
                value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

                - save_path: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
                  `(N, T, E)` if `batch_first=True`.

                Note: Due to the multi-head attention architecture in the transformer model,
                the save_path sequence length of a transformer is same as the input sequence
                (i.e. target) length of the decode.

                where S is the source sequence length, T is the target sequence length, N is the
                batch size, E is the feature number

            """

        src = self.pe(self.embedding(src))
        tgt = self.pe(self.embedding(tgt))

        """if need_ass_dis is True:
            # Do NOT use Transformer provided by Pytorch official for Batch Normalization should be used.
            output, sigma, mha_attn = self.Transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                                       src_key_padding_mask=src_key_padding_mask,
                                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                                       memory_mask=memory_mask,
                                                       memory_key_padding_mask=memory_key_padding_mask,
                                                       need_ass_dis=need_ass_dis)
            # mha_attn: (layers, mini-batch, T, L), sigma: (layers, mini-batch, T, 1)

        else:
            output = self.Transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                      src_key_padding_mask=src_key_padding_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask,
                                      need_ass_dis=need_ass_dis)
            mha_attn = None"""

        output, sigma, mha_attn = self.Transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                                   src_key_padding_mask=src_key_padding_mask,
                                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                                   memory_mask=memory_mask,
                                                   memory_key_padding_mask=memory_key_padding_mask,
                                                   need_ass_dis=need_ass_dis)
        # mha_attn: (layers, mini-batch, T, L), sigma: (layers, mini-batch, T, 1)

        output = self.mlp_head(output)  # output: (batch_size, 1)

        # prior cross-correlation
        prior = None
        if need_ass_dis is True and label is not None:
            sigma = torch.stack(sigma)
            sigma = sigma.transpose(0, 1)  # sigma: (mini-batch, layers, T, 1)
            sigma = torch.clamp(sigma, 0.25, 100)
            sigma = sigma.cpu()

            if exists(self.max_life):  # with max_life, rul label is 0-1
                prior = torch.stack(
                    [torch.stack(
                        [prior_association(mha_attn.size(-2),
                                           mha_attn.size(-1),
                                           mha_attn.size(-1) - label[
                                               i].cpu().detach().numpy() * self.max_life - self.window_size,
                                           sigma[i, j]
                                           # 5
                                           )
                         for i in range(mha_attn.size(1))]  # mha_attn: (layers, batch_size, T, L)
                    ) for j in range(mha_attn.size(0))]
                ).to(self.device)

            else:  # without max_life, rul label is 0-100+
                prior = torch.stack(
                    [prior_association(mha_attn.size(-2),
                                       mha_attn.size(-1),
                                       mha_attn.size(-1) - label[i].cpu().detach().numpy() - self.window_size,
                                       # sigma
                                       5
                                       )
                     for i in range(mha_attn.size(1))]
                ).to(self.device)

            """prior = generate_prior(sigma, label, mha_attn.size(-2), mha_attn.size(-1), self.num_decoder_layers,
                                   max_life=self.max_life, window_size=self.window_size).to(self.device)"""
            # prior: (layers, batch_size, T, L)

        """if exists(mha_attn) and label is not None:
            return output, prior, mha_attn, sigma
        elif exists(mha_attn) and label is None:
            return output, mha_attn, sigma
        else:
            return output"""
        return output, prior, mha_attn, sigma

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    @staticmethod
    def create_mask(src, tgt, PAD_IDX: int = 0, batch_first: bool = False, device=None, dtype=None):
        """create masks

        Args:
            src: src in Transformer (required)
            tgt: tgt in Transformer (required)
            PAD_IDX: (optional)
            batch_first: (optional)
            device: (optional)

        Returns:
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

        Shape:
            - src: (N, L, C) if batch_first = ``True``, otherwise (L, N, C)
            - tgt: (N, L, C) if batch_first = ``True``, otherwise (L, N, C)
        """
        if batch_first:
            src_seq_len = src.shape[1]
            tgt_seq_len = tgt.shape[1]
        else:
            src_seq_len = src.shape[0]
            tgt_seq_len = tgt.shape[0]

        src_mask = RULformer.generate_square_subsequent_mask(src_seq_len)
        tgt_mask = RULformer.generate_square_subsequent_mask(tgt_seq_len)

        src_padding_mask = torch.all(src == PAD_IDX, dim=-1)
        tgt_padding_mask = torch.all(tgt == PAD_IDX, dim=-1)
        return src_mask.to(device), tgt_mask.to(device), src_padding_mask.to(device), tgt_padding_mask.to(device)
