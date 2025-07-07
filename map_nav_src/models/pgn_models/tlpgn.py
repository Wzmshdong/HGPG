import functools

import torch
from torch import nn

from pgn_models.pgn_switch import pgn_switch


def _get_act_fn(act_fn):
    if act_fn == 'softmax':
        return functools.partial(torch.softmax, dim=-1)
    elif act_fn == 'sigmoid':
        return torch.sigmoid
    else:
        raise ValueError("Invalid IDP activation function")

class TLPGN(nn.Module):
    def __init__(
            self,
            nr_output_vectors,
            mixture_size,
            vector_dim,
            model_type,
            pgn_act_fn,
            **kwargs,
    ) -> None:
        super().__init__()
        self.nr_output_vectors = nr_output_vectors
        self.mixture_size = mixture_size
        self.vector_dim = vector_dim
        self.model = pgn_switch(model_type,
                               out_features=nr_output_vectors * mixture_size,
                               **kwargs)

        tl_vectors = torch.empty(
            mixture_size,
            vector_dim,
            dtype=torch.float32,
            device='cuda',
        )
        torch.nn.init.normal_(tl_vectors, std=0.02)
        self.tl_vectors = torch.nn.Parameter(tl_vectors)
        self.act_fn = self._get_act_fn(pgn_act_fn)

    def forward(self, images):
        
        images = images.unsqueeze(3).permute(0,2,1,3)
        logits = self.model(images)
        split_logits = logits.reshape(
            len(logits),
            self.nr_output_vectors,
            self.mixture_size
        )
        mixture_coeffs = self.act_fn(
            split_logits
        )
        pgn_prompts = torch.einsum(
            'bom,mv->bov',
            [mixture_coeffs, self.tl_vectors]
        )
        return pgn_prompts

    @staticmethod
    def _get_act_fn(act_fn):
        if act_fn == 'softmax':
            return functools.partial(torch.softmax, dim=-1)
        elif act_fn == 'sigmoid':
            return torch.sigmoid
        else:
            raise ValueError("Invalid PGN activation function")
