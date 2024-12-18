import numpy as np
import torch


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class HannWindowEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        # get hann window weights
        kick_in_epoch = torch.tensor(self.kwargs["kick_in_epoch"], dtype=torch.float32)
        t = torch.clamp(self.kwargs["iter_epoch"] - kick_in_epoch, min=0.0)
        N = self.kwargs["full_band_epoch"] - kick_in_epoch
        m = N_freqs
        alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (
                1.0 - torch.cos(np.pi * torch.clamp(alpha - freq_idx, min=0.0, max=1.0))
            ) / 2.0
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq, w=w: w * p_fn(x * freq)
                )
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(
    multires,
    input_dims=3,
    mode="fourier",
    kick_in_epoch=None,
    iter_epoch=None,
    full_band_epoch=None,
):
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }
    if mode == "fourier":
        embedder_obj = Embedder(**embed_kwargs)
    elif mode == "hann":
        embed_kwargs["include_input"] = False
        if kick_in_epoch is not None:
            embed_kwargs["kick_in_epoch"] = kick_in_epoch
        if iter_epoch is not None:
            embed_kwargs["iter_epoch"] = iter_epoch
        if full_band_epoch is not None:
            embed_kwargs["full_band_epoch"] = full_band_epoch
        embedder_obj = HannWindowEmbedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim
