from torch import nn, optim
import math
import torch
import torch.nn.functional as F


class SirenNet(nn.Module):
    """Sinusoidal Representation Network (SIREN) with residual connections (ReSIREN).
    Adapted from: https://github.com/microsoft/satclip/blob/main/satclip/location_encoder.py

        residual_connections: Turn the network into a ReSIREN model
        h_siren: Use the H-SIREN activation after the first layer
        return_hidden_embs: Append lower-resolution embeddings from lower layers to the output
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        dropout=False,
        residual_connections=False,
        h_siren=False,
        return_hidden_embs=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.return_hidden_embs = return_hidden_embs

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    h_siren=h_siren,
                    dropout=dropout,
                    residual_connections=residual_connections,
                )
            )

        final_activation = nn.Identity() if not final_activation else final_activation
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
            dropout=False,
        )

    def forward(self, x, mods=None):

        res = []
        gaussian = None
        # Passing the gaussian (vector after dot product and before sin) between layers as residual connection
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, gaussian = layer(x, gaussian)
            if self.return_hidden_embs is not None:
                if i in self.return_hidden_embs:
                    res.append(x)

        x, _ = self.last_layer(x, gaussian)

        if len(res) > 0:
            res.append(x)
            x = torch.cat(res, dim=1)
        return x


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=False,
        residual_connections=False,
        h_siren=False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dim_out = dim_out
        self.dropout = dropout
        self.h_siren = h_siren

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

        self.residual_connections = residual_connections

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if not bias is None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x, prev_gaussian=None):
        # Apply base linear transform
        out = F.linear(x, self.weight, self.bias)
        # Use dropout (not recommended for reconstruction)
        if self.dropout:
            out = F.dropout(out, training=self.training)
        # If using ReSIREN, merge with the previous layers gaussian
        if (
            self.residual_connections
            and prev_gaussian is not None
            and out.shape == prev_gaussian.shape
        ):
            out = (out + prev_gaussian) / 2
        # Save the gaussian to pass it on
        gaussian = out
        # Apply the additional transformation for H-SIREN
        if self.h_siren and self.is_first:
            out = torch.sinh(2 * out)
        # Apply the activation (for nearly all layers Sine)
        out = self.activation(out)
        return out, gaussian
