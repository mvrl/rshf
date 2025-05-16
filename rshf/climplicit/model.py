import torch

from .direct import Direct
from .siren import SirenNet
from transformers import PretrainedConfig
from huggingface_hub import PyTorchModelHubMixin

VAR_NAMES = [
    "cmi",
    "clt",
    "hurs",
    "pet",
    "pr",
    "rsds",
    "sfcWind",
    "tas",
    "tasmax",
    "tasmin",
    "vpd",
]

CHELSA_MEAN = torch.tensor(
    [
        [-264.1493656],
        [3912.44628016],
        [5921.65964573],
        [9385.47468266],
        [697.03653109],
        [15219.37926928],
        [3498.8511804],
        [2819.56006368],
        [2864.08583811],
        [2773.46759638],
        [8039.37322797],
    ]
).T

CHELSA_STD = torch.tensor(
    [
        [1042.67560332],
        [1767.94018571],
        [1185.91587823],
        [6639.79069994],
        [883.56243405],
        [7843.49167037],
        [1637.09237995],
        [174.43791946],
        [181.69448751],
        [167.07485901],
        [7516.98198719],
    ]
).T


class Climplicit(torch.nn.Module, PyTorchModelHubMixin):
    """
    CLIMPLICIT IMPLICITLY ENCODES GLOBAL CLIMATIC INFORMATION.

    config: Dict
    -> "return_chelsa": bool, defines whether the implicit embeddings
        or original CHELSA reconstructions should be returned by the forward pass.

    EXAMPLE USAGE:

    import torch
    from rshf.climplicit import Climplicit

    model = Climplicit.from_pretrained("Jobedo/climplicit", config={"return_chelsa": False})

    loc = [8.550155, 47.396702]  # Lon/Lat or our office
    april = 4  # April
    batchsize = 10

    # Call with a month
    month = torch.ones(batchsize) * april
    print("Output shape with month:", model(torch.tensor([loc] * batchsize), month).shape)
    # >>> Output shape with month: torch.Size([10, 256])

    # Call without month
    print("Output shape without month:", model(torch.tensor([loc] * batchsize)).shape)
    # >>> Output shape without month: torch.Size([10, 1024])

    # Return the CHELSA reconstruction instead of Climplicit embeddings
    model = Climplicit.from_pretrained("Jobedo/climplicit", config={"return_chelsa": True})
    print("Output shape of CHELSA reconstruction with month:", model(torch.tensor([loc] * batchsize), month).shape)
    # >>> Output shape of CHELSA reconstruction with month: torch.Size([10, 11])
    """
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        if type(config) is dict:
            self.config = PretrainedConfig().from_dict(config)

        self.location_encoder = SirenNet(
            dim_in=4,
            dim_hidden=512,
            dim_out=256,
            num_layers=16,
            dropout=False,
            h_siren=True,
            residual_connections=True,
        )
        
        self.pos_embedding = Direct(lon_min=-180, lon_max=180, lat_min=-90, lat_max=90)

        self.chelsa_regressor = torch.nn.Linear(256, 11)        

        self.return_chelsa = self.config.return_chelsa

    def forward(self, coordinates, month=None):
        # Apply the positional embedding
        loc = self.pos_embedding(coordinates)

        if month is None:
            res = []
            # Get the Climplicit embeddings for four months across the year
            for m in [3, 6, 9, 12]:
                month = torch.ones(len(coordinates)) * m
                loc_month = torch.concat(
                    [
                        loc,
                        torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1).to(loc.device),
                        torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1).to(loc.device),
                    ],
                    dim=-1,
                )
                x = self.location_encoder(loc_month)
                if self.return_chelsa:
                    x = self.chelsa_regressor(x)
                    x = x * CHELSA_STD + CHELSA_MEAN
                res.append(x)
            return torch.cat(res, dim=-1)

        # If we have a month
        # Append the month to the positional embedding
        loc_month = torch.concat(
            [
                loc,
                torch.sin(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
                torch.cos(month / 12 * torch.pi * 2).unsqueeze(dim=-1),
            ],
            dim=-1,
        )
        # Return the Climplicit embedding
        x = self.location_encoder(loc_month)
        if self.return_chelsa:
            x = self.chelsa_regressor(x)
            x = x * CHELSA_STD + CHELSA_MEAN
        return x


if __name__ == "__main__":
    print(Climplicit.from_pretrained("Jobedo/climplicit", config={"return_chelsa": False}))
