import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Union
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from transformers import PretrainedConfig


def rad_to_cart(coords):
    """Convert lat/lon coordinates in radians to Cartesian (x, y, z) coordinates.
    
    Args:
        coords: Array of shape (N, 2) with [lon, lat] in radians
        
    Returns:
        Array of shape (N, 3) with [x, y, z] Cartesian coordinates
    """
    lon, lat = coords[:, 0], coords[:, 1]
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=1)


class RANGEConfig(PretrainedConfig):
    """Configuration class for RANGE model.
    
    Args:
        model_type: Type of RANGE model ('RANGE' or 'RANGE+')
        beta: Beta value for RANGE+ interpolation (default: 0.5)
        temp: Temperature for semantic similarity (default: 12.0 for RANGE+, 15.0 for RANGE)
        geo_temp: Temperature for geographic similarity (RANGE+ only, default: 40.0)
        db_size: Size of the database ('large' or 'med')
    """
    def __init__(
        self,
        model_type: str = "RANGE+",
        beta: float = 0.5,
        temp: Optional[float] = None,
        geo_temp: Optional[float] = None,
        db_size: str = "large",
        **kwargs
    ):
        super().__init__()
        self.model_type = model_type
        self.beta = beta
        self.db_size = db_size
        
        if model_type == "RANGE+":
            self.temp = temp if temp is not None else 12.0
            self.geo_temp = geo_temp if geo_temp is not None else 40.0
        elif model_type == "RANGE":
            self.temp = temp if temp is not None else 15.0
            self.geo_temp = None
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'RANGE' or 'RANGE+'")
    
    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self


class RANGE(nn.Module, PyTorchModelHubMixin):
    """RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings
    
    RANGE is a geospatial embedding model that generates visual embeddings for geographic 
    coordinates by retrieving and aggregating features from similar locations in a database.
    
    Example Usage:
        >>> from rshf.range import RANGE, RANGEConfig
        >>> config = RANGEConfig(model_type="RANGE+", beta=0.5)
        >>> model = RANGE(config)
    """
    
    def __init__(self, config: Union[RANGEConfig, dict]):
        """Initialize RANGE model.
        
        Args:
            config: RANGEConfig object or dict containing configuration
        """
        super(RANGE, self).__init__()
        
        if isinstance(config, dict):
            self.config = RANGEConfig(**config)
        else:
            self.config = config
            
        self.model_type = self.config.model_type
        self.beta = self.config.beta
        self.temp = self.config.temp
        self.geo_temp = self.config.geo_temp
        
        # Initialize SatCLIP location encoder
        from rshf.satclip import SatClip
        self.loc_model = SatClip.from_pretrained("MVRL/satclip-loc-enc-vit16-l40").double()
        self.loc_model.eval()
        for param in self.loc_model.parameters():
            param.requires_grad = False
            
        # Database will be loaded when needed
        self.db_loaded = False
        self.db_satclip_embeddings = None
        self.db_high_resolution_satclip_embeddings = None
        self.db_locs_xyz = None
        
        # Output dimension
        if self.model_type in ["RANGE", "RANGE+"]:
            self.location_feature_dim = 1280  # 1024 (high-res) + 256 (SatCLIP)
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        """Load a pretrained RANGE model.
        
        Args:
            pretrained_model_name_or_path: Path to the pretrained model
        """
        raise NotImplementedError("""
            This method is not available for RANGE model. Use config instead to initialize the model.
            Example Usage:
                >>> from rshf.range import RANGE, RANGEConfig
                >>> config = RANGEConfig(model_type="RANGE+", beta=0.5)
                >>> model = RANGE(config)
            """)
    
    def load_database(self, db_path: Optional[str] = None):
        """Load the RANGE database.
        
        Args:
            db_path: Path to the database .npz file. If None, will download from HuggingFace.
        """
        if self.db_loaded:
            return
            
        if db_path is None:
            db_filename = f"range_db_{self.config.db_size}.npz"
            db_path = hf_hub_download(
                'MVRL/RANGE-database',
                db_filename,
                repo_type='dataset'
            )
        
        # Load database
        range_db = np.load(db_path, allow_pickle=True)
        db_locs_latlon = range_db['locs'].astype(np.float32)
        
        # Load and normalize SatCLIP embeddings
        db_satclip_embeddings = range_db['satclip_embeddings'].astype(np.float32)
        db_satclip_embeddings = db_satclip_embeddings / np.linalg.norm(
            db_satclip_embeddings, ord=2, axis=1, keepdims=True
        )
        self.db_satclip_embeddings = torch.tensor(db_satclip_embeddings)
        
        # Load high-resolution image embeddings
        self.db_high_resolution_satclip_embeddings = torch.tensor(
            range_db['image_embeddings'].astype(np.float32)
        )
        
        # Convert database locations to Cartesian coordinates
        db_locs_rad = db_locs_latlon * math.pi / 180
        db_locs_xyz = rad_to_cart(db_locs_rad)
        self.db_locs_xyz = torch.tensor(db_locs_xyz)
        
        self.db_loaded = True
    
    def forward(self, coords: torch.Tensor, db_path: Optional[str] = None) -> torch.Tensor:
        """Generate RANGE embeddings for given coordinates.
        
        Args:
            coords: Tensor of shape (N, 2) with [lon, lat] coordinates in degrees
            db_path: Optional path to database file (will download if not provided)
            
        Returns:
            Tensor of shape (N, location_feature_dim) with location embeddings
        """
        # Ensure database is loaded
        if not self.db_loaded:
            self.load_database(db_path)
        
        # Move database to device
        device = coords.device
        db_satclip_embeddings = self.db_satclip_embeddings.to(device)
        db_high_res_embeddings = self.db_high_resolution_satclip_embeddings.to(device)
        
        # Get SatCLIP embeddings for query locations
        curr_loc_embeddings = self.loc_model(coords.double()).float()
        
        # Normalize and compute semantic similarity
        curr_loc_embeddings = curr_loc_embeddings / curr_loc_embeddings.norm(p=2, dim=-1, keepdim=True)
        semantic_similarity = curr_loc_embeddings.float() @ db_satclip_embeddings.t()
        
        # Scale similarity using temperature and convert to probabilities
        semantic_similarity = torch.nn.functional.softmax(semantic_similarity * self.temp, dim=-1)
        
        # Compute high-resolution embeddings as weighted sum
        high_res_embeddings = semantic_similarity @ db_high_res_embeddings
        
        if self.model_type == "RANGE":
            # RANGE: concatenate high-res features with low-res location features
            loc_embeddings = torch.cat([high_res_embeddings, curr_loc_embeddings], dim=1)
            
        elif self.model_type == "RANGE+":
            # RANGE+: also use geographic proximity
            db_locs_xyz = self.db_locs_xyz.to(device)
            
            # Convert query locations to Cartesian coordinates
            query_locs_rad = coords.cpu().numpy() * math.pi / 180
            query_locs_xyz = torch.tensor(rad_to_cart(query_locs_rad)).float().to(device)
            
            # Compute angular similarity
            angular_similarity = query_locs_xyz @ db_locs_xyz.T
            angular_similarity = torch.nn.functional.softmax(angular_similarity * self.geo_temp, dim=-1)
            
            # Get geographically-weighted high-res embeddings
            angular_high_res_embeddings = angular_similarity @ db_high_res_embeddings
            
            # Interpolate between semantic and geographic embeddings
            averaged_high_res_embeddings = (
                (1 - self.beta) * angular_high_res_embeddings +
                self.beta * high_res_embeddings
            )
            
            # Concatenate with low-res location features
            loc_embeddings = torch.cat([averaged_high_res_embeddings, curr_loc_embeddings], dim=1)
        
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        return loc_embeddings
