import open_clip
from transformers import CLIPVisionModelWithProjection, ClapAudioModelWithProjection, ClapProcessor, PretrainedConfig
from ..sinr import SINR
from ..geoclip import GeoCLIP
from torchvision import transforms
import torchaudio


class TaxaBindConfig(PretrainedConfig):
    """
    Configuration class to store the configuration of a `TaxaBind` model.

    Arguments:
        clip: str (default: 'ViT-B-16'). OpenCLIP model name for the image/text encoder.
        image_resize: int (default: 224). Resize dimension for image preprocessing.
        image_crop: int (default: 224). Center crop size for image preprocessing.
        normalize_mean: list (default: [0.485, 0.456, 0.406]). Normalization mean for images.
        normalize_std: list (default: [0.229, 0.224, 0.225]). Normalization std for images.
        audio_encoder: str (default: None). HuggingFace model name for the audio encoder.
        get_audio_processor: str (default: None). HuggingFace model name for the audio processor.
        audio_sample_rate: int (default: 48000). Target audio sample rate in Hz.
        audio_max_length_s: int (default: 10). Maximum audio clip length in seconds.
        audio_padding: str (default: 'repeatpad'). Padding strategy for audio inputs.
        audio_truncation: bool (default: True). Whether to truncate audio to max_length_s.
        location_encoder: str (default: None). HuggingFace repo ID for the GeoCLIP location encoder.
        sinr: str (default: None). HuggingFace repo ID for the SINR environment encoder.
        sat_encoder: str (default: None). HuggingFace model name for the satellite image encoder.
        sat_resize: int (default: 224). Resize dimension for satellite image preprocessing.
        sat_crop: int (default: 224). Center crop size for satellite image preprocessing.
    """
    def __init__(
        self,
        clip='ViT-B-16',
        image_resize=224,
        image_crop=224,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        audio_encoder=None,
        get_audio_processor=None,
        audio_sample_rate=48000,
        audio_max_length_s=10,
        audio_padding='repeatpad',
        audio_truncation=True,
        location_encoder=None,
        sinr=None,
        sat_encoder=None,
        sat_resize=224,
        sat_crop=224,
    ):
        super(TaxaBindConfig, self).__init__()
        self.clip = clip
        self.image_resize = image_resize
        self.image_crop = image_crop
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.audio_encoder = audio_encoder
        self.get_audio_processor = get_audio_processor
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_length_s = audio_max_length_s
        self.audio_padding = audio_padding
        self.audio_truncation = audio_truncation
        self.location_encoder = location_encoder
        self.sinr = sinr
        self.sat_encoder = sat_encoder
        self.sat_resize = sat_resize
        self.sat_crop = sat_crop

    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
        return self

class TaxaBind:
    def __init__(self, config: PretrainedConfig):
        super(TaxaBind, self).__init__()
        self.config = config
        if type(config) is dict:
            self.config = PretrainedConfig().from_dict(config)
    
    def get_image_text_encoder(self):
        return open_clip.create_model_and_transforms(self.config.clip)[0]
    
    def get_tokenizer(self):
        return open_clip.get_tokenizer(self.config.clip)
    
    def get_image_processor(self):
        return transforms.Compose([transforms.Resize((self.config.image_resize, self.config.image_resize)),
        transforms.CenterCrop(self.config.image_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std)])
    
    def get_audio_encoder(self):
        return ClapAudioModelWithProjection.from_pretrained(self.config.audio_encoder)
    
    def process_audio(self, track, sr):
        processor = ClapProcessor.from_pretrained(self.config.get_audio_processor)
        track = track.mean(axis=0)
        track = torchaudio.functional.resample(track, orig_freq=sr, new_freq=self.config.audio_sample_rate)
        output = processor(audios=track, sampling_rate=self.config.audio_sample_rate, max_length_s=self.config.audio_max_length_s, return_tensors="pt",padding=self.config.audio_padding,truncation=self.config.audio_truncation)
        return output
    
    def get_audio_processor(self):
        return self.process_audio
    
    def get_location_encoder(self):
        return GeoCLIP.from_pretrained(self.config.location_encoder)
    
    def get_env_encoder(self):
        return SINR.from_pretrained(self.config.sinr)
    
    def get_sat_encoder(self):
        return CLIPVisionModelWithProjection.from_pretrained(self.config.sat_encoder)
    
    def get_sat_processor(self):
        return transforms.Compose([transforms.Resize((self.config.sat_resize, self.config.sat_resize)),
        transforms.CenterCrop(self.config.sat_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std)])