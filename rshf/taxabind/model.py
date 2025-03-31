import open_clip
from transformers import CLIPVisionModelWithProjection, ClapAudioModelWithProjection, ClapProcessor, PretrainedConfig
from ..sinr import SINR
from ..geoclip import GeoCLIP
from torchvision import transforms
import torchaudio

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