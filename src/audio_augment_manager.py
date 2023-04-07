import librosa
import numpy as np
import soundfile as sf
import inspect
from inspect import signature

class Audio_augment_manager:
    def __init__(
      self, audio, augment_bp:dict
    ):
        self.__audio=audio
        self.__augment_bp = augment_bp
    
    def get_original_audio(self):
        return self.__audio

    def get_augment_blueprint(self):
        return self.__augment_bp

    def __call__(self):
        for aug_type, argments in self.__augment_bp.items():
            pass

    def get_augment_type_dict(self):
        return {
            "d":signature(),
            "ds":signature()
        }

def append_write_noise():
    pass

def apply_reverberation():
    pass

def change_pitch():
    pass

def change_speed(self, rate:float=0.75):
    return librosa.effects.time_stretch(self.__audio, rate=rate)

