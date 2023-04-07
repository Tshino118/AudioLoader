import librosa
import numpy as np
import soundfile as sf
import inspect
from inspect import signature


class Audio_augment_manager:
    def __init__(
            self, audio, 
            processes={
                "append_noise":{"args":None},
                "apply_reverberation":{"args":None},
                "change_pitch":{"args":None},
                "change_speed":{"args":None}
            }
        ):
        self.__audio=audio
        self.__processes=processes

    def __call__(self):
        for aug_type, argments in self.__processes:
            pass

    def apply_processing():
        pass

def append_write_noise(audio):
    pass

def apply_reverberation(audio):
    pass

def change_pitch(audio):
    pass

def change_speed(audio, rate:float=0.75):
    return librosa.effects.time_stretch(audio, rate=rate)
