

from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write

import augment
from aug_type import AugType
from path_info import PathInfo


class Augmentation:
    def __init__(self):
        pass

    #--------------------------------------------------
    # use librosa
    #--------------------------------------------------
def noise_injection(data):
    noise_factor = np.random.uniform(0.001, 0.009)       ##(0.001, 0.008)
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def shifting_time(data, shift_direction='left'):
    shift_max=np.random.uniform(0.1, 0.9)
    shift = np.random.randint(self.__source_info["rate"] * shift_max)
    if shift_direction == "right":
        shift = -shift
    elif shift_direction == "both":
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def changing_pitch(data):
    return librosa.effects.pitch_shift(data, sr=self.__source_info["rate"], n_steps=np.random.uniform(2, 10))

def changing_speed(data):
    return librosa.effects.time_stretch(data, rate=np.random.uniform(0.1, 10))               ##0.8, 1.5

#--------------------------------------------------
# use EffectChain
#--------------------------------------------------
def apply_noise(data):
    noise_generator = lambda: torch.zeros_like(data).uniform_()
    y_noise = augment.EffectChain().additive_noise(noise_generator, snr=15).apply(data, src_info=self.__source_info)
    return y_noise

def apply_time(data):
    y_time = augment.EffectChain().time_dropout(max_seconds=0.5).apply(data, src_info=self.__source_info)
    return y_time

def apply_pitch(data):
    # output signal properties
    target_info = {'channels': 1,
                    'length': 0,  # not known beforehand
                    'rate': 16_000}
    random_pitch = lambda: np.random.randint(-400, -200)
    y_pitch = augment.EffectChain().pitch(random_pitch).rate(16_000).apply(data, src_info=self.__source_info,
                                                                            target_info=target_info)
    return y_pitch

def apply_band(data):
    y_band = augment.EffectChain().sinc('-a', '120', '500-100').apply(data, src_info=self.__source_info)
    return y_band

def apply_clip(data):
    y_clip = augment.EffectChain().clip(0.5).apply(data, src_info=self.__source_info)
    return y_clip

def apply_reverb(data):
    y_reverb = augment.EffectChain().reverb(70, 70, 70).channels(1).apply(data, src_info=self.__source_info)
    return y_reverb

if __name__ == "__main__":
    Augmentation()
