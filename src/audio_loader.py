import librosa
import numpy as np
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt

class AudioLoader:
    def __init__(self, filename):
        self.filename = filename
        self.audio, self.sampling_rate = librosa.load(filename, sr=16000, mono=True)
    
    def get_audio(self):
        return self.audio
    
    def get_sampling_rate(self):
        return self.sampling_rate

def get_melspectrogram(data, samplerate):
    melspectrogram = librosa.feature.melspectrogram(y=data, sr=samplerate)
    return melspectrogram

def save_melspectrogram(path, data, samplerate):
    plt.figure()
    melspectrogram = get_melspectrogram(data, samplerate)
    S_dB = librosa.power_to_db(melspectrogram, ref=np.max)
    librosa.display.specshow(S_dB, sr=samplerate, fmax=8000)
    plt.xlabel("Time [sec]")
    plt.ylabel("Mel")
    plt.savefig(path)
    plt.close()
    pass

def norm_audio(data):
    max_data = np.max(data)
    data = data/(max_data+1e-6)
    return data

def save_audio(path, data, samplerate, subtype='PCM_24'):
    return sf.write(path, data, samplerate, subtype=subtype)

def save_image_signal(path, data, samplerate):
    plt.figure(figsize=(8,4), )
    librosa.display.waveshow(y=data, sr=samplerate)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.savefig(path)
    plt.close()
    return 

def save_image_mel_spectrogram(path, data, samplerate):
    plt.figure(figsize=(8,4), )
    librosa.display.waveshow(y=data, sr=samplerate)
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.savefig(path)
    plt.close()
    return 

def n_cross_vad(data, N=0.02, frame = 320, threshold = 0.99):
    """
    VAD for Ncross Method
    '''
    data        :音声データ(ndarray)
    N           :N-Crossライン.背景ノイズの大きさにあわせて決めてあげてください(0~1)
    frame       :分析窓.この範囲でどれくらいN-Crossライン上の正負が切り替わったかを分析します
    threshold   :分析窓の中でどのくらい正負が切り替われば音声とみなすかの閾値です(0~1)

    return ノイズ部分がバッサリ切られたデータ
    """
    data /= np.max(np.abs(data))
    data += N

    data_pad1 = np.sign(np.pad(data, [0,1], 'constant'))
    data_pad2 = np.sign(np.pad(data, [1,0], 'constant'))
    dif = data_pad1-data_pad2

    zerocross_list = np.where(dif==0, 1, 0)[:-1]

    cnt = len(zerocross_list)

    ret=[]
    n_cross_dict={}

    for idx in range(int(cnt/frame)):
        start=idx*frame
        end = (idx+1)*frame
        seg = zerocross_list[start:end]
        add = np.sum(seg)/frame
        zero_cross_flag = "silence"
        if add < threshold:
            ret.append(data[start:end])
            zero_cross_flag = "audio"
        n_cross_dict[idx]={"type":zero_cross_flag, "range":[start, end], "silence_proportion":add}
    return n_cross_dict, ret

def main():
    pass

if __name__ == "__main__":
    audio_file = 'example.wav'
    loader = AudioLoader(audio_file)

    audio_data = loader.get_audio()
    sampling_rate = loader.get_sampling_rate()