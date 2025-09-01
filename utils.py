from scipy import signal
import numpy as np


def filt(fs,
         filt_signal,
         filt_type,
         order,
         f0,
         f1=None):
    '''
    @param:
        filt_type :   ["bandpass","highpass","lowpass"]
        fs        :   sampling rate
        order     :   Order of batterworth filter
        f0,f1     :   Critical frequency or frequencies
                    For bandpass, f0 and f1 need to be specified (f0 < f1)
                    For highpass or lowpass, f1 can be ignored
                    For a Butterworth filter, this is the point at which the gain drops to 
                    1/sqrt(2) that of the passband (the “-3 dB point”)

    @return  
        filted 

    '''
    if filt_type == "bandpass":
        assert (f0 < f1)
        b, a = signal.butter(order, (f0 / (fs / 2), f1 / (fs / 2)), filt_type)
    elif filt_type in ["highpass", "lowpass"]:
        b, a = signal.butter(order, f0 / (fs / 2), filt_type)
    else:
        raise ValueError(
            "filt_type should be in ['bandpass','highpass','lowpass']")
    filted = signal.filtfilt(b, a, filt_signal, axis=0)
    return filted


def roll(data, shift, pad_value=0):
    '''
        Roll the data by shift
        - If shift > 0, pad the first shift elements with pad_value
        - If shift < 0, pad the last shift elements with pad_value
        - If shift == 0, return the original data
        - pad_value is 0 by default
    '''
    result = np.zeros_like(data)
    if shift > 0:
        result[:shift] = pad_value
        result[shift:] = data[:-shift]
    elif shift < 0:
        result[shift:] = pad_value
        result[:shift] = data[-shift:]
    else:
        result = data
    return result


def acoustic_fft(fs=48000,
                 s=None,
                 fftlen=None,
                 axis=-1):
    '''
        @description: package np.fft.fft and np.fft.freq
        @param: 
            - fs        :   sampling rate
            - s         :   signal
            - fftlen    :   fft length
            - axis      :   axis to do fft (default -1)
    '''

    if fftlen is None:
        fftlen = s.shape[axis]
    fft = np.fft.fft(s, fftlen, axis)
    fft_length = (int)(fft.shape[0] / 2)
    fft_abs = np.abs(fft[:fft_length])
    fft_freq = np.fft.fftfreq(fft.shape[0], d=1 / fs)
    fft_freq = fft_freq[:fft_length]
    return fft_abs, fft_freq


def normalize(data, axis=0):
    return data / np.max(data, axis=axis)
