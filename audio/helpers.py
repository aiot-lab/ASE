import utils
import numpy as np
from numba import jit
from loguru import logger


def filt_I_Q(data_seq_I=None, data_seq_Q=None, fs=48000):
    if data_seq_I is not None:
        data_seq_I = utils.filt(fs=fs,
                                filt_signal=data_seq_I,
                                filt_type="lowpass",
                                order=20,
                                f0=4e3)
    if data_seq_Q is not None:
        data_seq_Q = utils.filt(fs=fs,
                                filt_signal=data_seq_Q,
                                filt_type="lowpass",
                                order=20,
                                f0=4e3)
    return data_seq_I, data_seq_Q


def construct_I_Q(data_frame_i, data_frame_j, fc, time_stamp, delay_num):
    data_seq_I = np.concatenate((data_frame_i, np.zeros(
        delay_num)), axis=0) * np.cos(2 * np.pi * fc * time_stamp)
    data_seq_Q = -np.concatenate((np.zeros(delay_num),
                                  data_frame_i[delay_num:],
                                  data_frame_j[:delay_num]), axis=0) * np.sin(2 * np.pi * fc * time_stamp)
    return data_seq_I, data_seq_Q

def mod_I_Q(data_seq_I, data_seq_Q, fc, time_stamp):
    data_seq_I = data_seq_I * \
        np.cos(2 * np.pi * fc * time_stamp)
    data_seq_Q = data_seq_Q * \
        np.sin(2 * np.pi * fc * time_stamp)
    return data_seq_I, data_seq_Q


def demod_I_Q(data_seq_I_demod, data_seq_Q_demod, data_seq_I, data_seq_Q, delay_num, samples_per_time, i):
    data_seq_I_demod[i * samples_per_time:
                     i * samples_per_time + delay_num] = data_seq_I[:delay_num]
    data_seq_I_demod[i * samples_per_time + delay_num:(
        i + 1) * samples_per_time + delay_num] = data_seq_I[delay_num:]

    data_seq_Q_demod[i * samples_per_time + delay_num:(
        i + 1) * samples_per_time + delay_num] = data_seq_Q[delay_num:]

    return data_seq_I_demod, data_seq_Q_demod


def init_channel():
    cir = np.array([])
    cfr = np.array([])
    return cir, cfr


def normalize_data(datarec=None, dataseq=None):
    try:
        from utils import normalize
        datarec = normalize(datarec)
        dataseq = normalize(dataseq)
        return datarec, dataseq
    except ImportError:
        logger.error("ImportError: cannot import normalize function")


def process_cir(cir_i, cir):
    # concatenate channels
    assert cir_i is not None
    cir = cir_i if cir.shape[0] == 0 else np.concatenate(
        (cir, cir_i), axis=2)
    return cir


def process_cfr(cfr_i, cfr):
    assert cfr_i is not None
    # take the magnitude of the cfr
    cfr_i = np.square(cfr_i)
    # remove mean
    cfr_i = cfr_i - np.mean(cfr_i, axis=1, keepdims=True)
    # concatenate channels
    cfr = cfr_i if cfr.shape[0] == 0 else np.concatenate(
        (cfr, cfr_i), axis=2)
    return cfr, cfr_i


def get_SPL_dB(data_PCM):
    avg_PCM = np.sqrt(np.mean(np.square(data_PCM.flatten())))
    SPL_dB = 10 * np.log10(avg_PCM*40 / 2e-5)
    return round(SPL_dB, 1)


def filter_CFR(cfr, cfr_freq, play_arg):
    assert cfr_freq is not None

    filter_freq_low_idx = np.argwhere(
        cfr_freq >= play_arg.fc - play_arg.bandwidth / 2)[0][0]
    filter_freq_high_idx = np.argwhere(
        cfr_freq > play_arg.fc + play_arg.bandwidth / 2)[0][0] + 1

    assert filter_freq_low_idx < filter_freq_high_idx

    cfr_freq = cfr_freq[filter_freq_low_idx:filter_freq_high_idx]
    cfr_new = cfr[filter_freq_low_idx:filter_freq_high_idx, ...]

    assert cfr_new.shape[0] == cfr_freq.shape[0]

    return cfr_new, cfr_freq


def normalize_cfr(cfr):
    cfr = np.square(cfr)
    cfr = cfr / np.sum(cfr, axis=0, keepdims=True)
    return cfr


def get_acoustic_device():
    import sounddevice as sd
    devices = sd.query_devices()
    devices_input_list = [devices[i]['name'] for i in range(
        len(devices)) if devices[i]['max_input_channels'] > 0]
    devices_output_list = [devices[i]['name'] for i in range(
        len(devices)) if devices[i]['max_output_channels'] > 0]
    return devices_input_list, devices_output_list
