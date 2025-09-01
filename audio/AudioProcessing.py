import sys
if 'streamlit' in sys.modules:
    from stqdm import stqdm
else:
    from tqdm import tqdm as stqdm

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool
from audio.helpers import filt_I_Q, construct_I_Q, mod_I_Q, demod_I_Q
from numba import prange
import numba
from loguru import logger
import numpy as np
from math import floor
from scipy import signal
import utils
import enum

import os
sys.path.append(os.getcwd())


class AudioProcess():
    def __init__(self, data, play_arg) -> None:
        self._data = data
        self._fs = play_arg.sampling_rate

    def perform_filter(self, process_arg):
        if not hasattr(process_arg, "frequency_filter"):
            setattr(process_arg, "frequency_filter", 15e3)
        return utils.filt(fs=self._fs,
                          filt_signal=self._data,
                          filt_type="highpass",
                          order=4,
                          f0=process_arg.frequency_filter)

    def crop_data(self, remove_start_idx, remove_end_idx):
        data_croped = self._data
        if remove_start_idx > 0:
            data_croped = data_croped[remove_start_idx:]
        if remove_end_idx > 0:
            data_croped = data_croped[:-remove_end_idx]
        return data_croped

    def __call__(self, process_arg) -> np.ndarray:
        remove_start_idx = np.argwhere(
            self._data > process_arg.remove_start_threshold)[0][0]
        process_arg.remove_start_idx = max(process_arg.remove_start_time * self._fs, remove_start_idx) \
            if hasattr(process_arg, "remove_start_time") else remove_start_idx
        process_arg.remove_end_idx = process_arg.remove_end_time * self._fs \
            if hasattr(process_arg, "remove_end_time") else -1

        self._data = self.crop_data(
            process_arg.remove_start_idx, process_arg.remove_end_idx)
        if hasattr(process_arg, "perform_filter") and process_arg.perform_filter:
            self._data = self.perform_filter(process_arg)
        return self._data


class AudioDecoder():
    def __init__(self, data, play_arg, process_arg) -> None:
        self._data = data
        self.play_arg = play_arg
        self.process_arg = process_arg

        self._fc = play_arg.fc
        self._fs = play_arg.sampling_rate

    def _orth_demodulator(self):
        self.time_stamp = np.linspace(start=0,
                                      stop=(self.play_arg.samples_per_time +
                                            self.play_arg.delay_num - 1) / self._fs,
                                      num=self.play_arg.samples_per_time + self.play_arg.delay_num)

        self.num_frame = floor(self._data.shape[0] / self.play_arg.N_padding)
        self.data_frame = self._data[:self.num_frame * self.play_arg.N_padding]
        self.data_frame = self.data_frame.reshape(
            self.num_frame, self.play_arg.N_padding)
        self.data_frame = self.data_frame.T
        self.delay_num = self.play_arg.delay_num
        self.samples_per_time = self.play_arg.samples_per_time
        # print(f"self.num_frame: {self.num_frame}")
        # print(f"self.data_frame.shape: {self.data_frame.shape}")
        # print(f"self.data_frame: {self.data_frame}")
        # print(f"self.delay_num: {self.delay_num}")
        # print(f"self.samples_per_time: {self.samples_per_time}")
        # print(f"self.time_stamp: {self.time_stamp.shape}")
        # print(f"N_padding: {self.play_arg.N_padding}")

        return self._demodulate_I_Q()

    def _demodulate_I_Q(self):
        self.data_seq_I_demod = np.zeros(
            (self.samples_per_time * self.num_frame - self.delay_num, ))
        self.data_seq_Q_demod = np.zeros_like(self.data_seq_I_demod)
        # with Pool(processes=4) as pool:
        #     pool.map(self._demod_task, range(self.num_frame - 1))
        # with Pool(processes=4) as pool:
        #     for i in range(self.num_frame - 1):
        #         pool.apply_async(self._demod_task, args=(i, ))
        for i in stqdm(range(self.num_frame - 1)):
            self._demod_task(i)

        assert self.data_seq_I_demod[0] != 0

        self.data_seq_I_demod[-self.delay_num:] = 0
        self.data_seq_Q_demod[:self.delay_num] = 0
        self.data_seq_I_demod = np.expand_dims(self.data_seq_I_demod, axis=1)
        self.data_seq_Q_demod = np.expand_dims(self.data_seq_Q_demod, axis=1)
        return np.concatenate([self.data_seq_I_demod, self.data_seq_Q_demod], axis=1)

    def _demod_task(self, i):
        data_seq_I, data_seq_Q = construct_I_Q(
            self.data_frame[:, i], self.data_frame[:, i + 1], self._fc, self.time_stamp, self.delay_num)

        data_seq_I, data_seq_Q = filt_I_Q(data_seq_I, data_seq_Q, self._fs)

        data_seq_I, data_seq_Q = mod_I_Q(
            data_seq_I, data_seq_Q, self._fc, self.time_stamp)

        demod_I_Q(
            self.data_seq_I_demod, self.data_seq_Q_demod, data_seq_I, data_seq_Q, self.delay_num, self.samples_per_time, i)

    def __call__(self):
        # Check if the attributes exist before using them
        has_demodulation = hasattr(self.process_arg, "demodulation_I_Q")
        has_num_channels = hasattr(self.process_arg, "num_all_channels")

        # Only proceed with demodulation if both attributes exist and meet conditions
        if has_demodulation and has_num_channels and self.process_arg.demodulation_I_Q and self.process_arg.num_all_channels == 2:
            return self._orth_demodulator()
        else:
            # Default processing when not using demodulation
            return self.data


def channelWaveHandler(wave_type, **kwargs):
    if wave_type == "Kasami":
        return KasamiChannelEstimation(**kwargs)
    elif wave_type == "ZC":
        return ZCChannelEstimation(**kwargs)
    else:
        raise NotImplementedError


class ChannelEstimation():
    '''
        @description: Channel Estimation using correlation (Base)
        @param
            datarec     :   received data (cropped)
            dataplay    :   played data (cropped)
            channels    :   number of channels
            fs          :   sampling rate
            CIR         :   channel impulse response
            CFR         :   channel frequency response
    '''

    def __init__(self, play_arg, datarec, dataplay):
        self._datarec = datarec
        self._dataplay = dataplay
        self._channels = None
        self.play_arg = play_arg
        self._fs = play_arg.sampling_rate
        # self._wave_channel_handler = self.channelWaveHandler(
        #     wave_type=self.play_arg.wave, play_arg=self.play_arg, datarec=self._datarec, dataplay=self._dataplay)

        # print(self._CIR.shape)
        # self._CFR = None

    def update(self, CIR=None, CFR=None):
        if CIR is not None:
            self._CIR = CIR
        if CFR is not None:
            self._CFR = CFR

    @property
    def datarec(self):
        return self._datarec

    # TODO: modify channels
    @property
    def channels(self):
        # if datarec has attribute channels
        return self._dataplay.shape[1]

    @property
    def dataplay(self):
        return self._dataplay

    @property
    def CIR(self):
        self._CIR = self._get_CIR()
        return self._CIR

    @property
    def CFR(self):
        self._CFR, self._CFR_freq = self._get_CFR(use_self=True)
        return self._CFR, self._CFR_freq

    def CFR(self, CIR, use_self=False):
        self._CFR, self._CFR_freq = self._get_CFR(CIR, use_self)
        return self._CFR, self._CFR_freq

    @property
    def fs(self):
        return self._fs

    # def __str__(self) -> str:
    #     return ">" * 50 + "<" * 50 + "\n" + \
    #         "Channel Estimation Name " + str(self.__class__.__name__)


class ZCChannelEstimation(ChannelEstimation):
    def __init__(self, play_arg, datarec, dataplay):
        super().__init__(play_arg, datarec, dataplay)

    def _get_CIR(self):
        try:
            cir = signal.correlate(self.datarec, self.dataplay, mode='full')

        except AttributeError as a:
            os.error("AttributeError: " +
                     str(a.__class__.__name__) + " " + str(a))
        except ValueError as v:
            os.error("ValueError: " + str(v.__class__.__name__) + " " + str(v))
        finally:
            frames = self.datarec.shape[0] // self.dataplay.shape[0]
            cir = cir[-self.datarec.shape[0]:]
            cir = cir[:self.dataplay.shape[0] *
                      frames]
            cir = cir.reshape(frames, self.dataplay.shape[0])
            cir = np.transpose(cir)
            cir = np.expand_dims(cir, axis=len(cir.shape))
            return cir

    def _get_CFR(self, CIR=None, use_self=False):
        try:
            if use_self:
                CFR, CFR_freq = utils.acoustic_fft(self.fs, self.CIR, axis=0)
            else:
                assert CIR is not None
                CFR, CFR_freq = utils.acoustic_fft(self.fs, CIR, axis=0)
            return CFR, CFR_freq
        except Exception as e:
            print("Exception: " + str(e.__class__.__name__) + " " + str(e))
            return None


class KasamiChannelEstimation(ChannelEstimation):
    '''
        @description: Channel Estimation using correlation (Kasami Sequence)
        @param
            datarec     :   received data (cropped)
            dataplay    :   played data (cropped)
        @_get_CIR
            *  Correlate datarec and dataplay
            @return CIR
        @_get_CFR
            *  Use FFT to get CFR
            @return CFR

    '''

    def __init__(self, play_arg, datarec, dataplay):
        super().__init__(play_arg, datarec, dataplay)
        # self._CIR = self._get_CIR()
        # self._CFR = self._get_CFR()
        # self.play_arg = play_arg
        # self.datarec = datarec
        # self.dataplay = dataplay

    def _get_CIR(self):
        try:
            cir = signal.correlate(self.datarec, self.dataplay, mode='full')

        except AttributeError as a:
            os.error("AttributeError: " +
                     str(a.__class__.__name__) + " " + str(a))
        except ValueError as v:
            os.error("ValueError: " + str(v.__class__.__name__) + " " + str(v))
        finally:
            frames = self.datarec.shape[0] // self.dataplay.shape[0]
            cir = cir[-self.datarec.shape[0]:]
            cir = cir[:self.dataplay.shape[0] *
                      frames]
            cir = cir.reshape(frames, self.dataplay.shape[0])
            cir = np.transpose(cir)
            cir = np.expand_dims(cir, axis=len(cir.shape))
            return cir

    def _get_CFR(self, CIR=None, use_self=False):
        try:
            if use_self:
                CFR, CFR_freq = utils.acoustic_fft(self.fs, self.CIR, axis=0)
            else:
                assert CIR is not None
                CFR, CFR_freq = utils.acoustic_fft(self.fs, CIR, axis=0)
            return CFR, CFR_freq
        except Exception as e:
            print("Exception: " + str(e.__class__.__name__) + " " + str(e))
            return None

    def merge_CFR(self, CFR=None, use_self=False):
        def _merge(CFR_subc):
            try:
                num_channels = CFR_subc.shape[1]
            except IndexError:
                raise IndexError("CFR.shape: " + str(CFR_subc.shape))
            for i in range(CFR_subc.shape[0]):
                for j in range(num_channels):
                    yield CFR_subc[i, j]

        def _cat(CFR):
            assert len(CFR.shape) == 3
            CFR_merged = np.zeros(
                (CFR.shape[0], CFR.shape[1] * 2, 1))
            for subc in stqdm(range(CFR.shape[0])):
                CFR_merged[subc, :] = np.array(
                    list(_merge(CFR[subc, ...]))).reshape(-1, 1)
            return CFR_merged

        if use_self:
            return _cat(self.CFR)
        else:
            assert CFR is not None
            return _cat(CFR)

    def __str__(self) -> str:
        return ">" * 50 + "Channel Estimation" + "<" * 50 + "\n" + \
            "Channel Estimation Name " + str(self.__class__.__name__) + "\n" + \
            "dataplay.shape: " + str(self.dataplay.shape) + "\n" + \
            "datarec.shape: " + str(self.datarec.shape)


class chirpChannelEstimation(ChannelEstimation):
    '''
        @description: Channel Estimation using correlation (Chirp Sequence)
        @param
            datarec     :   received data (cropped)
            dataplay    :   played data (cropped)
        @_get_CIR
            *  Datarec and dataplay should be cropped to the same length
            *  fft datarec
            @return CIR
        @_get_CFR
            *  Use FFT to get CFR
            @return CFR

    '''

    def __init__(self, sampling_rate, datarec, dataplay):
        super().__init__(sampling_rate, datarec, dataplay)
        # self._CFR = self._get_CFR()
        self._CIR = self._get_CIR()
        # self._CFR = self._get_CFR()

    def _get_CIR(self):
        try:
            self._mixed = self._mix_signal()
            # logger.info("mixed signal shape: " + str(self._mixed.shape))
            # todo: check whether fft axis = 0 (default -1)
            CIR, _ = utils.acoustic_fft(self.fs, self._mixed)
            # logger.info("CIR shape: " + str(CIR.shape))
            return CIR
        except AttributeError as a:
            logger.error("AttributeError: " +
                         str(a.__class__.__name__) + " " + str(a))
        except ValueError as v:
            logger.error("ValueError: " + str(v.__class__.__name__))

    def _get_CFR(self, CIR=None, use_self=False):
        try:
            if use_self:
                CFR, _ = utils.acoustic_fft(self.fs, self.CIR, axis=0)
            else:
                CFR, _ = utils.acoustic_fft(self.fs, CIR, axis=0)
            return CFR
        except Exception as e:

            print("Exception: " + str(e.__class__.__name__) + " " + str(e))
            return None

    def _mix_signal(self):
        # if self.dataplay.shape[0] != self.datarec.shape[0]:
        #     lag = preProcess.get_sync_lag(self.dataplay, self.datarec)
        #     logger.debug("lag: " + str(lag))
        #     self._dataplay, self._datarec = preProcess.align_signal(
        #         self._dataplay, self._datarec, lag)
        #     # logger.info("dataplay.shape: " + str(self.dataplay.shape))
        #     # logger.info("datarec.shape: " + str(self.datarec.shape))
        return self.datarec * self.dataplay


# @archived
# class preProcess():
#     def __init__(self,
#                  datarec,
#                  dataplay,
#                  sampling_rate,
#                  start_freq=None,
#                  end_freq=None
#                  ):
#         self.datarec = datarec
#         self.dataplay = dataplay
#         self._fs = sampling_rate
#         self._start_freq = start_freq
#         self._end_freq = end_freq
#         self._mixed = None

#     @property
#     def bandwidth(self):
#         return self.end_freq - self.start_freq

#     @property
#     def center_freq(self):
#         return (self.start_freq + self.end_freq) / 2

#     @property
#     def fs(self):
#         return self._fs

#     @property
#     def start_freq(self):
#         return self._start_freq

#     @property
#     def end_freq(self):
#         return self._end_freq

#     @property
#     def mixed(self):
#         return self._mixed

#     @mixed.setter
#     def mixed(self, mixed):
#         self._mixed = mixed

#     @staticmethod
#     def get_sync_lag(src, dst):
#         corr = signal.correlate(src, dst, mode='same', method='fft')
#         # get corerelation lag
#         lag = corr.argmax() - (corr.shape[0] - 1) / 2
#         if src.shape[0] > dst.shape[0]:
#             lag = -lag
#         return (int)(lag)

#     @staticmethod
#     def get_sync_lag_no_corr(src, dst, thres=0.0015):
#         return np.argwhere(dst > thres)[0][0]

#     def multiply(self):
#         start = self._get_sync_lag(self.datarec, self.dataplay)
#         print("start lag: ", start)
#         self._align(start)
#         self.mixed = self.datarec * self.dataplay
#         peaks = signal.find_peaks(
#             self._fft()[0], height=1000)
#         print("peaks: ", peaks)
#         return self.mixed

#     def _align(self, lag):
#         self.align_signal(self.dataplay, self.datarec, lag)

#     @staticmethod
#     def align_signal(dataplay, datarec, lag):
#         len = dataplay.shape[0]
#         dataplay = dataplay[lag:len]
#         datarec = datarec[lag:len]
#         return dataplay, datarec

#     def _fft(self):
#         return utils.acoustic_fft(self.fs, self.mixed)

#     def filt(self, filt_signal, filt_type, order, f0, f1=None):
#         return utils.filt(self.fs, filt_signal, filt_type, order, f0, f1)

#     def fft(self, signal):
#         return utils.acoustic_fft(self.fs, signal)

#     def plot_fft(self, fft_abs, fft_freq):
#         plt.plot(fft_freq, fft_abs)
#         plt.show()

#     def _plot_fft(self):
#         fft_abs, fft_freq = self._fft()
#         return self.plot_fft(fft_abs, fft_freq)

#     def coherent_detector(self, if_signal, center_freq, time):
#         in_phase = if_signal * np.exp(-1j * 2.0 * np.pi * center_freq * time)
#         quadrature = if_signal * \
#             np.exp(-1j * 2.0 * np.pi * center_freq * time + 0.5 * np.pi)

#         in_phase_fft, in_phase_fft_freq = self.fft(in_phase)
#         quadrature_fft, quadrature_fft_freq = self.fft(quadrature)

#         # TODO: specify the range of frequency
#         in_phase_fft = utils.filt(
#             self.fs, in_phase_fft, "bandpass", 4, self.start_freq, self.end_freq)
#         quadrature_fft = utils.filt(
#             self.fs, quadrature_fft, "bandpass", 4, self.start_freq, self.end_freq)

#         in_phase_fft = self.filt(in_phase_fft, "lowpass", 4, 1000)
#         quadrature_fft = self.filt(quadrature_fft, "lowpass", 4, 1000)

#         in_phase = np.fft.ifft(in_phase_fft)
#         quadrature = np.fft.ifft(quadrature_fft)
#         return in_phase, quadrature

#         # self._filt(in_phase_fft, "lowpass", 4, 2000)

#         # self.plot_fft(in_phase_fft, in_phase_fft_freq)
#         # self.plot_fft(quadrature_fft, quadrature_fft_freq)

#     def vis_IQ(self, in_phase, quadrature):
#         plt.plot(in_phase, quadrature)
#         plt.show()
