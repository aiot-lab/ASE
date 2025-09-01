import sys
import os
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# print("\n".join(sys.path))

from audio import Wave


from scipy import signal
import numpy as np
import utils
import scipy.io as scio
from tqdm import tqdm
from loguru import logger
if "streamlit" in sys.modules:
    from stqdm import stqdm
else:
    from tqdm import tqdm as stqdm


class ACF():
    '''
    @description            :   ACF of CSI
    @param _CFR             :   3-D(2-D) array, shape: (subcarriers, frames (t), <num_channels>)
    @param _windows         :   args.windows
    @param _windows_step    :   args.windows_step
    @param _windows_width   :   args.windows_width
    @param _topK            :   args.topK select the topK CFRs
    @param num_channels     :   args.num_channels 
    @param frames           :   args.frames
    @method ACF             :   ACF of CSI
    @method _get_topK       :   select the topK CFRs
    '''

    def __init__(self,
                 CFR=None,
                 process_arg=None,
                 ) -> None:

        self._CFR = CFR

        try:
            if hasattr(process_arg, "num_topK_subcarriers") and process_arg.num_topK_subcarriers > 0:
                self._topK = min(
                    process_arg.num_topK_subcarriers, self._CFR.shape[0])
            else:
                self._topK = self._CFR.shape[0]
        finally:
            assert self._topK <= self._CFR.shape[0], "topK should be less than the number of subcarriers"
        try:
            self.num_channels = self._CFR.shape[2]
            self.frames = self._CFR.shape[1]
        except IndexError:
            raise IndexError("CFR.shape: ", self._CFR.shape)

        self._windows_step = round(
            process_arg.windows_step / self.num_channels)
        self._windows_width = process_arg.windows_width // self.num_channels

    @property
    def CFR(self):
        return self._CFR

    @property
    def topK(self):
        return self._topK

    @property
    def windows_width(self):
        return self._windows_width

    @property
    def windows_step(self):
        return self._windows_step

    @property
    def acf(self):
        return self._get_ACF()

    @property
    def ms(self):
        return self.motion_statistics()

    def _get_ACF(self):
        acf_matrix = np.array([])
        for channel_idx in range(self.num_channels):
            acf_matrix_t = np.array([])
            for t in stqdm(range(0, self.frames - self.windows_width + 1, self.windows_step), desc="ACF"):
                acf_matrix_f = np.array([])
                if self._topK != self._CFR.shape[0]:
                    self._CFR = self._get_topK(t, channel_idx)
                # else:
                #     self._topK = self._CFR.shape[0]
                for f in range(self._topK):
                    CFR_f = self._CFR[f, t:t + self.windows_width, channel_idx]
                    CFR_f_normal = (CFR_f - np.mean(CFR_f)) / \
                        np.linalg.norm(CFR_f - np.mean(CFR_f))
                    # autocorrelation
                    acf = signal.correlate(
                        CFR_f_normal, CFR_f_normal, mode="full")
                    acf = acf[acf.size // 2:].reshape(-1, 1)
                    # print("acf.shape: ", acf.shape)
                    acf_matrix_f = acf if acf_matrix_f.shape[0] == 0 else np.concatenate(
                        (acf_matrix_f, acf), axis=1)

                acf_matrix_t = np.expand_dims(acf_matrix_f, axis=1) \
                    if acf_matrix_t.shape[0] == 0 \
                    else np.append(
                        acf_matrix_t, np.expand_dims(acf_matrix_f, axis=1), axis=1)
                # print("acf_matrix_t.shape: ", acf_matrix_t.shape)
            # acf_matrix_t = np.transpose(acf_matrix_t, (0, 2, 1))
            acf_matrix = np.expand_dims(acf_matrix_t, axis=3) \
                if acf_matrix.shape[0] == 0 \
                else np.append(acf_matrix, np.expand_dims(acf_matrix_t, axis=3), axis=3)

        # acf_matrix = np.transpose(acf_matrix, (1, 2, 0, 3))
        return np.transpose(acf_matrix, [2, 0, 1, 3])

    def _get_topK(self, t_i, channel_idx):
        # take the window of CFR [:,t_i:t_i+windows_width,1]
        CFR_ti = self.CFR[:, t_i:t_i + self._windows_width, channel_idx]
        MS_ci = self._get_motion_statistics(t_i, channel_idx)
        sorted = np.argsort(MS_ci)[::-1]  # descending order
        CFR_topk = CFR_ti[sorted[-self.topK:]]
        return CFR_topk
        # print("CFR_topk.shape: ", CFR_topk.shape)

    def motion_statistics(self):
        ms = np.array([])
        for channel_idx in range(self.num_channels):
            ms_t = np.array([])
            # todo: modify channel logic
            if self.num_channels == 1:
                cfr_channel = self.CFR
            else:
                cfr_channel = self.CFR[:, :, channel_idx]
            # logger.debug("cfr_channel.shape: {}".format(cfr_channel.shape))
            # logger.debug("self.frames: {}".format(self.frames))
            for t in tqdm(range(0, self.frames - self.windows_width + 1, self.windows_step), leave=False):
                # logger.debug("t: {}".format(t))
                ms_ti = self._get_motion_statistics(t, cfr_channel)
                ms_t = np.expand_dims(ms_ti, axis=1) \
                    if ms_t.shape[0] == 0 \
                    else np.append(ms_t, np.expand_dims(ms_ti, axis=1), axis=1)
            ms_t = np.transpose(ms_t, (1, 0))

            ms = np.expand_dims(ms_t, axis=len(ms_t.shape)) \
                if ms.shape[0] == 0 \
                else np.append(ms, np.expand_dims(ms_t, axis=len(ms_t.shape)), axis=len(ms_t.shape))

        ms = np.transpose(ms, (1, 0, 2))
        return ms  # [subcarrier, time, channel]

    def _get_motion_statistics(self, t_i, cfr_ti):
        '''
            @description: get motion statistics of a window
            @param t_i: time index
            @param cfr_ti: CFR of time t_i (subcarrier, window_width)
            @note compute the first acf
        '''
        CFR_ti = np.squeeze(cfr_ti[:, t_i:t_i + self._windows_width])
        CFR_mean_diff = CFR_ti - \
            np.mean(CFR_ti, axis=1, keepdims=True).reshape(-1, 1)
        ms_ti = np.sum(CFR_mean_diff[:, :-1] * CFR_mean_diff[:, 1:],
                       axis=1) / np.sum(CFR_mean_diff**2, axis=1)
        return ms_ti

    def _get_speed(self):
        pass

    @staticmethod
    def get_motion_curve(ms):
        ms_mean = np.mean(ms, axis=1)
        return ms_mean

    def __str__(self):
        return "Model:" + self.__class__.__name__


def MRC(acf_matrix, weight_matrix, ):
    assert acf_matrix.shape[3] == 1
    weight_matrix = np.squeeze(weight_matrix)
    assert len(weight_matrix.shape) == 2
    acf_mrc = np.zeros(
        (acf_matrix.shape[1], acf_matrix.shape[2]))
    for t_idx in range(acf_matrix.shape[2]):
        acf_ti = np.squeeze(acf_matrix[:, :, t_idx, 0])
        weight_ti = weight_matrix[:, t_idx]
        try:
            np.seterr(divide='ignore', invalid='ignore')
            weight_ti = np.nan_to_num(
                weight_ti / np.sum(weight_ti, keepdims=True))
        except RuntimeWarning:
            pass
        acf_mrc_ti = np.sum(acf_ti * weight_ti.reshape(-1, 1), axis=0)
        acf_mrc[:, t_idx] = acf_mrc_ti
    return acf_mrc
