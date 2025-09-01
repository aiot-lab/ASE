import numpy as np
from numba import njit, prange
from stqdm import stqdm
from loguru import logger
import pandas as pd

from itertools import groupby, chain
from operator import itemgetter
from scipy.signal import find_peaks, peak_prominences
from scipy.special import j0
from scipy.optimize import curve_fit

F_REF = 12e3
V_SOUND = 343
MAX_SPEED = 2.5
MIN_SPEED = 0.05


def init_zc(acf_matrix):
    assert acf_matrix.ndim == 4
    zc_o = np.zeros((acf_matrix.shape[0],
                     acf_matrix.shape[2],
                     acf_matrix.shape[3]))
    zc_d = np.zeros_like(zc_o)
    return zc_o, zc_d


def zero_crossing(array):
    assert array.shape[0] > 1
    return (array[:-1] * array[1:] < 0).sum()


def diff(array):
    return array[1:] - array[:-1]


def interp_tau(max_tau=0.5):
    return np.linspace(0, max_tau, int(max_tau * 1000))


def interp_acf(acf, tau):
    # max_tau = 1
    tau_q = interp_tau()
    acf = np.interp(tau_q, tau[tau <= 0.5], acf[tau <= 0.5])
    return acf


def alignment(rho_f_t_c, tau, freq_i, freq_ref=F_REF):
    N = len(rho_f_t_c)
    coeff = freq_i / freq_ref
    tau_q = tau * coeff
    rho_f_t_c_interp = np.interp(tau_q, tau, rho_f_t_c)
    rho_f_t_c_interp = rho_f_t_c_interp[:N]
    rho_f_t_c_interp = interp_acf(rho_f_t_c_interp, tau_q)
    return rho_f_t_c_interp


def remove_and_align(acf_matrix, acf_ms, freq, tau):
    tau_q = interp_tau()
    zc_o = np.zeros((acf_matrix.shape[0],
                     acf_matrix.shape[2],
                     acf_matrix.shape[3]))
    zc_d = np.zeros_like(zc_o)
    zc_o_aligned = np.zeros_like(zc_o)
    zc_d_aligned = np.zeros_like(zc_o)
    acf_matrix_o = np.zeros(
        (acf_matrix.shape[0], tau_q.shape[0], acf_matrix.shape[2], acf_matrix.shape[3]))
    acf_matrix_wo_alignment = acf_matrix.copy()
    acf_ms_o = acf_ms.copy()
    # logger.debug(f"zc_o.shape: {zc_o.shape}")
    # todo: remove outliers
    for cdx in range(acf_matrix.shape[3]):
        for tdx in range(acf_matrix.shape[2]):
            for fdx in range(acf_matrix.shape[0]):
                rho_f_t_c = acf_matrix[fdx, :, tdx, cdx]
                rho_f_t_c_diff = diff(rho_f_t_c)
                zc_o[fdx, tdx, cdx] = zero_crossing(rho_f_t_c)
                zc_d[fdx, tdx, cdx] = zero_crossing(rho_f_t_c_diff)

                if zc_d[fdx, tdx, cdx] > 135 or zc_o[fdx, tdx, cdx] < 10:
                    acf_matrix_o[fdx, :, tdx, cdx] = np.zeros(
                        (tau_q.shape[0],))
                    acf_matrix_wo_alignment[fdx, :, tdx,
                                            cdx] = np.zeros_like(rho_f_t_c)
                    acf_ms_o[fdx, tdx, cdx] = 0
                elif max(rho_f_t_c_diff) - min(rho_f_t_c_diff) < 1e-2:
                    acf_matrix_o[fdx, :, tdx, cdx] = np.zeros(
                        (tau_q.shape[0],))
                    acf_matrix_wo_alignment[fdx, :, tdx,
                                            cdx] = np.zeros_like(rho_f_t_c)
                    acf_ms_o[fdx, tdx, cdx] = 0
                else:
                    acf_matrix_o[fdx, :, tdx, cdx] = alignment(acf_matrix[fdx, :, tdx, cdx],
                                                               tau=tau,
                                                               freq_i=freq[fdx])

                zc_o_aligned[fdx, tdx, cdx] = zero_crossing(
                    acf_matrix_o[fdx, :, tdx, cdx])
                zc_d_aligned[fdx, tdx, cdx] = zero_crossing(
                    diff(acf_matrix_o[fdx, :, tdx, cdx]))
            if np.sum(acf_matrix_o[:, 0, tdx, cdx]) <= 2:
                acf_matrix_o[:, :, tdx, cdx] = np.zeros_like(
                    acf_matrix_o[:, :, tdx, cdx])
                acf_ms_o[:, tdx, cdx] = np.zeros_like(acf_ms_o[:, tdx, cdx])

    return acf_matrix_o, acf_ms_o, zc_o, zc_d, zc_o_aligned, zc_d_aligned, acf_matrix_wo_alignment


def get_threshold(ms):
    from skimage.filters import threshold_otsu
    ms_threshold = threshold_otsu(ms, nbins=10)
    return ms_threshold


def group_time_idx(motion_list):
    groups = []
    assert isinstance(motion_list, list)

    for _, g in groupby(enumerate(motion_list), lambda x: x[0] - x[1]):
        groups.append(list(map(itemgetter(1), g)))
    return groups


def ravel_list(t_list):
    return list(chain.from_iterable(t_list))


def merge_time_motion(merged_time, dual_time, motion_list_group):
    t_list_merged = []
    idx_list_merged = []
    for group in motion_list_group:
        if len(group) == 1:
            continue
        begin = np.where(merged_time >= dual_time[group[0]])[0][0]
        end = np.where(merged_time <= dual_time[group[-1]])[0][-1]
        assert end >= begin
        t_list_merged.append(list(merged_time[begin: end]))
        idx_list_merged.append(list(range(begin, end)))
    t_merged = np.array(ravel_list(t_list_merged))
    return t_merged, ravel_list(idx_list_merged)


def check_motion(motion_time, merged_idx=None, merged_t=None, ):
    return np.isin(merged_t[merged_idx], motion_time)


def find_ref(func):
    x = np.linspace(0, 10, 5000)
    y = func(x)
    pkx, _ = find_peaks(y)
    return x[pkx[0]]


def get_x0():
    return find_ref(j0)


def get_speed(x0, tau_s, f_ref=F_REF):
    return (x0 * V_SOUND) / (2 * np.pi * f_ref * tau_s)


def tau_upper_bound(v_min=MIN_SPEED, f_ref=F_REF):
    x0 = get_x0()
    return (x0 * V_SOUND) / (2 * np.pi * f_ref * v_min)


def tau_lower_bound(v_max=MAX_SPEED, f_ref=F_REF):
    x0 = get_x0()
    return (x0 * V_SOUND) / (2 * np.pi * f_ref * v_max)


def decay_weight(origin_peak, origin_weight, fit_function=None):
    assert (len(origin_peak) == origin_weight.shape[1])
    if fit_function is None or fit_function == "exp":
        fit_function = exp_decay
    elif fit_function == "sigmoid":
        fit_function = sigmoid
    else:
        raise ValueError("fit_function must be exp or sigmoid")
    new_weight = origin_weight.copy()
    for t_idx, o_peak in enumerate(origin_peak):
        if o_peak.size > 0:
            o_peak_median, o_peak_mean = np.median(o_peak), np.mean(o_peak)
            o_peak_min, o_peak_max = np.min(o_peak), np.max(o_peak)
            q1, q3 = np.quantile(o_peak, 0.25), np.quantile(o_peak, 0.75)
            iqr = q3 - q1
            q1 = max(q1 - 1.5 * iqr, o_peak_min)
            q3 = min(q3 + 1.5 * iqr, o_peak_max)
            u1, u2 = min(o_peak_median, o_peak_mean), max(
                o_peak_median, o_peak_mean)
            try:
                if fit_function == sigmoid:
                    p0_1 = [1, (o_peak_min + u1) / 2]
                    p0_2 = [1, (u2 + q3) / 2]
                    coeff1, _ = curve_fit(fit_function, np.array(
                        [q1, u1]), np.array([0, 1]), maxfev=3000, p0=p0_1)
                    coeff2, _ = curve_fit(fit_function, np.array(
                        [u2, q3]), np.array([1, 0]), maxfev=3000, p0=p0_2)
                elif fit_function == exp_decay:
                    b_1, a_log_1 = np.polyfit(
                        np.array([q1, u1]), np.log(np.array([0.1, 1])), 1)
                    b_2, a_log_2 = np.polyfit(
                        np.array([u2, q3]), np.log(np.array([1, 0.1])), 1)
                    coeff1 = [np.exp(a_log_1), b_1]
                    coeff2 = [np.exp(a_log_2), b_2]
            except RuntimeError as e:
                pass
            a1, b1 = coeff1[0], coeff1[1]
            a3, b3 = coeff2[0], coeff2[1]
            # if fit_function == sigmoid:
            #     #     print(a1, b1)
            #     if t_idx == 245:
            #         print(a3, b3)
            #         print(u2, q3)

            def f1(x):
                return fit_function(x, a1, b1)

            def f2(x):
                return 1

            def f3(x):
                return fit_function(x, a3, b3)

            w = []
            for peak in o_peak:
                if peak < u1:
                    w.append(f1(peak))
                elif peak < u2:
                    w.append(f2(peak))
                elif peak >= u2:
                    w.append(f3(peak))
            modified_weight = []
            j = 0
            for f_idx in range(origin_weight.shape[0]):
                if origin_weight[f_idx, t_idx] != 0:
                    modified_weight.append(
                        origin_weight[f_idx, t_idx] * w[j])
                    j += 1
                else:
                    modified_weight.append(0)
            modified_weight = np.array(modified_weight)
            new_weight[:, t_idx] = modified_weight / np.max(
                modified_weight, keepdims=True)
    return new_weight


def sigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a * (x - b)))


def exp_decay(x, a, b):
    return a * np.exp(-b * x)


def get_peak_prominence(acf_merged_matrix,
                        time_stamp,
                        tau,
                        prominence_threshold=1e-2,
                        ):
    tau_peak = np.zeros(
        (acf_merged_matrix.shape[0], acf_merged_matrix.shape[2]))
    tau_peak_prominence = np.zeros(
        (acf_merged_matrix.shape[0], acf_merged_matrix.shape[2]))
    origin_peak = []
    # kmeans_center = np.zeros((acf_merged_matrix.shape[2], 2))
    for t_idx in range(acf_merged_matrix.shape[2]):
        acf_ti = np.squeeze(acf_merged_matrix[:, :, t_idx, 0])
        if check_motion(motion_time=time_stamp["motion"]["merged"],
                        merged_idx=t_idx,
                        merged_t=time_stamp["merged"]):
            for f_idx in range(acf_merged_matrix.shape[0]):
                acf_ti_fi = acf_ti[f_idx, :]
                peak_position, _ = find_peaks(acf_ti_fi,
                                              prominence=3e-2,)

                if peak_position.size > 0:
                    tau_p = tau["merged"][peak_position[0]]
                    if tau_p >= tau_upper_bound() or tau_p <= tau_lower_bound():
                        acf_merged_matrix[f_idx, :, t_idx, 0] = np.zeros(
                            (acf_merged_matrix.shape[1],))
                        continue
                    peak_prominence = peak_prominences(
                        acf_ti_fi, peak_position)[0]
                    tau_peak[f_idx, t_idx] = tau_p
                    tau_peak_prominence[f_idx, t_idx] = peak_prominence[0]
            if np.sum(tau_peak_prominence[:, t_idx]) != 0:
                tau_peak_prominence[:, t_idx] = tau_peak_prominence[:, t_idx] / \
                    np.max(tau_peak_prominence[:, t_idx], keepdims=True)
            # weight_peak_data = np.multiply(
            #     tau_peak_prominence[:, t_idx], tau_peak[:, t_idx])
            # weight_peak.append(weight_peak_data[weight_peak_data != 0])
            origin_peak.append(tau_peak[tau_peak[:, t_idx] != 0, t_idx])
            # peak_data = tau_peak[tau_peak[:, t_idx] != 0, t_idx]
            # if peak_data.size >= 2:
            #     kmeans = KMeans(n_clusters=2, n_init="auto").fit(
            #         peak_data.reshape(-1, 1), sample_weight=tau_peak_prominence[tau_peak[:, t_idx] != 0, t_idx])
            #     kmeans_center[t_idx, :] = np.squeeze(
            #         kmeans.cluster_centers_)
        else:
            origin_peak.append(np.array([]))
    tau_peak_prominence[np.isnan(tau_peak_prominence)] = 0
    return tau_peak, tau_peak_prominence, origin_peak


def get_speed(acf_mrc,
              time_stamp,
              tau):
    tau_s = np.zeros((acf_mrc.shape[1],))
    vel = np.zeros((acf_mrc.shape[1],))
    x0 = get_x0()
    for t_idx in range(acf_mrc.shape[1]):
        if check_motion(motion_time=time_stamp["motion"]["merged"],
                        merged_idx=t_idx,
                        merged_t=time_stamp["merged"]):
            acf_ti = acf_mrc[:, t_idx]

            peak_position, _ = find_peaks(acf_ti,
                                          prominence=3e-2,)
            if peak_position.size > 0:
                tau_s[t_idx] = tau["merged"][peak_position[0]]
            vel[t_idx] = (x0 * V_SOUND) / (2 * np.pi * F_REF * tau_s[t_idx])
            if tau_s[t_idx] < tau_lower_bound():
                for i in range(1, min(3, peak_position.size)):
                    tau_i = tau["merged"][peak_position[i]]
                    if tau_i > tau_lower_bound():
                        vel[t_idx] = (x0 * V_SOUND) / \
                            (2 * np.pi * F_REF * tau_i)
                        tau_s[t_idx] = tau_i
                        break
                    vel[t_idx] = np.nan
        else:
            tau_s[t_idx] = np.nan
    vel[np.isinf(vel)] = np.nan
    nan_idx = np.argwhere(np.isnan(vel)).ravel().tolist()
    group_nan_idx = group_time_idx(nan_idx)
    # if group_nan_idx[0][0] == time_stamp["motion"]["idx"][0]:
    #     start_idx = group_nan_idx[0][0] - 1
    #     end_idx = group_nan_idx[0][-1] + 1
    #     vel = interp_nan(start_idx, end_idx, vel)
    # if group_nan_idx[-1][-1] == time_stamp["motion"]["idx"][-1]:
    #     start_idx = group_nan_idx[-1][0] - 1
    #     end_idx = group_nan_idx[-1][-1] + 1
    #     vel = interp_nan(start_idx, end_idx, vel)
    for g_idx in range(len(group_nan_idx)):
        if len(group_nan_idx[g_idx]) <= 3 or group_nan_idx[g_idx][0] == time_stamp["motion"]["idx"][0] or group_nan_idx[g_idx][-1] == time_stamp["motion"]["idx"][-1]:
            start_idx = group_nan_idx[g_idx][0] - 1
            end_idx = group_nan_idx[g_idx][-1] + 1
            vel = interp_nan(start_idx, end_idx, vel)
    return vel, tau_s


def interp_nan(start_idx, end_idx, vel):
    vel[start_idx:end_idx + 1] = pd.DataFrame(
        vel[start_idx:end_idx + 1]).interpolate().values.ravel()
    return vel


def compute_detection_rate(velocity_mat):
    number_of_nan = sum(np.isnan(velocity_mat))
    number_of_inf = sum(np.isinf(velocity_mat))
    detection_rate = (len(velocity_mat) - number_of_nan -
                      number_of_inf) / len(velocity_mat)
    return detection_rate


def compute_mean_error(vel, vel_gt):
    assert len(vel) == len(vel_gt)
    v_error = np.abs(np.subtract(vel, vel_gt))
    v_error = v_error[~np.isnan(v_error)]
    v_mean_error = np.sum(v_error) / len(v_error)
    return v_mean_error, v_error


def compute_hist(vel_error):
    counts, bins = np.histogram(np.abs(vel_error), bins=100)
    pdf = counts / sum(counts)
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0)
    bins[0] = 0
    return bins, cdf
