import numpy as np
import pandas as pd
import time
from scipy.io import loadmat
import os
from matplotlib import pyplot as plt

print("hello")
figure_save_path = "/Users/ethan/study/lab/code/ASE/fig/imwut"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path)


def get_data_path(data_idx, root_path=r"../data/"):
    dirs = [dirs for _, dirs, _ in os.walk(root_path)][0]

    name = [dir
            for dir in dirs if dir is not None
            and (dir.split("_")[-1]).isnumeric()
            and int(dir.split("_")[-1]) == data_idx]
    name = name[0]
    data_path = os.path.join(root_path, name, name)
    return data_path


def load_cdf(data_idx, root_path="../data/", baseline=False):
    data_path = get_data_path(data_idx, root_path=root_path)
    if not baseline:
        cdf_save_path = data_path + "_error_cdf.npy"
        hist_save_path = data_path + "_error_bins.npy"
    else:
        cdf_save_path = data_path + "_error_bl_cdf.npy"
        hist_save_path = data_path + "_error_bl_bins.npy"
    cdf = np.load(cdf_save_path)
    bins = np.load(hist_save_path)
    return cdf, bins


def load_acf_mat(data_idx, root_path="../data/"):
    data_path = get_data_path(data_idx, root_path=root_path)
    acf_mat_path = data_path + "_acf_mrc_matrix.npy"
    idx_path = data_path + "_idx_motion.npy"
    tau_peak_path = data_path + "_tau_peak.npy"
    vel_path = data_path + "_vel_o.npy"
    tau_merge_path = data_path + "_tau_merged.npy"
    acf_mat = np.load(acf_mat_path)
    idx_motion = np.load(idx_path)
    tau_peak = np.load(tau_peak_path)
    vel = np.load(vel_path)
    tau = np.load(tau_merge_path)

    return acf_mat, idx_motion, tau_peak, vel, tau


def load_vel_mean_error(data_idx, root_path="../data/", baseline=False, st=False):
    data_path = get_data_path(data_idx, root_path=root_path)
    if not baseline:
        v_mean_error_path = data_path + "_mean_error.npy"
    else:
        if not st:
            v_mean_error_path = data_path + "_baseline_error.npy"
        else:
            v_mean_error_path = data_path + "_st_baseline_error.npy"
    try:
        v_mean_error = np.load(v_mean_error_path)
    except FileNotFoundError:
        try:
            v_mean_error = np.load(data_path + "_v_mean_error.npy")
        except FileNotFoundError:
            raise FileNotFoundError(
                "File not found: {}".format(v_mean_error_path))
    except Exception as e:
        raise e
    return v_mean_error


def load_detection_rate(data_idx, root_path="../data/", type="pdf"):
    data_path = get_data_path(data_idx, root_path=root_path)
    dr_path = data_path + "_detect_rate.npy"
    dr = np.load(dr_path)
    return dr


def save_fig(fig, figure_name, save_root_path=figure_save_path, fig_type=".pdf", overwrite=False):
    save_path = os.path.join(save_root_path, figure_name)
    fig_path = save_path + fig_type
    fig_png_path = save_path + ".png"
    if not os.path.exists(fig_path) or overwrite:
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')

    else:
        print("File already exists")
    if not os.path.exists(fig_png_path) or overwrite:
        fig.savefig(fig_png_path, dpi=300, bbox_inches='tight')
    else:
        print("File already exists")


def load_violin_data(data_idx, t_idx, root_path="../data/"):
    data_path = get_data_path(data_idx, root_path=root_path)
    peak_violin_path = data_path + "_peak_violin_{}.npy".format(t_idx)
    cfr_freq_path = data_path + "_cfr_freq.npy"
    tau_p_p_path = data_path + "_tau_p_p_{}.npy".format(t_idx)
    tau_peak_path = data_path + \
        "_tau_peak_{}.npy".format(t_idx)
    weight_path = data_path + "_weight_3_{}.npy".format(t_idx)
    try:
        peak = np.load(peak_violin_path)
        freq = np.load(cfr_freq_path)
        weight = np.load(tau_p_p_path)
        tau_peak = np.load(tau_peak_path)
        weight_decay = np.load(weight_path)
    except FileNotFoundError:
        os.error("File not found")
    except Exception as e:
        print(e)
    return peak, freq, weight, tau_peak, weight_decay


weight_name = ["ms", "prominence", "sigmoid", "exponential"]
weight_choose_idx = 3


def load_mrc(data_idx, root_path="../data/", weight_choose_idx=weight_choose_idx):
    data_path = get_data_path(data_idx, root_path=root_path)
    v_mean_error_mrc_save_path = data_path + \
        "_mean_error_{}".format(weight_name[weight_choose_idx])
    v_error_bins_mrc_save_path = data_path + \
        "_error_bins_{}".format(weight_name[weight_choose_idx])
    v_error_cdf_mrc_save_path = data_path + \
        "_error_cdf_{}".format(weight_name[weight_choose_idx])

    v_mean_error = np.load(v_mean_error_mrc_save_path + ".npy")
    bins = np.load(v_error_bins_mrc_save_path + ".npy")
    cdf = np.load(v_error_cdf_mrc_save_path + ".npy")

    return v_mean_error, bins, cdf


def load_motion_indicator(data_idx, root_path="../data/"):
    data_path = get_data_path(data_idx, root_path=root_path)
    zc_o_aligned_m_path = data_path + "_zc_o_aligned_m.npy"
    zc_d_aligned_m_path = data_path + "_zc_d_aligned_m.npy"
    zc_count_threshold_path = data_path + "_zc_count_threshold.npy"
    zc_o_aligned_m_count_path = data_path + "_zc_o_aligned_m_count.npy"
    time_stamp_dual_path = data_path + "_time_stamp_dual.npy"
    cfr_freq_path = data_path + "_cfr_freq.npy"

    zc_o_aligned_m = np.load(zc_o_aligned_m_path)
    zc_d_aligned_m = np.load(zc_d_aligned_m_path)
    zc_count_threshold = np.load(zc_count_threshold_path)
    zc_o_aligned_m_count = np.load(zc_o_aligned_m_count_path)
    time_stamp_dual = np.load(time_stamp_dual_path)
    cfr_freq = np.load(cfr_freq_path)
    return zc_o_aligned_m, zc_d_aligned_m, zc_count_threshold, zc_o_aligned_m_count, time_stamp_dual, cfr_freq


def plot_dual_yaxis(y1, y2, xticklabels, xlabel, ylabel, y2label, label_name=None,
                    colors=['tab:blue', 'tab:green'],
                    width=0.2,
                    y1_lim=None,
                    y2_lim=None,
                    figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    x = np.arange(len(xticklabels))
    if label_name is not None:
        ax2.bar(x + width, y2, width=width,
                color=colors[1], label=label_name[1], edgecolor='black', hatch='\\')
        ax.bar(x, y1, width=width,
               color=colors[0], label=label_name[0], edgecolor='black', hatch='/')
    else:
        ax2.bar(x + width, y2, width=width,
                color=colors[1], edgecolor='black', hatch='\\')
        ax.bar(x, y1, width=width,
               color=colors[0], edgecolor='black', hatch='/')

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(xticklabels)

    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, color=colors[0])
    ax.spines['left'].set_color(colors[0])
    ax.tick_params(axis='y', colors=colors[0])
    ax2.set_ylabel(y2label,
                   color=colors[1], rotation=270, labelpad=20)
    ax2.spines['right'].set_color(colors[1])
    ax2.tick_params(axis='y', colors=colors[1])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    if y1_lim is not None:
        ax.set_ylim(y1_lim)
    if y2_lim is not None:
        ax2.set_ylim(y2_lim)
    return fig
