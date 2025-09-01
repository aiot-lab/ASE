
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import click
import json
import os

import parser_config
from index_manager import index_config
from check_param import *
from parser_config import parse_dataplay_param
from audio.Audio import AudioPlayer, AudioPlayandRecord
from audio.AudioProcessing import channelWaveHandler
from audio.helpers import init_channel, normalize_data, process_cir, process_cfr, filter_CFR, normalize_cfr, get_acoustic_device
from model.ACF import ACF, MRC
from model.helpers import *
from model.timestamp import get_time, get_tau
from vis.visual import Visual

from scipy.signal import find_peaks, peak_prominences, stft
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

from loguru import logger
import anyconfig as ac

from app_utils import *
from config_gui import *

from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
from time import sleep

# compress warning
import warnings
warnings.filterwarnings("ignore")

# --- Helper functions copied from app.py dependencies ---


def analysis_core(global_arg, play_arg, device_arg, process_arg, config_data, show_metric=True):
    weight_choose_idx = config_data.get(
        "analysis_settings", {}).get("weight_choose_idx", 3)

    # >>>>>>>>>>>>>>>>> load and preprocess data >>>>>>>>>>>>>>>>>>>>>>
    dataloader, data = load_data(data_rec_idx=global_arg.rec_idx,
                                 _play_arg=play_arg,
                                 _global_arg=global_arg,
                                 _process_arg=process_arg)

    dataseq = dataloader.dataseq
    datarec = dataloader.datarec

    data_power = compute_dB(dataloader)
    setattr(play_arg, "power", data_power)
    logger.info("Data power: {}".format(str(data_power) + " dB"))

    param_dict = {
        "play_arg": play_arg,
        "global_arg": global_arg,
        "process_arg": process_arg,
        "device_arg": device_arg
    }

    # >>>>>>>>>>>>>>>>> decode and normalize data >>>>>>>>>>>>>>>>>>>>>>
    datarec = decode_data(dataloader, play_arg, process_arg)
    logger.info("Decoded data shape: {}".format(datarec.shape))

    datarec, dataseq = normalize_data(datarec, dataseq)
    dataloader.update(datarec=datarec, dataseq=dataseq)

    # >>>>>>>>>>>>>>>>> channel estimation >>>>>>>>>>>>>>>>>>>>>>
    cir, cfr = init_channel()
    cfr_freq = None

    for datarec, dataseq, channel_idx in tqdm(dataloader, desc="Channel Estimation"):
        ce = channelWaveHandler(play_arg.wave,
                                play_arg=play_arg,
                                datarec=datarec,
                                dataplay=dataseq)
        cir_i = ce.CIR
        cir = process_cir(cir_i, cir)
        cfr_i, cfr_freq = ce.CFR(cir_i)
        cfr, _ = process_cfr(cfr_i, cfr)

    cfr = np.squeeze(cfr)
    cfr_freq = np.squeeze(cfr_freq)

    # >>>>>>>>>>>>>>>>> merge, filter, normalize cfr >>>>>>>>>>>>>>>>>>>>>>
    cfr_merged = merge_CFR(ce, cfr=cfr)

    cfr, _ = filter_CFR(cfr, cfr_freq, play_arg)
    cfr_merged, cfr_freq = filter_CFR(cfr_merged, cfr_freq, play_arg)

    cfr = normalize_cfr(cfr)
    cfr_merged = normalize_cfr(cfr_merged)

    # >>>>>>>>>>>>>>>>> compute ACF >>>>>>>>>>>>>>>>>>>>>>
    acf_merged = ACF(CFR=cfr_merged, process_arg=process_arg)
    acf_merged_matrix = acf_merged.acf
    acf_merged_ms = acf_merged.ms

    acf = ACF(CFR=cfr, process_arg=process_arg)
    acf_ms = acf.ms
    acf_matrix = acf.acf

    # >>>>>>>>>>>>>>>>>>>>>>> get time stamp <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    time_stamp["merged"] = get_time(window_step=process_arg.windows_step,
                                    total_duration=play_arg.duration,
                                    rho_time=acf_merged_matrix.shape[2],
                                    CFR_time=cfr_merged.shape[1])

    time_stamp["dual"] = get_time(window_step=round(process_arg.windows_step / cfr.shape[2]),
                                  total_duration=play_arg.duration,
                                  rho_time=acf_matrix.shape[2],
                                  CFR_time=cfr.shape[1])

    tau["merged"] = get_tau(window_time=process_arg.windows_time,
                            CFR_rate=play_arg.channel_rate,
                            tau_time=acf_merged_matrix.shape[1],)
    tau["dual"] = get_tau(window_time=process_arg.windows_time,
                          CFR_rate=play_arg.channel_rate,
                          tau_time=acf_matrix.shape[1],)

    # >>>>>>>>>>>>>>>>>>>>>>> remove and align <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    acf_merged_matrix, acf_merged_ms, zc_o, zc_d, zc_o_aligned, zc_d_aligned, acf_merged_wo_alignment = remove_and_align(acf_merged_matrix,
                                                                                                                         acf_merged_ms,
                                                                                                                         freq=cfr_freq,
                                                                                                                         tau=tau["merged"])

    acf_matrix, acf_ms, _, _, zc_o_aligned_m, zc_d_aligned_m, _ = remove_and_align(acf_matrix,
                                                                                   acf_ms,
                                                                                   freq=cfr_freq,
                                                                                   tau=tau["dual"])
    from model.helpers import interp_tau
    tau_q = interp_tau()
    tau["merged"] = tau_q
    tau["dual"] = tau_q

    assert acf_matrix.shape[1] == tau["merged"].shape[0]

    # >>>>>>>>>>>>>>>>>>>>>>> compute motion time <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    zc_o_aligned_m_count = np.mean(
        np.count_nonzero(zc_o_aligned_m, axis=0), axis=1)
    zc_count_threshold = get_threshold(zc_o_aligned_m_count) - 15
    if zc_count_threshold < 30:
        zc_count_threshold = 30

    motion_list = np.where(zc_o_aligned_m_count >=
                           zc_count_threshold)[0].tolist()
    motion_list_group = group_time_idx(motion_list)

    t_motion_merged, idx_motion_list = merge_time_motion(merged_time=time_stamp["merged"],
                                                         dual_time=time_stamp["dual"],
                                                         motion_list_group=motion_list_group,)

    time_stamp["motion"] = {"dual": time_stamp["dual"][motion_list],
                            "merged": t_motion_merged,
                            "idx": idx_motion_list, }

    # >>>>>>>>>>>>>>>>>>>>>>> compute peak prominence <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    tau_peak, tau_peak_prominence, origin_peak = get_peak_prominence(acf_merged_matrix,
                                                                     time_stamp,
                                                                     tau,
                                                                     prominence_threshold=1e-2,
                                                                     )

    # >>>>>>>>>>>>>>>>>>>>>>> Get MRC Weight <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    weight_1 = acf_merged_ms
    weight_2 = tau_peak_prominence
    weight_3 = decay_weight(origin_peak, weight_2, fit_function="sigmoid")
    weight_4 = decay_weight(origin_peak, weight_2, fit_function="exp")

    # >>>>>>>>>>>>>>>>>>>>>>>> MRC <<<<<<<<<<<<<<<<<<<<<<<<<<<
    acf_mrc_weight1 = MRC(acf_merged_matrix, weight_1)
    acf_mrc_weight2 = MRC(acf_merged_matrix, weight_2)
    acf_mrc_weight3 = MRC(acf_merged_matrix, weight_3)
    acf_mrc_weight4 = MRC(acf_merged_matrix, weight_4)

    weight = [acf_mrc_weight1, acf_mrc_weight2,
              acf_mrc_weight3, acf_mrc_weight4]
    acf_mrc = weight[weight_choose_idx]

    # >>>>>>>>>>>>>>>>>>>>>>>> Compute Speed <<<<<<<<<<<<<<<<<<<<<<<<<<<
    vel, tau_s = get_speed(acf_mrc, time_stamp, tau)
    vel[vel < 0] = np.nan

    tau_s[np.isnan(tau_s)] = np.inf
    tau_s[tau_s == 0] = np.nan
    tau_s[np.isinf(tau_s)] = 0

    dr = compute_detection_rate(vel)
    logger.info("Detection Rate: {:.2f}%".format(dr * 100))

    return vel

# --- CLI Setup ---


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    """Acoustic Speed Estimation Command-Line Tool"""
    pass


@cli.command()
@click.option('--config', 'config_path', default="config_file/speed_gui.json", help='Path to the configuration file.')
@click.option('--rec-idx', 'rec_idx_cli', type=int, default=None, help='Recording index to process.')
def analysis(config_path, rec_idx_cli):
    """Run analysis using a specified configuration file."""

    play_arg = parser_config.PlayArgs()
    global_arg = parser_config.GlobalArgs()
    device_arg = parser_config.DeviceArgs()
    process_arg = parser_config.ProcessArgs()

    with open(config_path, 'r') as f:
        parser_config.parse_json_file(
            f, play_arg, process_arg, device_arg, global_arg)
        f.seek(0)  # Reset file pointer to read the full config dict
        config_data = json.load(f)

    analysis_settings = config_data.get("analysis_settings", {})
    data_handling_settings = config_data.get("data_handling", {})

    if rec_idx_cli is not None:
        setattr(global_arg, "rec_idx", rec_idx_cli)
    else:
        setattr(global_arg, "rec_idx", analysis_settings.get("data_rec_idx"))

    logger.info(f"Starting analysis for recording index: {global_arg.rec_idx}")

    init_process(global_arg)
    parse_dataplay_param(data_handling_settings.get(
        "load_data_play_path"), play_arg)

    set_dataplay(play_arg, data_handling_settings.get(
        "load_data_play_path"))

    global_arg, data_name, rec_idx = index_config(
        global_arg=global_arg, play_arg=play_arg, device_arg=device_arg)
    set_and_check_param(global_arg, play_arg, device_arg, process_arg)

    analysis_core(global_arg, play_arg, device_arg,
                  process_arg, config_data)


if __name__ == "__main__":
    cli()
