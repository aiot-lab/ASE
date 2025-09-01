
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

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
from sklearn.cluster import KMeans


from loguru import logger
import anyconfig as ac

from utils_gui import *
from config_gui import *

from stqdm import stqdm


import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
from time import sleep


default_json_file = "./config_file/speed_gui.json"

with open(default_json_file, 'r') as json_file:
    parser_config.parse_json_file(json_file,
                                  play_arg=play_arg,
                                  process_arg=process_arg,
                                  device_arg=device_arg,
                                  global_arg=global_arg)


st.set_page_config(
    page_title="ASE",
    page_icon="🔈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Title the app
st.title('Acoustic Speed Estimation (ASE)')

st.divider()

st.sidebar.markdown("## Setup Panel")

action_option = st.sidebar.selectbox(
    'Select action', options=['New Exp', 'Analysis'], index=1, key="action_option")


def show_sidebar():
    init_session_state("search_dict", None)
    with st.expander("Search Dataplay Parameters"):
        search_dataplay_parameters()

    search_dict = {
        "frame_length": int(st.session_state.frame_length_option),
        "duration": int(st.session_state.duration_option),
        "N_padding": int(st.session_state.N_padding_option),
        "orth": st.session_state.orth_option,
        "nchannels": int(st.session_state.nchannels_option),
        "delay_num": int(st.session_state.delay_num_option),
        "waveform": st.session_state.waveform_option,
    }
    if st.session_state.search_dict != search_dict:
        st.session_state.search_dict = search_dict
    st.session_state.search_dict

    load_data_play_path = st.selectbox(
        "Select corresponding data play", get_data_play(search_dict=st.session_state.search_dict), key="load_data_play_path")
    data_path_option = st.selectbox("Choose Data Setting",
                                    ["default setting", "others"],
                                    key="data_path_option"
                                    )
    if action_option == "New Exp":
        process = st.checkbox("Process after exp", value=True, key="process")
        location = st.selectbox("Select location", options=[
                                "OptiTrack", "Home", "HW101", "CYC313", "RailTrack"], key="location")
        people = st.text_input("People", value=None, key="people")
        exp_name = st.text_input("Exp name", value=None, key="exp_name")
        with st.form(key="exp_form"):
            exp_form()
            st.session_state.exp_form_submitted

    else:
        with st.form(key="analysis_form"):
            init_session_state("rec_idx", 0)
            analysis_form()
            st.session_state.analysis_submit


def show_parameter(play_arg, global_arg, process_arg, device_arg, show_param=True):
    param_dict = {
        "play_arg": play_arg,
        "global_arg": global_arg,
        "process_arg": process_arg,
        "device_arg": device_arg
    }
    if show_param:
        with st.expander("Show Parameters"):
            param_dict
    return param_dict


def analysis(global_arg, play_arg, device_arg, process_arg, show_metric=True):

    # >>>>>>>>>>>>>>>>> load and preprocess data >>>>>>>>>>>>>>>>>>>>>>
    dataloader, data = load_data(data_rec_idx=global_arg.rec_idx,
                                 _play_arg=play_arg,
                                 _global_arg=global_arg,
                                 _process_arg=process_arg)

    dataseq = dataloader.dataseq
    datarec = dataloader.datarec

    logger.debug("datarec shape: {}".format(datarec.shape))

    data_power = compute_dB(dataloader)
    setattr(play_arg, "power", data_power)
    logger.info("Data power: {}".format(str(data_power) + " dB"))
    param_dict = show_parameter(
        play_arg, global_arg, process_arg, device_arg)
    if show_metric:
        show_metric_param(**param_dict)
    # >>>>>>>>>>>>>>>>> decode and normalize data >>>>>>>>>>>>>>>>>>>>>>
    time_start_decode = perf_counter()

    datarec = decode_data(dataloader, play_arg, process_arg)

    time_decode = perf_counter() - time_start_decode
    logger.info("Decoding time: {}".format(time_decode))
    logger.info("Decoded data shape: {}".format(datarec.shape))
    datarec, dataseq = normalize_data(datarec, dataseq)
    dataloader.update(datarec=datarec, dataseq=dataseq)

    # >>>>>>>>>>>>>>>>> channel estimation >>>>>>>>>>>>>>>>>>>>>>
    cir, cfr = init_channel()
    cfr_freq = None

    time_start_cfr = perf_counter()
    for datarec, dataseq, channel_idx in dataloader:
        ce = channelWaveHandler(play_arg.wave,
                                play_arg=play_arg,
                                datarec=datarec,
                                dataplay=dataseq)
        # get cir
        cir_i = ce.CIR
        cir = process_cir(cir_i, cir)
        # get cfr
        cfr_i, cfr_freq = ce.CFR(cir_i)
        cfr, _ = process_cfr(cfr_i, cfr)
    cfr = np.squeeze(cfr)
    cfr_freq = np.squeeze(cfr_freq)

    time_cfr = perf_counter() - time_start_cfr
    logger.info("Time for CFR: {:.3f}s".format(time_cfr))

    logger.debug("cir.shape: {}".format(cir.shape))
    logger.debug("cfr.shape: {}".format(cfr.shape))

    # >>>>>>>>>>>>>>>>> merge cfr >>>>>>>>>>>>>>>>>>>>>>
    time_start_merge = perf_counter()
    cfr_merged = merge_CFR(ce, cfr=cfr)
    time_merge = perf_counter() - time_start_merge
    logger.info("Time for merge: {:.3f}s".format(time_merge))

    # >>>>>>>>>>>>>>>>> filter cfr >>>>>>>>>>>>>>>>>>>>>>
    cfr, _ = filter_CFR(cfr, cfr_freq, play_arg)
    cfr_merged, cfr_freq = filter_CFR(cfr_merged, cfr_freq, play_arg)

    # >>>>>>>>>>>>>>>>> normalize cfr >>>>>>>>>>>>>>>>>>>>>>
    cfr = normalize_cfr(cfr)
    cfr_merged = normalize_cfr(cfr_merged)

    logger.debug("cfr.shape: {}".format(cfr.shape))
    logger.debug("cfr_merged.shape: {}".format(cfr_merged.shape))
    logger.debug("cfr_freq.shape: {}".format(cfr_freq.shape))

    # >>>>>>>>>>>>>>>>> compute ACF >>>>>>>>>>>>>>>>>>>>>>
    time_start_acf = perf_counter()
    acf_merged = ACF(CFR=cfr_merged,
                     process_arg=process_arg)
    acf_merged_matrix = acf_merged.acf
    acf_merged_ms = acf_merged.ms

    time_acf = perf_counter() - time_start_acf
    logger.info("Time for ACF: {:.3f}s".format(time_acf))

    logger.debug("acf_merged_matrix.shape: {}".format(
        acf_merged_matrix.shape))
    logger.debug("acf_merged_ms.shape: {}".format(acf_merged_ms.shape))

    # todo: check windows_width and step_size
    acf = ACF(CFR=cfr, process_arg=process_arg)
    acf_ms = acf.ms
    acf_matrix = acf.acf

    logger.debug("acf_matrix.shape: {}".format(acf_matrix.shape))
    logger.debug("acf_ms.shape: {}".format(acf_ms.shape))

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

    # >>>>>>>>>>>>>>>>>>>>>>> visualize ms, zc sum, zc count <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    zc_o_aligned_m_count = np.mean(
        np.count_nonzero(zc_o_aligned_m, axis=0), axis=1)
    zc_count_threshold = get_threshold(zc_o_aligned_m_count) - 15
    if zc_count_threshold < 30:
        zc_count_threshold = 30
    vis_ms_line(y=zc_o_aligned_m_count,
                x=time_stamp["dual"],
                title="Motion Indicator",
                threshold=zc_count_threshold,)

    # >>>>>>>>>>>>>>>>>>>>>>> compute motion time <<<<<<<<<<<<<<<<<<<<<<<<<<<<
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

    logger.debug("tau_peak.shape: {}".format(tau_peak.shape))
    logger.debug("acf_merged_matrix.shape: {}".format(acf_merged_matrix.shape))

    logger.debug("len(origin_peak): {}".format(len(origin_peak)))
    logger.debug("tau_peak_prominence.shape: {}".format(
        tau_peak_prominence.shape))

    # >>>>>>>>>>>>>>>>>>>>>>> Get MRC Weight <<<<<<<<<<<<<<<<<<<<<<<<<<<<
    time_start_weight = perf_counter()

    weight_1 = acf_merged_ms
    weight_2 = tau_peak_prominence
    weight_3 = decay_weight(origin_peak, weight_2, fit_function="sigmoid")
    weight_4 = decay_weight(origin_peak, weight_2, fit_function="exp")

    time_weight = perf_counter() - time_start_weight
    logger.info("Time for weight: {:.3f} s".format(time_weight))

    logger.debug("weight_1.shape: {}".format(weight_1.shape))

    # >>>>>>>>>>>>>>>>>>>>>>>> MRC <<<<<<<<<<<<<<<<<<<<<<<<<<<
    acf_mrc_weight2 = MRC(acf_merged_matrix, weight_2)
    acf_mrc_weight1 = MRC(acf_merged_matrix, weight_1)
    acf_mrc_weight3 = MRC(acf_merged_matrix, weight_3)
    acf_mrc_weight4 = MRC(acf_merged_matrix, weight_4)

    logger.debug("acf_mrc_weight1.shape: {}".format(acf_mrc_weight1.shape))
    weight = [acf_mrc_weight1, acf_mrc_weight2,
              acf_mrc_weight3, acf_mrc_weight4]
    acf_mrc = weight[weight_choose_idx]
    logger.debug("acf_mrc.shape: {}".format(acf_mrc.shape))

    # >>>>>>>>>>>>>>>>>>>>>>>> Compute Speed <<<<<<<<<<<<<<<<<<<<<<<<<<<
    time_start_speed = perf_counter()
    vel, tau_s = get_speed(acf_mrc, time_stamp, tau)

    time_speed = perf_counter() - time_start_speed
    logger.info("time_speed: {:.3f}s".format(time_speed))
    vel[vel < 0] = np.nan
    tau_s[np.isnan(tau_s)] = np.inf
    tau_s[tau_s == 0] = np.nan
    tau_s[np.isinf(tau_s)] = 0

    # >>>>>>>>>>>>>>>>>>>>>>>> Visualize Speed <<<<<<<<<<<<<<<<<<<<<<<<<<<
    vis_vel = Visual(y=vel,
                     x=time_stamp["merged"],
                     xaxis_title="Time (s)",
                     yaxis_title="Velocity (m/s)",
                     title="Velocity",
                     )
    vis_vel.line()
    st.plotly_chart(vis_vel.fig, use_container_width=True)

    # >>>>>>>>>>>>>>>>>>>>>>>> Compute Detection Rate <<<<<<<<<<<<<<<<<<<<<<<<<<<
    dr = compute_detection_rate(vel)
    st.write("Detection Rate: {:.2f}%".format(dr * 100))

    return vel


def main(play_arg, global_arg, device_arg, process_arg):
    init_session_state("analysis_submit", False)
    init_session_state("exp_form_submitted", False)

    with st.sidebar:
        show_sidebar()

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Analysis <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if st.session_state.action_option == "Analysis" and st.session_state.analysis_submit:
        init_process(global_arg)

        parse_dataplay_param(
            st.session_state.load_data_play_path, play_arg)

        if st.session_state.data_idx_input:
            setattr(global_arg, "rec_idx", int(
                st.session_state.data_idx_input))
        if st.session_state.data_path_option == "default setting":
            set_dataplay(play_arg)

        global_arg, data_name, rec_idx = index_config(global_arg=global_arg,
                                                      play_arg=play_arg,
                                                      device_arg=device_arg)

        set_and_check_param(global_arg, play_arg, device_arg, process_arg)

        vel = analysis(
            global_arg, play_arg, device_arg, process_arg)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> New Exp <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if st.session_state.action_option == "New Exp" and st.session_state.exp_form_submitted == True:
        # >>>>>>>>>>>>>>>>>>>>>>> Initialize <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        init_cache()
        sleep(1)
        init_device(device_arg)
        init_device(device_arg)
        init_exp(global_arg)
        player = init_player(global_arg)
        parse_dataplay_param(
            st.session_state.load_data_play_path, play_arg)
        if st.session_state.data_path_option == "default setting":
            set_dataplay(play_arg)

        # >>>>>>>>>>>>>>>>>>>>>>> Index <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if hasattr(global_arg, "rec_idx"):
            delattr(global_arg, "rec_idx")
        global_arg, data_name, rec_idx = index_config(global_arg=global_arg,
                                                      play_arg=play_arg,
                                                      device_arg=device_arg)
        logger.info("Data name: {}".format(data_name))
        logger.info("idx: {}".format(rec_idx))

        # >>>>>>>>>>>>>>>>>>>>>>> Set and Check Parameter <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        set_and_check_param(global_arg, play_arg, device_arg, process_arg)
        param_dict = show_parameter(
            play_arg, global_arg, process_arg, device_arg, show_param=False)
        print(param_dict)
        show_metric_param(
            **param_dict, perform_analysis=False)

        logger.info("Delay {}s".format(global_arg.delay))

        # >>>>>>>>>>>>>>>>>>>>>>> Delay <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        delay_progress_bar(delay_time=global_arg.delay)
        # >>>>>>>>>>>>>>>>>>>>>>> Start Playing <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        player = set_player(player, play_arg=play_arg,
                            global_arg=global_arg)

        assert player is not None, "player is None"

        logger.info("Start playing ...")
        with st.spinner("Start playing ..."):
            player.begin()
            sleep(play_arg.duration)
            data_record = player.get_record()
            if global_arg.set_save:
                player.save_record()
            player.end()

        # >>>>>>>>>>>>>>>>>>>>>>> Analysis <<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if st.session_state.process:
            init_process(global_arg)
            set_and_check_param(global_arg, play_arg,
                                device_arg, process_arg)
            analysis(global_arg, play_arg, device_arg,
                     process_arg, show_metric=False)


if __name__ == "__main__":
    main(play_arg, global_arg, device_arg, process_arg)
