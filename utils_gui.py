import streamlit as st
import numpy as np
import plotly.express as px
from time import sleep
from loguru import logger
import anyconfig as ac
import json
import yaml
import os
import plotly.graph_objects as go
from stqdm import stqdm

from dataloader import AcousticDataset, AcousticDataloader
from audio.AudioProcessing import AudioProcess, AudioDecoder, ChannelEstimation
from audio.Audio import AudioPlayer, AudioPlayandRecord
from audio.helpers import get_SPL_dB, get_acoustic_device
from model.ACF import ACF
from vis.visual import Visual


@st.cache_data
def load_data(data_rec_idx, _play_arg, _global_arg, _process_arg):
    # 2.1 Loading Data recordings
    data = AcousticDataset(_global_arg, file_type="mat").dataset

    # 2.2 Process Data recordings
    if _process_arg.set_preprocess:
        preprocess = AudioProcess(data, _play_arg)
        data = preprocess(process_arg=_process_arg)

    # 2.3 Pair Data record / play
    dataloader = AcousticDataloader(dataset=data, play_arg=_play_arg)

    return dataloader, data


@st.cache_data
def decode_data(_dataloader, _play_arg, _process_arg):
    assert isinstance(_dataloader, AcousticDataloader)
    datarec = _dataloader.datarec
    assert datarec is not None

    decoder = AudioDecoder(datarec, _play_arg, _process_arg)
    data_rec_decode = decoder()
    return data_rec_decode


@st.cache_resource
def vis_audio(_data, fs):
    st.audio(_data, format="audio/wav", start_time=0,
             sample_rate=fs)


def init_session_state(key, init_state):
    if key not in st.session_state.keys():
        st.session_state[key] = init_state


def search_dataplay_parameters():
    waveform_option = st.selectbox(
        "Waveform", options=["Kasami", "ZC", "Sine"], index=0, key="waveform_option")
    frame_length_option = st.selectbox(
        "Frame Length", options=[15, 39, 63, 1023], index=2, key="frame_length_option")
    duration_option = st.selectbox(
        "Duration", options=[40, 80], index=0, key="duration_option")
    N_padding_option = st.selectbox(
        "Modulated Length", options=[256, 480, 512, 1023, 1024], index=2, key="N_padding_option")
    orth_option = st.selectbox(
        "Orthogonal", options=["true", "false"], index=0, key="orth_option")
    nchannels_option = st.selectbox(
        "Nchannels", options=[1, 2, 4], index=0, key="nchannels_option")
    delay_num_option = st.selectbox(
        "Delay_num", options=[0, 256, 512], index=1, key="delay_num_option")


def get_data_play(data_play_path="./data_play/", search_dict=None):
    # get all the wav filr in the data_play folder
    import os
    data_play = [file for file in os.listdir(
        data_play_path) if file.endswith(".wav")]
    if search_dict is not None:
        data_play_search = []
        for dataplay in data_play:
            params = dataplay[:-4].split("_")
            frame_length = int(params[1])
            duration = int(params[3])
            modulation_length = int(params[6])
            orth = params[4]
            waveform = params[0]

            if len(params) > 8:
                nchannels = int(params[8])
                delay_num = int(params[9])
            if frame_length == search_dict["frame_length"] and duration == search_dict["duration"] \
                    and modulation_length == search_dict["N_padding"] and orth == search_dict["orth"] \
            and nchannels == search_dict["nchannels"] and search_dict["delay_num"] == delay_num and waveform == search_dict["waveform"]:
                data_play_search.append(dataplay)
        return data_play_search

    return data_play


def find_idx_max(data_path="./data/"):
    import os
    dirs = [dirs for _, dirs, _ in os.walk(data_path)][0]
    _idx = max([int(dir.split("_")[-1])
                for dir in dirs if dir is not None
                and (dir.split("_")[-1]).isnumeric()])
    return _idx


def on_click_analysis():
    if st.session_state.rec_idx != st.session_state.data_idx_input:
        st.cache_data.clear()
        st.cache_resource.clear()
    st.session_state.rec_idx = st.session_state.data_idx_input


def exp_form():
    input_devices, output_devices = get_acoustic_device()
    input_devices_option = st.selectbox(
        "Choose Input Device", options=input_devices, key="input_device")
    output_devices_option = st.selectbox(
        "Choose Output Device", options=output_devices, key="output_device")

    # Simple checkbox to enable/disable RealSense without extra warning
    use_realsense = st.checkbox(
        "Enable RealSense D435", value=False, key="use_realsense")

    exp_form_submitted = st.form_submit_button("Submit")
    if exp_form_submitted:
        st.session_state.exp_form_submitted = True


def analysis_form():
    if st.session_state.data_path_option == "others":
        data_path = st.file_uploader("Upload Data Path", type=["wav"])
    elif st.session_state.data_path_option == "default setting":
        st.write("default file location: ./data/")
        st.write("Choose the index of the data you want to use")

        data_idx_input = st.number_input(
            'Type Data Index', min_value=0, max_value=find_idx_max(), value=find_idx_max(), step=1, key="data_idx_input")

    analysis_submit = st.form_submit_button(
        "Submit", on_click=on_click_analysis)

    if st.session_state.analysis_submit != analysis_submit:
        st.session_state.analysis_submit = analysis_submit


def show_metric_param(play_arg, global_arg, process_arg, device_arg, perform_analysis=True):
    if perform_analysis:
        display_dict = {
            "data_idx": global_arg.rec_idx,
            "Amplitude": str(play_arg.amplitude),
            "wave": play_arg.wave,
            "CFR Rate": str(play_arg.channel_rate) + "Hz",
            "Power": str(play_arg.power) + "dB",
        }
    else:
        display_dict = {
            "data_idx": global_arg.rec_idx,
            "Amplitude": str(play_arg.amplitude),
            "wave": play_arg.wave,
            "CFR Rate": str(play_arg.channel_rate) + "Hz",
            "Delay": str(global_arg.delay),
        }
    col_metric = st.columns(len(display_dict))
    for idx, (key, value) in enumerate(display_dict.items()):
        col_metric[idx].metric(key, value)
    return col_metric


def compute_dB(_dataloader):
    assert isinstance(_dataloader, AcousticDataloader)
    datarec = _dataloader.datarec
    dataplay = _dataloader.dataplay
    return get_SPL_dB(datarec)


@st.cache_resource
def merge_CFR(_channel_estimation, cfr):
    assert isinstance(_channel_estimation, ChannelEstimation)
    return _channel_estimation.merge_CFR(cfr)


@st.cache_resource
def get_ACF(_acf):
    assert isinstance(_acf, ACF)
    return _acf.acf


@st.cache_resource
def get_ms(_acf):
    assert isinstance(_acf, ACF)
    return _acf.ms


@st.cache_resource
def vis_zc_heatmap(x, y, z,
                   title,
                   xaxis_title="Time (sequence)",
                   yaxis_title="Frequency (kHz)",
                   update=True,
                   styles=["surface", "heatmap"]):
    vis_zc = Visual(
        z=z,
        y=y,
        x=x,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    )
    vis_zc.heatmap(update=update, styles=styles)
    st.plotly_chart(vis_zc.figure, use_container_width=True)


@st.cache_resource
def vis_acf_mat_heatmap(x, y, z,
                        title,
                        xaxis_title="Frequency (kHz)",
                        yaxis_title="tau (sequence)",
                        update=True,
                        styles=["surface", "heatmap"],
                        animates=["Play", "Pause"],
                        frames=None):

    vis_acf_merged_matrix = Visual(x=x,
                                   y=y,
                                   z=z,
                                   title=title,
                                   xaxis_title=xaxis_title,
                                   yaxis_title=yaxis_title,
                                   )

    vis_acf_merged_matrix.heatmap(update=update,
                                  styles=styles,
                                  animates=animates,
                                  frames=frames)

    st.plotly_chart(vis_acf_merged_matrix.fig, use_container_width=True)


@st.cache_resource
def vis_ms_line(x, y,
                title,
                xaxis_title="Time (s)",
                yaxis_title="Motion Stat",
                threshold=None):
    vis_ms = Visual(x=x,
                    y=y,
                    title=title,
                    xaxis_title=xaxis_title,
                    yaxis_title=yaxis_title,

                    )
    vis_ms.line(threshold=threshold)
    st.plotly_chart(vis_ms.fig, use_container_width=True)


def vis_acf_mrc_heatmap(x, y, z,
                        title,
                        xaxis_title="time(s)",
                        yaxis_title="tau (s)",
                        update=False,
                        styles=["surface", "heatmap"],
                        caxis={"cmin": -0.3, "cmax": 0.8},
                        height=None,
                        y_range=None,
                        reverse=True,
                        peak=None):
    if height is None:
        height = 300
    vis_acf_mrc = Visual(x=x,
                         y=y,
                         z=z,
                         title=title,
                         xaxis_title=xaxis_title,
                         yaxis_title=yaxis_title,
                         height=height,
                         )

    vis_acf_mrc.heatmap(update=update,
                        styles=styles,
                        caxis=caxis,)
    vis_acf_mrc
    if peak is not None:
        vis_acf_mrc.fig.add_trace(go.Scatter(y=peak,
                                             x=x,
                                             mode="markers",
                                             marker={"color": "red", "size": 5},))

    if y_range is not None:
        vis_acf_mrc.update_yaxes(range=y_range)

    st.plotly_chart(vis_acf_mrc.fig, use_container_width=True)


def vis_acf_tau_scatter(x, y,
                        title,
                        xaxis_title="tau(s)",
                        yaxis_title="f(kHz)",
                        marker_size=None,
                        marker_color=None,
                        update=False,
                        animates=["Play", "Pause"],
                        x_range=None,):

    vis_acf_tau_f = Visual(x=x,
                           y=y,
                           title=title,
                           xaxis_title=xaxis_title,
                           yaxis_title=yaxis_title,)
    vis_acf_tau_f.scatter(
        marker_size=marker_size,
        marker_color=marker_color,
        update=update,
        animates=animates,
    )
    if x_range is not None:
        vis_acf_tau_f.update_xaxes(range=x_range)
    st.plotly_chart(vis_acf_tau_f.fig, use_container_width=True)


def init_device(device_arg):
    setattr(device_arg, "input_device", st.session_state.input_device)
    setattr(device_arg, "output_device", st.session_state.output_device)


def init_exp(global_arg):
    setattr(global_arg, "location", st.session_state.location)
    setattr(global_arg, "people", st.session_state.people)
    setattr(global_arg, "exp_name", st.session_state.exp_name)


def init_player(global_arg):
    setattr(global_arg, "set_playAndRecord", True)
    player = None
    return player


def init_process(global_arg):
    setattr(global_arg, "set_process", True)


def set_dataplay(play_arg):
    setattr(play_arg, "load_dataplay", True)
    setattr(play_arg, "dataplay_path", "./data_play/")
    setattr(play_arg, "dataplay_name", st.session_state.load_data_play_path)


def delay_progress_bar(delay_time):
    for _ in stqdm(range(delay_time), desc="Delay"):
        sleep(1)


def set_player(player, play_arg, global_arg):
    if global_arg.set_play:
        player = AudioPlayer(play_arg)
    elif global_arg.set_playAndRecord:
        player = AudioPlayandRecord(play_arg, path=global_arg.data_path)
    return player


def notion_update(play_arg, global_arg):
    notion_param_dict = {
        "duration": play_arg.duration,
        "amplitude": play_arg.amplitude,
        "modulation": play_arg.modulation,
        "nchannels": play_arg.nchannels,
        "samples_per_time": play_arg.samples_per_time,
        "orth": play_arg.orth,
        "wave": play_arg.wave,
        "location": global_arg.location,
        "people": global_arg.people,
    }
    from update_notion import NotionUpdater
    notion_updater = NotionUpdater(rec_idx=global_arg.rec_idx,
                                   notion_param_dict=notion_param_dict,
                                   notion_comment_list=[global_arg.exp_name])
    notion_updater.push_new_page()


def init_cache():
    st.cache_data.clear()
    st.cache_resource.clear()


def save_log(**kwargs):
    assert hasattr(kwargs["global_arg"], "data_path")
    kwargs["global_arg"].log_path = kwargs["global_arg"].data_path + ".log"
    logger.add(kwargs["global_arg"].log_path)
    param_json = dict()
    for key, value in kwargs.items():
        if value is not None:
            param_json[key] = value.__dict__
    print(param_json)
    with open(kwargs["global_arg"].data_path + "_params.json", "w") as f:
        try:
            json.dump(param_json, f, indent=4)
        except TypeError as e:
            import os
            os.error(e)
    with open(kwargs["global_arg"].data_path + "_params.yaml", "w") as f:
        try:
            yaml.dump(param_json, f, indent=4)
        except TypeError as e:
            import os
            os.error(e)
