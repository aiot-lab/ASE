import numpy as np
import plotly.express as px
from time import sleep
from loguru import logger
import anyconfig as ac
import json
import yaml
import os
import plotly.graph_objects as go
from tqdm import tqdm

from dataloader import AcousticDataset, AcousticDataloader
from audio.AudioProcessing import AudioProcess, AudioDecoder, ChannelEstimation
from audio.Audio import AudioPlayer, AudioPlayandRecord
from audio.helpers import get_SPL_dB, get_acoustic_device
from model.ACF import ACF
from vis.visual import Visual


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


def decode_data(_dataloader, _play_arg, _process_arg):
    assert isinstance(_dataloader, AcousticDataloader)
    datarec = _dataloader.datarec
    assert datarec is not None

    decoder = AudioDecoder(datarec, _play_arg, _process_arg)
    data_rec_decode = decoder()
    return data_rec_decode


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

    logger.info("Parameters:")
    for key, value in display_dict.items():
        logger.info(f"  {key}: {value}")


def compute_dB(_dataloader):
    assert isinstance(_dataloader, AcousticDataloader)
    datarec = _dataloader.datarec
    dataplay = _dataloader.dataplay
    return get_SPL_dB(datarec)


def merge_CFR(_channel_estimation, cfr):
    assert isinstance(_channel_estimation, ChannelEstimation)
    return _channel_estimation.merge_CFR(cfr)


def get_ACF(_acf):
    assert isinstance(_acf, ACF)
    return _acf.acf


def get_ms(_acf):
    assert isinstance(_acf, ACF)
    return _acf.ms


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def vis_zc_heatmap(x, y, z,
                   title,
                   save_path,
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
    ensure_dir(save_path)
    vis_zc.figure.write_html(save_path)
    logger.info(f"Saved ZC heatmap to {save_path}")


def vis_acf_mat_heatmap(x, y, z,
                        title,
                        save_path,
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
    ensure_dir(save_path)
    vis_acf_merged_matrix.fig.write_html(save_path)
    logger.info(f"Saved ACF Matrix heatmap to {save_path}")


def vis_ms_line(x, y,
                title,
                save_path,
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
    ensure_dir(save_path)
    vis_ms.fig.write_html(save_path)
    logger.info(f"Saved MS line plot to {save_path}")


def vis_acf_mrc_heatmap(x, y, z,
                        title,
                        save_path,
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

    ensure_dir(save_path)
    vis_acf_mrc.fig.write_html(save_path)
    logger.info(f"Saved ACF MRC heatmap to {save_path}")


def vis_acf_tau_scatter(x, y,
                        title,
                        save_path,
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
    ensure_dir(save_path)
    vis_acf_tau_f.fig.write_html(save_path)
    logger.info(f"Saved ACF Tau scatter plot to {save_path}")


def init_player(global_arg):
    setattr(global_arg, "set_playAndRecord", True)
    player = None
    return player


def init_process(global_arg):
    setattr(global_arg, "set_process", True)


def set_dataplay(play_arg, dataplay_name):
    setattr(play_arg, "load_dataplay", True)
    setattr(play_arg, "dataplay_path", "./data_play/")
    setattr(play_arg, "dataplay_name", dataplay_name)


def delay_progress_bar(delay_time):
    for _ in tqdm(range(delay_time), desc="Delay"):
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
