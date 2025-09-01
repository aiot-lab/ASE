import anyconfig as ac

# >>>>>>>>>>>>>>>>>>>>>>>Define the arguments<<<<<<<<<<<<<<<<<<<<<<<<<
ARG_LIST = ["global_arg", "play_arg", "process_arg", "device_arg"]

PLAY_ARG_LIST = ["sampling_rate", "duration", "amplitude",
                 "nchannels", "blocksize", "buffersize",
                 "modulation", "N_padding", "idle", "fc",
                 "wave", "load_data_play",
                 "dataplay_path", "dataplay_name",
                 "frame_length", "root", "f0", "f1", "delay_num"]

GLOBAL_ARG_LIST = ["delay", "set_play", "set_playAndRecord",
                   "set_save", "set_process", "compute_motion_stat",
                   "compute_speed", "task", "rec_idx", "save_root"]

PROCESS_ARG_LIST = ["windows_time", "windows_step",
                    "num_topK_subcarriers", "remove_start_time",
                    "remove_end_time", "perform_filter", "align", "set_preprocess"]

DEVICE_ARG_LIST = ["input_device", "output_device",
                   "input_channels", "output_channels"]

# >>>>>>>>>>>>>>>>>>>>>>>Set Parser Argument<<<<<<<<<<<<<<<<<<<<<<<<<


def set_play_args_arguments(parser):
    parser.add_argument('--sampling_rate', type=int,
                        default=48000, help="sampling rate")
    parser.add_argument("--duration", type=int, default=80,
                        help="duration of the wave")
    parser.add_argument("--amplitude", type=float,
                        default=0.1, help="amplitude of the wave")
    parser.add_argument("--nchannels", type=int, default=1,
                        help="number of channels")
    parser.add_argument("--blocksize", type=int,
                        default=1024, help="blocksize")
    parser.add_argument("--buffersize", type=int,
                        default=32, help="buffersize")
    parser.add_argument("--modulation", action="store_true",
                        help="w/ modulation")
    parser.add_argument("--N_padding", type=int, default=0,
                        help="padding length for modulation")
    parser.add_argument("--idle", type=int, default=0,
                        help="idle time")
    parser.add_argument("--orth", action="store_true",
                        help="w/ orthogonal modulation")
    parser.add_argument("--fc", type=float, default=20250,
                        help="carrier frequency")
    parser.add_argument("--wave", type=str, default="Kasami",
                        help="wave type")
    parser.add_argument("--load_data_play",
                        action="store_true", help="w/ load data to play")
    parser.add_argument("--dataplay_path", type=str,
                        default="./data_play/", help="path to dataplay")
    parser.add_argument("--dataplay_name", type=str,
                        help="name of dataplay")
    parser.add_argument("--frame_length", type=int,
                        default=1024, help="frame length")
    parser.add_argument("--root", type=int, default=None,
                        help="root of ZC sequence <ZC sequence>")
    parser.add_argument("--f0", type=float, default=18000,
                        help="start frequency <chirp>")
    parser.add_argument("--f1", type=float, default=22000,
                        help="end frequency <chirp>")
    parser.add_argument("--delay_num", type=int,
                        default=None, help="delay num of channel")


def set_process_args_arguments(parser):
    parser.add_argument('--windows_time', type=float,
                        default=2, help="windows time (s)")
    parser.add_argument("--windows_step", type=float,
                        default=None, help="windows step")
    parser.add_argument("--num_topK_subcarriers", type=int,
                        default=50, help="number of top K subcarriers")
    parser.add_argument("--remove_start_time", type=float,
                        default=0, help="remove the front time (s)")
    parser.add_argument("--remove_end_time", type=float,
                        default=0, help="remove the end time (s)")
    parser.add_argument("--perform_filter", action="store_true",
                        help="w/ perform 10k filter on data_rec")
    parser.add_argument("--align", action="store_true",
                        help="w/ align data_play and data_rec")
    parser.add_argument("--set_preprocess", action="store_true",
                        help="w/ preprocess data_rec")


def set_device_args_arguments(parser):
    parser.add_argument("--input_device", type=str,
                        default="default", help="input device")
    parser.add_argument("--output_device", type=str,
                        default="default", help="output device")
    parser.add_argument("--input_channels", type=int,
                        default=0, help="input channels")
    parser.add_argument("--output_channels", type=int,
                        default=0, help="output channels")


def set_global_args_arguments(parser):
    parser.add_argument("--delay", type=float, default=0,
                        help="delay before play")
    parser.add_argument("--set_play", action="store_true", help="play signal")
    parser.add_argument("--set_playAndRecord", action="store_true",
                        default=True, help="w/ play and record signal")
    parser.add_argument("--set_save", action="store_true",
                        default=True, help="record signal")
    parser.add_argument("--set_process", action="store_true",
                        help="w/ process recorded signal")
    parser.add_argument("--compute_motion_stat",
                        action="store_true", help="w/ compute motion statistics")
    parser.add_argument("--compute_speed",
                        action="store_true", help="w/ compute speed")
    parser.add_argument("--task", type=str, default="speed", help="task")
    parser.add_argument("--rec_idx", type=int, default=None,
                        help="Idx to record file")
    parser.add_argument("--save_root", type=str,
                        default="./data/", help="Root of save location")


def set_json_args_arguments(parser):
    parser.add_argument("json_file", type=str,
                        default="./config.json", help="json file")

# >>>>>>>>>>>>>>>>>>>>>>>Class Parser Argument<<<<<<<<<<<<<<<<<<<<<<<<<


class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)


class GlobalArgs(Args):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PlayArgs(Args):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ProcessArgs(Args):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DeviceArgs(Args):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# >>>>>>>>>>>>>>>>>>>>>>>Separate Parser Argument<<<<<<<<<<<<<<<<<<<<<<<<<


def parse_json_file(json_file, play_arg, process_arg, device_arg, global_arg):
    args_json_file = ac.load(json_file)
    for arg_item in ARG_LIST:
        if arg_item in args_json_file.keys():
            for key, value in args_json_file[arg_item].items():
                setattr(locals()[arg_item], key, value)


def parse_parser_args(arg_dict, play_arg, process_arg, device_arg, global_arg):
    for key, value in arg_dict.items():
        if key in PLAY_ARG_LIST:
            setattr(play_arg, key, value)
        elif key in PROCESS_ARG_LIST:
            setattr(process_arg, key, value)
        elif key in DEVICE_ARG_LIST:
            setattr(device_arg, key, value)
        elif key in GLOBAL_ARG_LIST:
            setattr(global_arg, key, value)
        else:
            raise ValueError("Unknown argument: {}".format(key))


# >>>>>>>>>>>>>>>>>>>>>>>Parser Dataplay Information<<<<<<<<<<<<<<<<<<<<<<<<<
DATAPLAY_NAME_ITEM_LIST = ["wave", "frame_length", "idle",
                           "duration", "orth", "modulation", "N_padding", "fc", "nchannels"]


def parse_dataplay_param(dataplay_name, play_arg):
    # split name by "_"
    params = dataplay_name[:-4].split("_")

    for idx_item, item in enumerate(DATAPLAY_NAME_ITEM_LIST):
        if idx_item < len(params):
            # convert string with int to int
            if item in ["frame_length", "idle", "duration", "N_padding", "fc"]:
                setattr(play_arg, item, int(params[idx_item]))
            else:
                setattr(play_arg, item, params[idx_item])
    if len(params) > 8:
        setattr(play_arg, "nchannels", min(int(params[8]), 2))

        setattr(play_arg, "delay_num", int(params[9]))

    else:
        setattr(play_arg, "nchannels", 1)
        setattr(play_arg, "delay_num", 0)

    if len(params) > 10:
        setattr(play_arg, "amplitude", float(params[10]))
    else:
        setattr(play_arg, "amplitude", 0.1)
