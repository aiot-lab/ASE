
import numpy as np
import warnings
from parser_config import GlobalArgs, PlayArgs, ProcessArgs, DeviceArgs

warnings.simplefilter('ignore', np.RankWarning)
np.seterr(divide='ignore', invalid='ignore')

time_stamp = dict()
tau = dict()

SAVE_PEAK_VIOLIN = False
SAVE_MOTION_INDICATOR = False
SAVE_DIFFERENT_MRC = False
SAVE_VEL_ORIGIN = False
SAVE_ACF_MAT = True

# OPTITRACK_ROOT = "./optitrack/data/"
OPTITRACK_ROOT = "./sheng/"

weight_name = ["ms", "prominence", "sigmoid", "exponential"]
weight_choose_idx = 2

global_arg = GlobalArgs()
play_arg = PlayArgs()
process_arg = ProcessArgs()
device_arg = DeviceArgs()

# print(global_arg)
# print(play_arg)
# print(process_arg)
# print(device_arg)
