import sounddevice as sd
from math import ceil, log2
from loguru import logger

WAVE_OPTIONS = ["chirp", "Kasami", "Golay", "ZC", "sine"]
SAMPLING_RATE_OPTIONS = [44100, 48000, 96000, 192000]


def check_channels(play_arg, device_arg):
    if hasattr(play_arg, "nchannels"):
        if hasattr(device_arg, "output_channels"):
            assert (play_arg.nchannels == device_arg.output_channels)
        else:
            device_arg.output_channels = min(play_arg.nchannels, 2)
    if not hasattr(device_arg, "input_channels"):
        device_arg.input_channels = 1


def set_and_check_device(play_arg, device_arg):
    sd.default.device = device_arg.input_device, device_arg.output_device
    sd.default.samplerate = play_arg.sampling_rate

    sd.default.channels = device_arg.input_channels, device_arg.output_channels
    assert (sd.check_input_settings() == None)
    assert (sd.check_output_settings() == None)


def check_record_and_save(global_arg):
    if hasattr(global_arg, "set_playAndRecord") and global_arg.set_playAndRecord:
        assert (global_arg.set_save == True)
        assert (not hasattr(global_arg, "set_play")
                or global_arg.set_play == False)
    if not hasattr(global_arg, "set_play"):
        global_arg.set_play = False
    if not hasattr(global_arg, "set_playAndRecord"):
        global_arg.set_playAndRecord = False
    logger.debug("set_play: {}".format(global_arg.set_play))


def check_load_dataplay(play_arg):
    if hasattr(play_arg, "load_dataplay") and play_arg.load_dataplay:
        assert (hasattr(play_arg, "dataplay_path")
                and play_arg.dataplay_path != None)
        assert (hasattr(play_arg, "dataplay_name")
                and play_arg.dataplay_name != None)


def set_and_check_wave(play_arg):
    assert (hasattr(play_arg, "wave") and play_arg.wave in WAVE_OPTIONS)
    assert (hasattr(play_arg, "sampling_rate")
            and play_arg.sampling_rate in SAMPLING_RATE_OPTIONS)
    assert (hasattr(play_arg, "amplitude") and play_arg.amplitude > 0)
    assert (hasattr(play_arg, "frame_length")
            and play_arg.frame_length != None)
    assert (hasattr(play_arg, "nchannels") and play_arg.nchannels > 0)
    assert (hasattr(play_arg, "duration") and play_arg.duration > 0)

    play_arg.samples_per_time = play_arg.frame_length
    if hasattr(play_arg, "modulation"):
        assert (hasattr(play_arg, "N_padding") and play_arg.N_padding != None)
        assert (hasattr(play_arg, "fc") and play_arg.fc != None)
        play_arg.samples_per_time = play_arg.N_padding

    play_arg.length = max(play_arg.frame_length, play_arg.N_padding)

    if not hasattr(play_arg, "bandwidth"):
        bandwidth = play_arg.frame_length * play_arg.sampling_rate / play_arg.N_padding
        setattr(play_arg, "bandwidth", bandwidth)

    if hasattr(play_arg, "idle") and play_arg.idle > 0:
        play_arg.length += play_arg.idle
    if hasattr(play_arg, "delay_num") and play_arg.delay_num > 0:
        play_arg.channel_rate = ceil(
            play_arg.sampling_rate / play_arg.delay_num)
    else:
        play_arg.channel_rate = ceil(play_arg.sampling_rate / play_arg.length)

    play_arg.iteration = play_arg.duration * \
        play_arg.sampling_rate // play_arg.length

    if hasattr(play_arg, "delay_num") and play_arg.delay_num > 0:
        play_arg.length += play_arg.delay_num

    # print(dir())
    # wave_set_check = getattr(globals(), "_set_and_check_wave_" + play_arg.wave)
    try:
        print("wave:" + play_arg.wave)
        eval("_set_and_check_wave_" + play_arg.wave)(play_arg)
    except AttributeError:
        raise NotImplementedError("Wave type not implemented yet.")


def _set_and_check_wave_Kasami(play_arg):
    # check frame length is 2^n - 1

    assert (log2(play_arg.frame_length + 1).is_integer())
    play_arg.nbits = log2(play_arg.frame_length + 1)


def _set_and_check_wave_ZC(play_arg):
    if not hasattr(play_arg, "root"):
        setattr(play_arg, "root", (play_arg.frame_length - 1) // 2)
        # play_arg.root = (play_arg.frame_length - 1) // 2


def _set_and_check_wave_chirp(play_arg):
    assert (not hasattr(play_arg, "modulation") or play_arg.modulation == None)
    assert (not hasattr(play_arg, "N_padding") or play_arg.N_padding == None)

    assert (hasattr(play_arg, "f0") and play_arg.f0 > 0)
    assert (hasattr(play_arg, "f1") and play_arg.f1 > 0
            and play_arg.f1 > play_arg.f0
            and play_arg.f1 < play_arg.sampling_rate / 2)


def _set_and_check_wave_sine(play_arg):
    pass


def _set_and_check_wave_Golay(play_arg):
    pass


def set_and_check_process(process_arg, play_arg):

    assert (hasattr(process_arg, "num_topK_subcarriers")
            and process_arg.num_topK_subcarriers > 0)
    assert (hasattr(process_arg, "windows_time")
            and process_arg.windows_time > 0)
    if play_arg.delay_num > 0:
        process_arg.windows_width = round(
            process_arg.windows_time * play_arg.sampling_rate / play_arg.delay_num)
    else:
        process_arg.windows_width = round(
            process_arg.windows_time * play_arg.sampling_rate / play_arg.samples_per_time)
    if (not hasattr(process_arg, "windows_step") or process_arg.windows_step == None):
        process_arg.windows_step = round(process_arg.windows_width / 10)
    if not hasattr(process_arg, "set_preprocess"):
        process_arg.set_preprocess = False

    # Always ensure these attributes are initialized to avoid errors in audio processing
    if not hasattr(process_arg, "demodulation_I_Q"):
        process_arg.demodulation_I_Q = False
    if not hasattr(process_arg, "num_all_channels"):
        process_arg.num_all_channels = 1


def set_and_check_delay(play_arg, process_arg):
    if hasattr(play_arg, "delay_num") and play_arg.delay_num > 0:
        if play_arg.nchannels == 1 and play_arg.orth:
            process_arg.num_all_channels = 2
            process_arg.demodulation_I_Q = True
        elif play_arg.nchannels == 2:
            process_arg.num_all_channels = 2
            process_arg.demodulation_I_Q = False
        elif play_arg.nchannels == 4:
            process_arg.num_all_channels = 4
            process_arg.demodulation_I_Q = True
    else:
        setattr(play_arg, "delay_num", 0)


def set_and_check_param(global_arg, play_arg, device_arg, process_arg):
    set_and_check_wave(play_arg)
    check_channels(play_arg, device_arg)
    if hasattr(device_arg, "input_device") and hasattr(device_arg, "output_device"):
        set_and_check_device(play_arg, device_arg)
    check_record_and_save(global_arg)
    check_load_dataplay(play_arg)
    if hasattr(global_arg, "set_process") and global_arg.set_process:
        set_and_check_delay(play_arg, process_arg)
        set_and_check_process(process_arg, play_arg)
    else:
        global_arg.set_process = False
