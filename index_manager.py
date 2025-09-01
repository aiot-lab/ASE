import os


class IndexManager():
    '''
    Universe Indexing for saving
    @ format: task_wave_input_device_output_device_time_rec_idx
    @ rec_idx: unique index for each record
    @ Note: contain folder of the same name
    @ Note: save_type is not determined by IndexManager
    '''
    INDEX_ITEM = ["task", "wave", "input_device",
                  "output_device", "time", "rec_idx"]
    idx = 0
    name = ""

    def __init__(self, global_arg, play_arg, device_arg) -> None:

        self.save_root = global_arg.save_root
        # try:
        #     self.set_save = global_arg.set_save
        # except AttributeError:
        #     self.set_save = False

        self._check_save_root()
        if not hasattr(global_arg, "rec_idx"):
            self.rec_idx = self._get_rec_idx()
            import time
            name_kwargs = {
                "task": global_arg.task,
                "wave": play_arg.wave,
                "input_device": device_arg.input_device,
                "output_device": device_arg.output_device,
                "time": time.strftime("%Y%m%d-%H%M%S"),
                "rec_idx": self.rec_idx
            }
            self.name = self._construct_save_name(**name_kwargs)
        else:
            self.rec_idx = global_arg.rec_idx
            self.name = self._find_rec_name()

    def __call__(self):
        return self.name, self.rec_idx

    def _check_save_root(self):
        if not os.path.exists(self.save_root):
            os.mkdir(self.save_root)

    def _find_rec_name(self):
        try:
            # change to function programming
            dirs = [dirs for _, dirs, _ in os.walk(self.save_root)][0]
            name = [dir
                    for dir in dirs if dir is not None
                    and (dir.split("_")[-1]).isnumeric()
                    and int(dir.split("_")[-1]) == self.rec_idx][0]

        finally:
            return name

    def _get_rec_idx(self):
        idx = 0
        try:
            dirs = [dirs for _, dirs, _ in os.walk(self.save_root)][0]
            _idx = max([int(dir.split("_")[-1])
                        for dir in dirs if dir is not None
                        and (dir.split("_")[-1]).isnumeric()]) + 1
            idx = max(_idx, idx)

        finally:
            return idx

    def _construct_save_name(self, **name_kwargs):
        name = ""
        for item in self.INDEX_ITEM:
            name += str(name_kwargs[item]) + "_"
        name = name[:-1]
        return name


def index_config(global_arg, play_arg, device_arg):
    index_manager = IndexManager(global_arg=global_arg,
                                 play_arg=play_arg,
                                 device_arg=device_arg)
    data_name, rec_idx = index_manager()

    if not hasattr(global_arg, "rec_idx"):
        setattr(global_arg, "offer_rec_idx", False)
    else:
        setattr(global_arg, "offer_rec_idx", True)

    global_arg.rec_idx = rec_idx if not hasattr(global_arg, "rec_idx") \
        else global_arg.rec_idx

    global_arg.data_name = data_name
    global_arg.data_folder = os.path.join(
        global_arg.save_root, global_arg.data_name)
    if not os.path.exists(global_arg.data_folder):
        os.mkdir(os.path.join(global_arg.data_folder))
    global_arg.data_path = os.path.join(
        global_arg.data_folder, global_arg.data_name)

    return global_arg, data_name, rec_idx
