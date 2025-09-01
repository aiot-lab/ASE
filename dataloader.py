import numpy as np
import os
from loguru import logger
from scipy.io import loadmat
import soundfile as sf
from audio.Audio import AcousticDataplayLoader


class Dataset():
    '''Dataset Class
    @description:           This is a base class for dataset
    @param global_arg:      global config arguments
    @param file_type:       file type, e.g. "mat", "wav"
    @attribute _data_dir:   data path plus file type
    @Note:                  Need to implement __getitem__ and _get_data method 
                            by inheriting this class
    '''

    def __init__(self, global_arg, file_type) -> None:
        self._file_type = file_type
        self._data_dir = global_arg.data_path + "." + self._file_type
        self._dataset = self._get_data()
        self.rec_idx = global_arg.rec_idx

    @property
    def dataset(self):
        return self._dataset

    @property
    def data_dir(self):
        return self._data_dir

    # Important: You have to implement this method by inheriting this class
    def __getitem__(self, index):
        raise NotImplementedError

    def _get_data(self):
        raise NotImplementedError

    @property
    def shape(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.rec_idx)

    def __eq__(self, o: object) -> bool:
        return self.rec_idx == o.rec_idx

    def __str__(self) -> str:
        return ">" * 50 + "Dataset Info" + "<" * 50 + "\n" + \
            "Dataset Type: " + str(self.__class__.__name__) + "\n" + \
            "Dataset Shape: {}".format(self.shape)


class AcousticDataset(Dataset):
    '''Acoustic Dataset Class
    @description:                   Class for acoustic dataset
    @attribute: __getitem__()       return the data of the index (1-d or 2-d)
    @attribute: _get_data()         load the data from the data path (mat or wav)
    '''

    def __init__(self, global_arg, file_type, data_path=None) -> None:
        super().__init__(global_arg, file_type)
        if self._dataset.shape[0] == 1:
            self._dataset = np.transpose(self._dataset)

    def __getitem__(self, index):
        try:
            if len(self.shape) == 1:
                return self.dataset[index]
            elif len(self.shape) == 2:
                return self.dataset[index[0], index[1]]
        except IndexError:
            logger.error("Index out of range")

    def _get_data(self):
        try:
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError
            if self._file_type == "mat":
                dataset = loadmat(self.data_dir)['data_rec']
            elif self._file_type == "wav":
                dataset, _ = sf.read(self.data_dir)
            else:
                raise ValueError("File type not supported")
            return dataset
        except Exception as e:
            print(e.__class__.__name__ + ":" + "Can not find the dataset")
            exit(e)

    @property
    def shape(self):
        return self.dataset.shape


class DataLoader():
    def __init__(self, dataset) -> None:
        self._dataset = dataset

    def __iter__(self):
        raise NotImplementedError

    @property
    def dataset(self):
        return self._dataset


class AcousticDataloader(DataLoader):
    def __init__(self,
                 dataset,
                 play_arg,
                 rec_idx=None) -> None:
        super().__init__(dataset)
        # self._datarec_size = self._dataset.shape
        # self._preprocess = preprocess
        # self._align_thres = align_thres
        # self.load_dataplay = load_dataplay
        dataplay_loader = AcousticDataplayLoader()
        self._dataplay, self._dataseq = dataplay_loader(play_arg)
        # self._dataplay = self._dataplay[:play_arg.samples_per_time, :]

        self._dataplay_size = self._dataplay.shape
        self._datarec_size = self._dataset.shape
        self._dataseq_size = self._dataseq.shape

        self._channels = self._datarec_size[1]

        self.rec_idx = rec_idx

    # def __str__(self) -> str:
    #     return ">" * 50 + "Dataloader Info" + "<" * 50 + "\n" + \
    #         "Dataset Shape: {}".format(self._datarec_size) + "\n" + \
    #         "Dataplay Shape: {}".format(self._dataplay_size) + "\n"

    def __hash__(self):
        return hash(self.rec_idx)

    def __eq__(self, o: object) -> bool:
        return self.rec_idx == o.rec_idx

    def __iter__(self):
        logger.info("datarec size: " + str(self._datarec_size))
        logger.info("dataseq size: " + str(self._dataseq_size))
        num_of_frames = self._datarec_size[0] // self._dataseq_size[0]
        # print(">" * 50 + "Loading data" + "<" * 50)
        logger.info("Loading data...")
        logger.info("num_of_frames: " + str(num_of_frames))
        # todo: add microphone channels
        # if self._align:
        #     for channel_idx in range(0, self._channels):
        #         start = self._lag
        #         logger.info("Loading channel " + str(channel_idx))
        #         while start < self._dataplay_size[0] and start + self._samples_per_time < self._dataplay_size[0]:
        #             datarec = self._dataset[start: start +
        #                                     self._samples_per_time, 0]
        #             dataplay = self._dataplay[start: start +
        #                                       self._samples_per_time, channel_idx]
        #             # logger.debug("start: {}".format(start))
        #             # logger.debug("self._datarec_size[0]: {}".format(self._dataplay_size[0]))
        #             # logger.debug("self._samples_per_time: {}".format(self._samples_per_time))
        #             # logger.debug("dataplay shape: {}".format(dataplay.shape))
        #             # logger.debug("datarec shape: {}".format(datarec.shape))
        #             start += 2 * self._samples_per_time
        #             yield datarec, dataplay, channel_idx
        # else:
        for channel_idx in range(self._channels):
            yield self._dataset[:, channel_idx], self._dataseq[:, channel_idx], channel_idx

            # datarec = self._dataset[num_of_frames * self._dataplay_size[0]:, 0]
            # dataplay = self._dataplay[:self._datarec_size[0] - num_of_frames * self._dataplay_size[0], channel_idx]
            # yield datarec, dataplay, channel_idx

    def update(self, datarec=None, dataseq=None):
        if datarec is not None:
            self._dataset = datarec
            self._datarec_size = self._dataset.shape
            self._channels = self._datarec_size[1]
        if dataseq is not None:
            self._dataseq = dataseq
            self._dataseq_size = self._dataseq.shape

    @property
    def dataplay(self):
        return self._dataplay

    @property
    def dataseq(self):
        return self._dataseq

    @property
    def datarec(self):
        return self._dataset
