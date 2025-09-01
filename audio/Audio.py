from loguru import logger
from scipy.io import savemat
import utils
from .Wave import Kasami_sequence
from .Wave import FMCW
import queue
import threading
import soundfile as sf
import numpy as np
import sounddevice as sd
from abc import abstractmethod
import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class Audio:
    def __init__(self,
                 playargs
                 ):
        try:
            logger.info(
                f"Initializing Audio with sampling rate: {playargs.sampling_rate} Hz")
            self.event = threading.Event()
            self._thread = False
            self._sampling_rate = playargs.sampling_rate
            self._blocksize = playargs.blocksize
            self._buffersize = playargs.buffersize
            self._nchannels = playargs.nchannels

            # Validate parameters
            if self._blocksize <= 0:
                logger.warning(
                    f"Invalid blocksize: {self._blocksize}, setting to default 1024")
                self._blocksize = 1024
            if self._buffersize <= 0:
                logger.warning(
                    f"Invalid buffersize: {self._buffersize}, setting to default 20")
                self._buffersize = 20

            # Create buffer queue with appropriate size
            self._q = queue.Queue(maxsize=self._buffersize)  # buffer queue
            self.datarec = np.array([], dtype=np.float32)

            # Log configuration
            logger.info(f"Audio configured with: blocksize={self._blocksize}, "
                        f"buffersize={self._buffersize}, channels={self._nchannels}")

        except Exception as e:
            logger.error(
                f"Error initializing Audio: {type(e).__name__}: {str(e)}")
            raise

    def begin(self):
        try:
            logger.info("Starting audio thread")
            self._thread = True
            threading.Thread(target=self._run).start()
        except Exception as e:
            logger.error(
                f"Error starting audio thread: {type(e).__name__}: {str(e)}")
            self._thread = False
            raise

    def end(self):
        try:
            logger.info("Ending audio thread")
            self._thread = False
            self.event.set()
        except Exception as e:
            logger.error(
                f"Error ending audio thread: {type(e).__name__}: {str(e)}")

    def getData(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            logger.warning("Buffer is empty: increase buffersize")
            print("Buffer is empty: increase buffersize", file=sys.stderr)
            return np.zeros(self._blocksize, dtype=np.float32)

    @abstractmethod
    def _run(self):
        pass

    @abstractmethod
    def _callback(self, indata, outdata, frames, time, status):
        pass

    def _get_buffer(self):
        return self._q.get()

    def get_record(self):
        return self.datarec

    def __str__(self):
        return str(self.__class__.__name__)


class AudioPlayer(Audio):

    def __init__(self,
                 playargs
                 ):
        super(AudioPlayer, self).__init__(
            playargs)
        self.stream = sd.Stream(
            samplerate=self._sampling_rate,
            blocksize=self._blocksize,
            dtype=np.float32,
            callback=self._callback,
            finished_callback=self.event.set,
        )
        dataplay_loader = AcousticDataplayLoader()
        self._data, _ = dataplay_loader(playargs)

        logger.debug("AudioPlayer::data_shape: {}".format(self._data.shape))

    def _run(self):
        try:
            self.stream.start()
            for _ in range(self._buffersize):
                data = self._data[:self._blocksize].astype(np.float32)
                self._q.put_nowait(data)
                self._data = np.roll(self._data, -self._blocksize)
            timeout = self._blocksize * self._buffersize / self._sampling_rate
            while self._thread:
                data = self._data[:self._blocksize].astype(np.float32)
                self._q.put(data, block=True, timeout=timeout)
                self._data = np.roll(self._data, -self._blocksize)
            self.event.wait()

        except queue.Full:
            pass
        except Exception as e:
            print(type(e).__name__ + ': ' + str(e))
        finally:
            logger.info("End")
            self.stream.stop()
            self.stream.close()
            self.end()

    def _callback(self, indata, outdata, frames, time, status):
        try:
            assert frames == self._blocksize
            if status.output_underflow:
                print('Output underflow: increase blocksize', file=sys.stderr)
                logger.warning('Output underflow: increase blocksize')
                raise sd.CallbackAbort
            if status.output_overflow:
                print('Output overflow: increase buffersize', file=sys.stderr)
                logger.warning('Output overflow: increase buffersize')
                raise sd.CallbackAbort
            # Handle input overflow even though we're not recording (for consistency)
            if status.input_overflow:
                print('Input overflow detected', file=sys.stderr)
                logger.warning('Input overflow detected')

            # Handle status more gracefully instead of asserting
            if status:
                # Log the status but continue instead of aborting
                status_str = str(status)
                print(f'Status flags: {status_str}', file=sys.stderr)
                logger.warning(f'Audio callback status flags: {status_str}')

            if not self._thread:
                logger.info("Thread stopped, stopping callback")
                raise sd.CallbackStop

            data = self.getData().reshape(-1, self._nchannels)
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):] = np.zeros(
                    len(outdata) - len(data)).reshape(-1, self._nchannels)
                logger.info("End of data reached, stopping callback")
                raise sd.CallbackStop
            else:
                outdata[:] = data
        except Exception as e:
            logger.error(
                f"Error in audio callback: {type(e).__name__}: {str(e)}")
            # Fill output with zeros to prevent audio glitches
            outdata[:] = np.zeros(
                (len(outdata), self._nchannels), dtype=np.float32)
            # Don't raise the exception to avoid crashes


class AudioPlayandRecord(AudioPlayer):
    def __init__(self,
                 playargs,
                 path):
        super().__init__(playargs)
        self.datarec = np.array([]).reshape(-1, 1)
        self.path = path  # path to save the recorded data
        logger.info(f"Initialized AudioPlayandRecord with path: {path}")
        logger.debug(
            f"Audio parameters: sampling_rate={playargs.sampling_rate}, blocksize={playargs.blocksize}, buffersize={playargs.buffersize}")

    def _run(self):
        try:
            logger.info("Starting audio stream")
            self.stream.start()
            for i in range(self._buffersize):
                try:
                    data = self._data[:self._blocksize].astype(np.float32)
                    self._q.put_nowait(data)
                    self._data = utils.roll(self._data, -self._blocksize)
                    assert ((self._data[-self._blocksize:] == 0).all())
                    logger.debug(
                        f"Filled buffer slot {i+1}/{self._buffersize}")
                except Exception as e:
                    logger.error(
                        f"Error filling buffer: {type(e).__name__}: {str(e)}")

            timeout = self._blocksize * self._buffersize / self._sampling_rate
            logger.info(
                f"Buffer filled, streaming with timeout {timeout:.3f}s")
            while self._thread:
                try:
                    data = self._data[:self._blocksize].astype(np.float32)
                    self._q.put(data, block=True, timeout=timeout)
                    self._data = utils.roll(self._data, -self._blocksize)
                except queue.Full:
                    logger.warning("Queue full - buffer overflow")
                except Exception as e:
                    logger.error(
                        f"Error in streaming loop: {type(e).__name__}: {str(e)}")
            self.event.wait()

        except queue.Full:
            logger.error("Queue full exception - playback may be interrupted")
        except Exception as e:
            logger.error(
                f"Error in audio thread: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            logger.info("Audio thread ending")
            try:
                if self.stream.active:
                    self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream: {str(e)}")
            self.end()

    def _callback(self, indata, outdata, frames, time, status):
        '''
        record simutaneously while playing
        '''
        try:
            assert frames == self._blocksize
            if status.output_underflow:
                print('Output underflow: increase blocksize', file=sys.stderr)
                logger.warning('Output underflow: increase blocksize')
                raise sd.CallbackAbort
            if status.output_overflow:
                print('Output overflow: increase buffersize', file=sys.stderr)
                logger.warning('Output overflow: increase buffersize')
                raise sd.CallbackAbort
            if status.input_overflow:
                print('Input overflow: some recording data has been lost',
                      file=sys.stderr)
                logger.warning(
                    'Input overflow: some recording data has been lost')
                # Don't abort as we want to continue recording even with some data loss

            # Handle status more gracefully
            if status:
                # Log the status but continue instead of aborting
                status_str = str(status)
                print(f'Status flags: {status_str}', file=sys.stderr)
                logger.warning(f'Audio callback status flags: {status_str}')

            if not self._thread:
                logger.info("Thread stopped, stopping callback")
                raise sd.CallbackStop

            data = self.getData().reshape(-1, self._nchannels)
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):] = np.zeros(
                    len(outdata) - len(data)).reshape(-1, self._nchannels)
                logger.info("End of data reached, stopping callback")
                raise sd.CallbackStop
            else:
                outdata[:] = data

            # Record the input data
            if len(indata) > 0:
                self.datarec = np.append(self.datarec, indata.copy())
        except Exception as e:
            logger.error(
                f"Error in audio recording callback: {type(e).__name__}: {str(e)}")
            # Fill output with zeros to prevent audio glitches
            outdata[:] = np.zeros(
                (len(outdata), self._nchannels), dtype=np.float32)

    def get_record(self):
        logger.info(f"Returning recorded data: {self.datarec.shape}")
        return self.datarec

    def save_record(self):
        try:
            logger.info(
                f"Saving recorded data to {self.path}.wav and {self.path}.mat")
            sf.write(self.path + ".wav", self.datarec,
                     self._sampling_rate)  # PCM
            savemat(self.path + ".mat", {"data_rec": self.datarec})
            logger.info(
                f"Saved at {self.path} (samples: {len(self.datarec)}, duration: {len(self.datarec)/self._sampling_rate:.2f}s)")
            return True
        except Exception as e:
            logger.error(
                f"Error saving recording: {type(e).__name__}: {str(e)}")
            return False


class AcousticDataplayLoader():
    def __init__(self) -> None:
        pass

    def _parse_args(self, play_arg):
        if play_arg.wave == "Kasami":
            self._set_Kasami_player(play_arg)
        elif play_arg.wave == "chirp":
            self._set_FMCW_player(play_arg)

    def _set_Kasami_player(self, play_arg):
        self.player = Kasami_sequence(play_arg)

    def _set_FMCW_player(self, play_arg):
        self.player = FMCW(play_arg)

    def __call__(self, play_arg):
        self._dataseq = None
        if play_arg.load_dataplay:
            if hasattr(play_arg, "load_data_seq") and play_arg.load_data_seq:
                dataseq_path = os.path.join(
                    play_arg.dataplay_path, play_arg.dataplay_name[:-4] +
                    "_dataseq.mat"
                )
                try:
                    from scipy.io import loadmat
                    self._dataseq = loadmat(dataseq_path)["data_seq"]
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "File {} not found".format(dataseq_path))

            dataplay_path = os.path.join(
                play_arg.dataplay_path, play_arg.dataplay_name)
            try:
                self._dataplay, _ = sf.read(dataplay_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "File {} not found".format(dataplay_path))
        else:
            self._parse_args(play_arg)
            self._dataplay = self.player()
        self._dataplay = self._dataplay.reshape(-1, play_arg.nchannels)
        return self._dataplay, self._dataseq
