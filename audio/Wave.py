from abc import abstractmethod
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import chirp
import sys


class Wave():
    def __init__(self):
        pass

    @property
    def signal(self):
        return self._get_signal()

    @signal.setter
    def signal(self, signal):
        self._signal = signal

    @property
    def time(self):
        return self._get_time()

    @time.setter
    def time(self, time):
        self._time = time

    @property
    def shape(self):
        return self._get_shape()

    def play(self):
        sd.play(self.signal, self.fs)
        sd.wait()

    def save(self, filename):
        sf.write(filename, self.signal, self.fs)

    def plot(self):
        plt.plot(self.time, self._get_signal())
        plt.show()

    @abstractmethod
    def _get_signal(self):
        pass

    @abstractmethod
    def _get_time(self):
        pass

    @abstractmethod
    def _get_shape(self):
        pass

    def __str__(self) -> str:
        return "Wave Name " + str(self.__class__.__name__) + "\n" + \
            "Wave Value " + str(self.signal) + "\n" + \
            "Wave Shape " + str(self.shape)


class Kasami_sequence(Wave):
    def __init__(self, playargs):
        self._parse_args(playargs)

    def _parse_args(self, playargs):
        self._bits = playargs.nbits
        self._channels = playargs.nchannels
        self._shape = playargs.frame_length
        self._iteration = playargs.iteration \
            if playargs.iteration else 80
        self._amplitude = playargs.amplitude \
            if playargs.amplitude else 0.01

    def _get_signal(self):
        return self.__call__()

    def __call__(self):
        self._sequence = self._kasami_generator()
        return self._sequence

    def _get_shape(self):
        return self._sequence.shape

    def _get_time(self):
        return np.arange(self._sequence.shape[0])

    @property
    def bits(self):
        return self._bits

    @property
    def channels(self):
        return self._channels

    def _kasami_generator(self):
        '''
        Kasami sequence generator
        See https://en.wikipedia.org/wiki/Kasami_code
            - mls       : maximum length sequence
            - _bits     : number of bits
            - _channels : number of channels
        Return  : Kasami sequence
        '''
        if self._bits % 2:
            sys.exit('Kasami_generator: nBits must be even for Kasami.')
        if self._bits % 4 == 0:
            sys.exit('Kasami_generator: nBits must not be 4* for Kasami.')
        if self._channels > 2**(self._bits / 2) - 1:
            sys.exit('Kasami_generator: Do not support that much sequences.')

        assert (self._bits % 2 == 0)
        assert (self._bits % 4 != 0)
        assert (self._channels <= 2**(self._bits / 2) - 1)

        mls, _ = signal.max_len_seq(self._bits)
        seq_a = 2 * mls - 1.0  # [0,1] => [-1,1]
        seq_a = seq_a.reshape(-1, 1)  # [2^n-1,] => [2^n-1,1]

        q = (int)(np.power(2, self._bits / 2) + 1)

        def _cyclic_decimation(seq, begin_idx=0, decim_factor=None):
            '''
            Cyclic decimation
            b(n) = a ((q * n) + begin_idx) mod N)
            '''
            N = seq.shape[0]
            idx = np.mod(begin_idx + decim_factor * np.arange(0, N), N)
            return seq[idx, :]

        seq_b = _cyclic_decimation(seq_a, decim_factor=q)
        kasami_sequence = np.array(
            [], dtype=np.float32).reshape([np.size(seq_b), 0])
        for i in range(0, self._channels):
            '''
                kasami_sequence = a(n) + shift(b(n))
            '''
            seq_b_shift = np.roll(seq_b, i, axis=0)
            seq = seq_a * seq_b_shift  # modulo-two arithmetic sum
            kasami_sequence = np.append(kasami_sequence, seq, axis=1)
        kasami_sequence = self._amplitude * kasami_sequence
        kasami = np.tile(kasami_sequence, (self._iteration, 1))
        return kasami


class FMCW(Wave):
    def __init__(self,
                 playargs
                 ):
        self._parse_args(playargs)

    def _parse_args(self, playargs):
        self.f0 = playargs.f0
        self.f1 = playargs.f1
        self.fs = playargs.fs
        self.period = playargs.frame_length
        self.iteration = playargs.iteration
        self.amplitude = playargs.amplitude
        self.zero_length = playargs.idle \
            if playargs.idle else 0

    def __call__(self):
        self._signal = self._fmcw_generator()
        return self._signal

    def plot_f(self):
        plt.plot(self.time, self.freq)
        plt.show()

    def _fmcw_generator(self):
        '''
        start_frequency: f0
        end_frequency: f1
        sampling_frequency: fs
        period : period
        amplitude: amplitude
        zero padding length : zero_length (0 means no padding)
        '''
        output = np.array([], dtype='float32').reshape(-1, 1)
        t = np.arange(0, self.period, 1 / self.fs)
        # t_t = np.array([], dtype='float32')
        for i in range(self.iteration):
            # (self.fs * self.period,)
            data = self.amplitude * \
                chirp(t, self.f0, self.period, self.f1).reshape(-1, 1)
            # print(data.shape)
            if self.zero_length == 0:
                data_zero = np.array([]).reshape(-1, 1)
            else:
                # (self.fs * self.zero_length,)
                data_zero = np.zeros(
                    int(self.fs * self.zero_length)).reshape(-1, 1)
            # (self.period * self.fs + self.zero_length * self.fs)
            output = np.concatenate([output, data, data_zero]).reshape(-1, 1)
        assert (output.shape[0] == self.fs * self.period * self.iteration +
                self.fs * self.zero_length * self.iteration)    # print(output.shape)
        return output

    def _get_frequency(self):
        f_t = np.array([], dtype='float32')
        for i in range(self.iteration):
            t = np.arange(0, self.period, 1 / self.fs)
            f = self.f0 + (self.f1 - self.f0) * t / self.period
            if self.zero_length == 0:
                f_zero = np.array([])
            else:
                f_zero = np.zeros(int(self.fs * self.zero_length))
            f_t = np.concatenate([f_t, f, f_zero])
        return f_t

    @property
    def start_freq(self):
        return self.f0

    @property
    def end_freq(self):
        return self.f1

    @property
    def freq(self):
        return self._get_frequency()

    def _get_shape(self):
        return self.signal.shape

    def _get_signal(self):
        return self._signal

    def _get_time(self):
        if self.zero_length == 0:
            return np.arange(0, self.period * self.iteration, 1 / self.fs)
        else:
            return np.arange(0, self.period * self.iteration + self.zero_length * self.iteration, 1 / self.fs)
