from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np


class Butterworth(object):

    @staticmethod
    def _bandpass_(low_cut, high_cut, fs, order=4):
        nyq = 0.5 * fs
        low = low_cut / nyq
        high = high_cut / nyq
        [b, a] = butter(order, [low, high], btype='band')
        return b, a

    @staticmethod
    def _high_(cutoff, fs, order=4):
        nyq = 0.5 * fs
        high = cutoff / nyq
        [b, a] = butter(order, high, btype='highpass')
        return b, a

    @staticmethod
    def _low_(cutoff, fs, order=4):
        nyq = 0.5 * fs
        high = cutoff / nyq
        [b, a] = butter(order, high, btype='lowpass')
        return b, a

    @staticmethod
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = Butterworth._bandpass_(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def butter_high_filter(data, cut, fs, order=4):
        b, a = Butterworth._high_(cut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def butter_low_filter(data, cut, fs, order=4):
        b, a = Butterworth._low_(cut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def butter_low_pass(data, para: list):
        # this is just an unpacking function
        if len(para) == 2:
            return Butterworth.butter_low_filter(data, para[0], para[1])
        if len(para) == 3:
            return Butterworth.butter_low_filter(data, para[0], para[1], para[2])


class OldComplementaryFilter(object):

    def __init__(self, buf: int = 100, rate: float = 1/100, alpha: float = 0.98, debug: bool = False):
        if debug:
            print("ComplementaryFilter")
            print("|-> Requires 10s of static for filter to stabilise")
            print("|-> Only pitch and roll is calculated")
            print("|-> Current Settings:")
            print("|---> Buffer size:\t\t{:d} frames".format(buf))
            print("|---> sampling rate:\t{:.3f}s".format(rate))
            print("|---> alpha:\t\t\t{:.3f}".format(alpha))
        self.gyro_buffer = []
        self.acc_buffer = []
        self.buffer_size = buf
        self.ori_buffer = np.zeros([buf, 3])
        self.rate = rate
        self.alpha = alpha

    def update(self, gyro, acc):
        self.gyro_buffer.append(gyro)
        self.acc_buffer.append(acc)
        if len(self.gyro_buffer) > self.buffer_size:
            self.gyro_buffer.pop(0)
        if len(self.acc_buffer) > self.buffer_size:
            self.acc_buffer.pop(0)
        self.__calculate_orientation__()
        return self.ori_buffer[-1, :]

    def __calculate_orientation__(self):
        acc = np.mean(np.array(self.acc_buffer), axis=0)
        gyro = np.mean(np.array(self.gyro_buffer), axis=0)
        prev_ori = np.mean(self.ori_buffer, axis=0)
        pitch_a = np.arctan2(acc[1], acc[2])
        pitch_b = prev_ori[0] + self.rate * gyro[0]

        roll_a = np.arctan2(acc[0], acc[2])
        roll_b = prev_ori[1] + self.rate * gyro[1]

        pitch = self.alpha * pitch_b + (1 - self.alpha) * pitch_a
        roll = self.alpha * roll_b + (1 - self.alpha) * roll_a
        tilt = np.array([[0, pitch, roll]])
        self.ori_buffer = np.append(self.ori_buffer, tilt, axis=0)

    def get_acc(self):
        return np.mean(self.acc_buffer, axis=0)


class ComplementaryFilter(object):
    """
    This is a time constant filter
    """
    def __init__(self, rate: float = 1/100, alpha: float = 0.98):
        self.rate = rate
        self.alpha = alpha
        self.ori_buffer = [0, 0, 0]
        self.magnetic_declination = 19.91*(np.pi/180)  # magnetic declination in Auckland, New Zealand

    def update(self, gyro, acc, mag=None):
        prev_ori = self.ori_buffer
        pitch_a = np.arctan2(-acc[0], np.sqrt(acc[2]**2+acc[1]**2))
        pitch_b = prev_ori[1] + self.rate * gyro[0]

        roll_a = np.arctan2(acc[1], np.sqrt(acc[2] ** 2 + acc[0] ** 2))
        roll_b = prev_ori[0] + self.rate * gyro[1]

        pitch = self.alpha * pitch_b + (1 - self.alpha) * pitch_a
        roll = self.alpha * roll_b + (1 - self.alpha) * roll_a
        if mag is None:
            self.ori_buffer = np.array([roll, pitch, 0])
        else:
            yaw_a = self.heading(mag)
            yaw_b = prev_ori[2] + self.rate * gyro[2]
            yaw = self.alpha * yaw_b + (1 - self.alpha) * yaw_a
            self.ori_buffer = np.array([roll, pitch, yaw])
        return self.ori_buffer

    def heading(self, m):
        yaw = np.arctan2(-m[0], m[1]) + self.magnetic_declination
        return yaw

    @property
    def ori(self):
        return [self.ori_buffer[0], -self.ori_buffer[1], self.ori_buffer[2]]


def process_data(data: pd.DataFrame, func, para):
    # Assumes get_first frame is time, frame or index
    # and data have the shape 0: frame/time/index, 1: data channel
    to_filter = data.to_numpy()
    for i in range(1, to_filter.shape[1]):
        d = to_filter[:, i]
        to_filter[:, i] = func(np.squeeze(d), para)
    return pd.DataFrame(data=to_filter, columns=data.columns)
