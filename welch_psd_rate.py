#!/usr/bin/env python

"""
extract ISBH and ISBD messages from AP_Logging files and produce FFT plots
"""
from __future__ import print_function

import math
import time

import numpy as np
import matplotlib.pyplot as plt
from pymavlink import mavutil
from scipy import signal
from scipy.fft import fft, fftfreq

import tkinter
from tkinter import filedialog
import os

# Open a window to select log file
root = tkinter.Tk()
root.withdraw()
currdir = os.getcwd()
filedir = filedialog.askopenfilename(parent=root, initialdir=currdir, title='Please select a directory')
root.destroy()

class IMUData:
    def __init__(
        self,
        sampling_rate: float,
        time_domain_data: dict,
        instance: int = None,
        imu_type: str = None,
    ) -> None:
        self._sampling_rate = sampling_rate
        self.time_domain_data = time_domain_data
        self.frequency_domain_data = {}
        self.instance = instance
        self.imu_type = imu_type

    def perform_welch_method(
        self,
        overlap_percentage: int = 0,
        segment_percentage: int = 0,
        window_type: str = None,
    ):
        overlap = overlap_percentage / 100
        segment_length = segment_percentage / 100

        for axis in list(self.time_domain_data.keys())[:-1]:
            points_per_segment = int(len(self.time_domain_data[axis]) * segment_length)
            num_overlap_points = int(points_per_segment // (1 / overlap))
            frequencies, spectral_densities = signal.welch(
                x=self.time_domain_data[axis],
                fs=self._sampling_rate,
                nperseg=points_per_segment,
                noverlap=num_overlap_points,
                window=window_type,
            )
            self.frequency_domain_data[axis] = FrequencyDomain(
                list(frequencies), list(spectral_densities)
            )

    def trim_frequency_domain_data(self, cutoff: int):
        for axis, data in self.frequency_domain_data.items():
            trimmed_frequencies = []
            trimmed_spectral_densities = []
            for i in range(0, len(data.frequencies)):
                if data.frequencies[i] > cutoff:
                    trimmed_frequencies.append(data.frequencies[i])
                    trimmed_spectral_densities.append(data.spectral_densities[i])

            self.frequency_domain_data[axis].frequencies = trimmed_frequencies
            self.frequency_domain_data[
                axis
            ].spectral_densities = trimmed_spectral_densities

    def plot_frequency_data(self):
        plt.figure()
        for axis, data in self.frequency_domain_data.items():
            if axis is not "fs":
                plt.plot(data.frequencies, data.spectral_densities, label=axis)
                plt.xlabel("Frequency ($Hz$)")
                plt.ylabel("PSD ($(Degrees / Second)^2 / Hz$)")
                plt.title("Rate Controller Power Spectral Density Estimate")
        plt.legend(loc="upper right")
        plt.show()


class FrequencyDomain:
    def __init__(self, frequencies: list, spectral_densities: list) -> None:
        self.frequencies = frequencies
        self.spectral_densities = spectral_densities

    def __str__(self) -> str:

        return "Frequencies: {}, \nSpectral Densities: {}".format(
            self.frequencies, self.spectral_densities
        )


def trim_fft_data(frequencies: np.ndarray, amplitudes: np.ndarray, cutoff: float):
    trimmed_frequencies = []
    trimmed_amplitudes = []
    for i in range(0, len(frequencies)):
        if frequencies[i] > cutoff:
            trimmed_frequencies.append(frequencies[i])
            trimmed_amplitudes.append(amplitudes[i])

    return trimmed_frequencies, trimmed_amplitudes


def avg_log_sample_rate(data: list, instance: int):
    sum = 0
    counter = 0
    for rate in data[instance]["fs"]:
        counter += 1
        sum += rate
    average = sum / counter
    return average


def read_ardupilot_log(file_name: str):
    print("Reading log file data...")
    mlog = mavutil.mavlink_connection(file_name)
    rate_data = {"rollrate": [], "pitchrate": [], "time": []}
    while True:
        msg = mlog.recv_match()
        if msg is None:
            break
        msg_type = msg.get_type()
        # Check message header for accel or gyro data
        if msg_type == "RATE":
            rate_data["rollrate"].append(msg.R * 180 / math.pi)
            rate_data["pitchrate"].append(msg.P * 180 / math.pi)
            rate_data["time"].append(msg.TimeUS)
    print("Finished reading log file...")
    return rate_data


def plot_fft(frequencies: list, amplitudes: list, N: float, axis: str):
    plt.figure()
    plt.plot(frequencies[1 : N // 2], 2.0 / N * np.abs(amplitudes[1 : N // 2]))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (Degrees / Second)")
    plt.title("Rate Controller FFT")
    plt.legend([axis])


###### User Defined Parameters #####
logfile = filedir
imu_type = "Gyro"
overlap_percent = 50
bin_width_percent = 1
psd_trim_frequency = 10
fft_trim_frequency = 5
window_type = "hann"
####################################

# Read in rate data from log file
rate_data = read_ardupilot_log(logfile)
average_sample_rate = len(rate_data["rollrate"]) / (
    (max(rate_data["time"]) - min(rate_data["time"])) / 1e6
)
print("Average sampling rate: {} Hz".format(average_sample_rate))

# Perform an FFT on the rollrate data
N = len(rate_data["rollrate"])
rollrate_fft = fft(rate_data["rollrate"])
rollrate_freq = fftfreq(N, 1 / average_sample_rate)
rollrate_frequencies, rollrate_amps = trim_fft_data(
    rollrate_freq, rollrate_fft, fft_trim_frequency
)
plot_fft(rollrate_frequencies, rollrate_amps, N, "Rollrate")

# Perform an FFT on the pitchrate data
pitchrate_fft = fft(rate_data["pitchrate"])
pitchrate_freq = fftfreq(N, 1 / average_sample_rate)
pitchrate_frequencies, pitchrate_amps = trim_fft_data(
    pitchrate_freq, pitchrate_fft, fft_trim_frequency
)
plot_fft(pitchrate_frequencies, pitchrate_amps, N, "Pitchrate")

imu_data = IMUData(
    sampling_rate=average_sample_rate, time_domain_data=rate_data, imu_type=imu_type
)
imu_data.perform_welch_method(
    overlap_percentage=overlap_percent,
    segment_percentage=bin_width_percent,
    window_type=window_type,
)

imu_data.trim_frequency_domain_data(psd_trim_frequency)
imu_data.plot_frequency_data()
plt.show()
plt.close('all')
