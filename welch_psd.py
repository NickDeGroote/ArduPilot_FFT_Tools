#!/usr/bin/env python

"""
extract ISBH and ISBD messages from AP_Logging files and produce FFT plots
"""
from __future__ import print_function

import sys

import matplotlib.pyplot as plt
from pymavlink import mavutil
from scipy import signal


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

        for axis in self.time_domain_data.keys():
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
        for axis, data in self.frequency_domain_data.items():
            if axis is not "fs":
                plt.plot(data.frequencies, data.spectral_densities, label=axis)
                plt.xlabel("Frequency ($Hz$)")
                plt.ylabel("PSD ($g^2 / Hz$)")
                plt.title(
                    "Power Spectral Density Estimate ({}) (IMU {})".format(
                        self.imu_type, self.instance
                    )
                )
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
    imu_type = None
    instance = None
    sample_rate = None
    mlog = mavutil.mavlink_connection(file_name)
    accel_data = [
        {"X": [], "Y": [], "Z": [], "fs": []},
        {"X": [], "Y": [], "Z": [], "fs": []},
        {"X": [], "Y": [], "Z": [], "fs": []},
    ]
    gyro_data = [
        {"X": [], "Y": [], "Z": [], "fs": []},
        {"X": [], "Y": [], "Z": [], "fs": []},
        {"X": [], "Y": [], "Z": [], "fs": []},
    ]
    while True:
        msg = mlog.recv_match()
        if msg is None:
            break
        msg_type = msg.get_type()
        # Check message header for accel or gyro data
        if msg_type == "ISBH":
            instance = msg.instance
            sample_rate = msg.smp_rate
            if msg.type == 0:
                imu_type = "Accelerometer"
            elif msg.type == 1:
                imu_type = "Gyro"
            else:
                print("Invalid IMU data type!!!")
        # Actual IMU data
        if msg_type == "ISBD":
            if imu_type == "Accelerometer":
                accel_data[instance]["X"].extend(msg.x)
                accel_data[instance]["Y"].extend(msg.y)
                accel_data[instance]["Z"].extend(msg.z)
                accel_data[instance]["fs"].append(sample_rate)
            elif imu_type == "Gyro":
                gyro_data[instance]["X"].extend(msg.x)
                gyro_data[instance]["Y"].extend(msg.y)
                gyro_data[instance]["Z"].extend(msg.z)
                gyro_data[instance]["fs"].append(sample_rate)
            else:
                print("IMU type was neither accelerometer or gyro")
    print("Finished reading log file...")
    return accel_data, gyro_data


###### User Defined Parameters #####
logfile = "00000016.BIN"
imu_number = 0
imu_type = "Gyro"
overlap_percent = 50
bin_width_percent = 1
trim_frequency = 5
window_type = "hann"
####################################

accel_data, gyro_data = read_ardupilot_log(logfile)
average_sample_rate = avg_log_sample_rate(accel_data, imu_number)

print("Average sampling rate: {} Hz".format(average_sample_rate))

imu_data = IMUData(
    sampling_rate=average_sample_rate,
    time_domain_data=accel_data[imu_number],
    imu_type=imu_type,
)
imu_data.perform_welch_method(
    overlap_percentage=overlap_percent,
    segment_percentage=bin_width_percent,
    window_type=window_type,
)
imu_data.trim_frequency_domain_data(trim_frequency)
imu_data.plot_frequency_data()
