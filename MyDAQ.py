import nidaqmx as dx
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq , rfft, rfftfreq
from scipy.signal import square



class MyDAQ():
    def __init__(self, device_name='myDAQ1'):
        self.device_name = device_name
        self.write_task = None
        self.read_task = None
        self.rate = 500
        self.samps_per_chan = 50

    def configure_write_task(self, rate=500, samps_per_chan=2500):
        self.write_task = dx.Task('AOTask')
        self.write_task.ao_channels.add_ao_voltage_chan(f'{self.device_name}/ao0')
        self.write_task.timing.cfg_samp_clk_timing(
            rate,
            sample_mode = dx.constants.AcquisitionType.FINITE,
            samps_per_chan=samps_per_chan
        )
        self.samps_per_chan = samps_per_chan
        self.rate= rate
        
    def write(self, data):
        if self.write_task is None:
            raise Exception("Write task is not configured.")
        
        self.write_task.write(data, auto_start=True)
        time.sleep(self.samps_per_chan / self.rate + 0.001)
        self.write_task.stop()
    
    def configure_read_task(self, rate=100, samps_per_chan=50):
        
        """Configures the read task with the given parameters.
            Rate = how many per seconds
            Samps_per_chan = number of data points
        
        """
        self.read_task = dx.Task(dx.Task('AITask'))
        self.read_task.ai_channels.add_ai_voltage_chan(f'{self.device_name}/ai0')
        self.read_task.timing.cfg_samp_clk_timing(
            rate,
            sample_mode=dx.constants.AcquisitionType.FINITE,
            samps_per_chan=samps_per_chan
        )
        self.samps_per_chan = samps_per_chan
        self.rate = rate

    def read(self):
        """Reads data from the MyDAQ input."""
        if self.read_task is None:
            raise Exception("Read task is not configured.")
        
        data = self.read_task.read(number_of_samples_per_channel=self.samps_per_chan)
        return data

    def read_write(self, data, rate=100, samps_per_chan= 50, read_channel='ai0' , write_channel= 'ao0'):
        self.read_task = dx.Task('AITask')
        self.write_task = dx.Task('AOTask')
        self.read_task.ai_channels.add_ai_voltage_chan(f'myDAQ1/{read_channel}')
        self.write_task.ao_channels.add_ao_voltage_chan(f'myDAQ1/{write_channel}')

        self.write_task.timing.cfg_samp_clk_timing(
            rate,
            sample_mode = dx.constants.AcquisitionType.FINITE,
            samps_per_chan=samps_per_chan
        )


        self.read_task.timing.cfg_samp_clk_timing(
            rate,
            sample_mode=dx.constants.AcquisitionType.FINITE,
            samps_per_chan=samps_per_chan
        )

        self.write_task.write(data, auto_start=True)

        data = self.read_task.read(number_of_samples_per_channel=self.samps_per_chan)

        self.write_task.stop()

        return data

    def generate_sine_wave(self, frequency=10, amplitude=5, phase_shift=0, offset=0):
        """Generates a sine wave with the specified parameters."""
        x = self.get_time_array()
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * x + phase_shift) + offset
        return sine_wave
    
    def get_time_array(self,include_last=False):
        if not include_last:
            return np.linspace(0, self.samps_per_chan / self.rate, num=self.samps_per_chan)
        else:
            return np.linspace(0, self.samps_per_chan / self.rate, num=self.samps_per_chan-1)

    def plot_data(self, data, output_path='output.pdf'):
        """Plots the data and saves the figure to a file."""
        x = self.get_time_array()
        plt.figure()
        plt.scatter(x, data)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('MyDAQ Reading test from Function Generator')
        plt.savefig(output_path, format='pdf')
        plt.show()

    def close(self):
        """Closes any open tasks."""
        if self.write_task is not None:
            self.write_task.close()
        if self.read_task is not None:
            self.read_task.close()

    def fft(self, data):
        """Performs FFT on the data and plots the frequency spectrum."""
        N = len(data)
        fft_values = fft(data)
        freqs = fftfreq(N, 1/self.rate)
        positive_freqs = freqs[:N//2]
        fft_magnitudes = np.abs(fft_values[:N//2])

        # Plot the frequency spectrum
        plt.figure()
        plt.plot(positive_freqs, fft_magnitudes)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('FFT Frequency Spectrum')
        plt.show()

        return positive_freqs, fft_magnitudes
    
    def rfft(self, data):
        """Performs FFT on the data and plots the frequency spectrum."""
        N = len(data)
        fft_values = rfft(data)
        freqs = rfftfreq(N, 1/self.rate)
        fft_magnitudes = np.abs(fft_values)

        # Plot the frequency spectrum
        plt.figure()
        plt.scatter(freqs, fft_magnitudes)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('FFT Frequency Spectrum')
        plt.show()

        return freqs, fft_magnitudes



# Example usage:
daq = MyDAQ()

# Configure for reading
signal_data = daq.generate_sine_wave()




#$data = daq.read_write(signal_data)

daq.plot_data(signal_data)

daq.fft(signal_data)
daq.rfft(signal_data)


#daq.plot_data(signal_data)
#data = daq.read_write(signal_data)
#data = daq.read()
#print(data)
#daq.plot_data(data)
# Perform FFT on the data
#daq.fft(signal_data)

daq.close()
print('poop')