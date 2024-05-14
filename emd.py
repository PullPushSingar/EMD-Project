from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import numpy as np

def emd(signal, fs, num_iterations: int = None, draw_plot: bool = False):
    '''
    Takes signal and sampling frequency
    and returns an array of imfs and a residue signal at the end of it

    Additional arguments:
    - `num_iterations` - how many imfs should the function create.
    If not specified then looping as long as the signal can be interpolated
    - `draw_plot` - a boolean specifying if the function should draw the plot
    (default: False)
    '''
    # Define parameters for the signal composition
    num_samples = len(signal)
    num_seconds = num_samples / fs
    tAxis = np.linspace(0, num_seconds, num_samples) # in seconds
    orig_x = signal
    imfs = []

    # Iterate through the signal as long as it is possible to interpolate it
    # Or for specified number of times
    while num_iterations is None or num_iterations > 0:
        # Find the peaks of the signal
        upper_peaks, _ = find_peaks(signal)
        lower_peaks, _ = find_peaks(-signal)

        try:
            # Interpolate between the peaks
            f1 = interp1d(upper_peaks/fs,signal[upper_peaks], kind = 'cubic', fill_value = 'extrapolate')
            f2 = interp1d(lower_peaks/fs,signal[lower_peaks], kind = 'cubic', fill_value = 'extrapolate')
            y1 = f1(tAxis)
            y2 = f2(tAxis)
        except:
            # Breaks out of the loop if interpolation cannot be done anymore
            break

        avg_envelope = (y1 + y2) / 2

        # Append imf to the array
        imfs.append(signal - avg_envelope)
        signal = avg_envelope

        if num_iterations is not None:
            num_iterations -= 1

    # Add residue to the final list
    imfs.append(signal)

    # Draw a plot if draw_plot argument is True
    if draw_plot:
        plt.figure()
        plt.subplot(len(imfs)+2,1,1)
        plt.plot(tAxis, orig_x)
        plt.title("Original signal")

        for i, imf in enumerate(imfs):
            plt.subplot(len(imfs)+2,1,i+2)
            plt.plot(tAxis,imf)
            plt.title(f"IMF {i+1}")

        plt.subplot(len(imfs)+2,1,len(imfs)+1)
        plt.plot(tAxis, signal)
        plt.title("Residue")

        # Plot sum of all IMFs for check
        sum_imfs = np.sum(imfs, axis=0)
        plt.subplot(len(imfs) + 2, 1, len(imfs) + 2)
        plt.plot(tAxis, sum_imfs)
        plt.title("Sum of all IMFs")
        plt.show()

    # Return the array of the imfs and residue
    return imfs, signal