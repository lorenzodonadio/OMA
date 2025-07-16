import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes

from scipy.signal import find_peaks, csd


def estimate_damping_ratios_half_power(anspd, fq, peaks):
    """
    Estimate damping ratios using the half-power method.

    Parameters:
    - anspd: 1D array of amplitude-normalized power spectral density values
    - fq: 1D array of frequency values (same length as anspd)
    - peaks: 1D array of indices of peak locations in anspd

    Returns:
    - damping_ratios: List of tuples [(f0, zeta), ...]
    """
    damping_ratios = []
    freqs = []
    for peak_idx in peaks:
        f0 = fq[peak_idx]
        A0 = anspd[peak_idx]
        A_half = A0 / np.sqrt(2)

        # Search left for the half-power frequency f1
        i1 = peak_idx
        while i1 > 0 and anspd[i1] > A_half:
            i1 -= 1
        if i1 == 0:
            continue  # couldn't find left half-power point
        # Linear interpolation for better accuracy
        f1 = np.interp(A_half, [anspd[i1], anspd[i1 + 1]], [fq[i1], fq[i1 + 1]])

        # Search right for the half-power frequency f2
        i2 = peak_idx
        while i2 < len(anspd) - 1 and anspd[i2] > A_half:
            i2 += 1
        if i2 == len(anspd) - 1:
            continue  # couldn't find right half-power point
        f2 = np.interp(A_half, [anspd[i2 - 1], anspd[i2]], [fq[i2 - 1], fq[i2]])

        # Estimate damping ratio
        damping_ratios.append(((f2 - f1) / (2 * f0)))
        freqs.append(f0)

    return freqs, damping_ratios


def plot_spectral_components(
    data: np.ndarray,
    fq: np.ndarray,
    detected_freqs: None | np.ndarray = None,
    suptitle="Spectral Components: Magnitude, Phase",
    cos_of_phase=False,
):
    _, nrow, ncol = data.shape
    fig, ax = plt.subplots(nrow, ncol, figsize=(14, 10), sharex=True)
    fig.suptitle(suptitle, fontsize=16)

    for i in range(nrow):
        for j in range(ncol):
            axx: Axes = ax[i][j]

            # Primary: Magnitude
            mag = np.abs(data[:, i, j])
            axx.plot(fq, mag, label="|x|", color="tab:blue")
            axx.set_yscale("linear")
            axx.set_ylim(0.99 * np.min(mag), 1.1 * np.max(mag))
            axx.tick_params(labelsize=8)
            axx.set_yscale("log")

            # Add vertical lines for detected frequencies
            if detected_freqs is not None:
                for f in detected_freqs:
                    axx.axvline(f, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
            # Secondary: Phase
            ax_phase = axx.twinx()
            phase = np.angle(data[:, i, j], deg=True)
            if cos_of_phase:
                ax_phase.plot(fq, np.cos(phase), color="tab:orange", alpha=0.6)
                ax_phase.set_ylim(-1, 1)
            else:
                ax_phase.plot(fq, phase, color="tab:orange", alpha=0.6)
                ax_phase.set_ylim(-181, 181)

            # ax_phase.plot(fq, np.cos(phase), label='cos(Phase (°))', color='tab:orange', alpha=0.6)
            # ax_phase.plot(fq, np.cos(phase/(2*np.pi)), label='cos(Phase (°))', color='tab:orange', alpha=0.6)
            # if i == nrow:
            # ax_phase.set_ylim(0,185)

            if j == ncol - 1:
                if cos_of_phase:
                    ax_phase.set_ylabel("Cos(phase) (-)", fontsize=8)
                else:
                    ax_phase.set_ylabel("Phase (°)", fontsize=8)
            if i == nrow - 1:
                axx.set_xlabel("f (Hz)", fontsize=9)
            if j == 0:
                axx.set_ylabel("|x|", fontsize=9)

            ax_phase.tick_params(axis="y", labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, ax


def plot_modes(modes: np.ndarray, append_zero=True, figsize=None):
    if append_zero:
        modes = np.concat([np.zeros((1, modes.shape[1])), modes])
    figsize = figsize or (2 * modes.shape[1], 5)
    fig, axes = plt.subplots(1, modes.shape[1], figsize=figsize)
    yy = range(len(modes[:, 0]))

    for i, ax in enumerate(axes):

        # ax.set_title(f'f :{self.omega_k[i]/(2*np.pi):2.2f}Hz',fontsize = 10)
        ax.plot(np.zeros_like(yy), yy, "k-.", alpha=0.5)
        ax.plot(modes[:, i], yy, "--o")
        if i > 0:
            ax.set_yticks([])
        else:
            ax.set_ylabel("DOF")
        ax.set_xlabel("x [m]")
        ax.tick_params(labelsize=8)
        ax.set_ylim((yy[0], yy[-1]))

    return fig, axes


def get_modes_from_Tref(Tref: np.ndarray, peaks, ref_col=0):
    modeshapes = []
    for p in peaks:
        amp = []
        phase = []
        mode = []
        for dof in range(Tref.shape[2]):
            # Tij = Tref[p,ref_col,dof]
            Tij = Tref[p, dof, ref_col]
            amp.append(np.abs(Tij))
            phase.append(np.angle(Tij))
            if np.cos(phase[-1]) > 0.85:
                mode.append(amp[-1])
            else:
                mode.append(-amp[-1])

            # print(np.abs(Syref))
        modeshapes.append(
            {
                "peakidx": p,
                "amplitudes": np.array(amp),
                "phases": np.array(phase),
                "mode": np.array(mode),
            }
        )
    return modeshapes


# -----------------alejo -----------------

import functools
import inspect
import numpy as np
import time


def print_input_sizes(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Function to get size or shape of the argument
        def get_size(arg):
            if isinstance(arg, np.ndarray):
                return f"shape: {arg.shape}"
            try:
                return f"length: {len(arg)}"
            except TypeError:
                return f"not applicable (type: {type(arg).__name__})"

        # Get the function's argument names
        arg_names = inspect.signature(func).parameters

        # Print the function name
        print(f"Function '{func.__name__}' called with:")

        # Print positional arguments with their names
        for i, (arg_name, arg) in enumerate(zip(arg_names, args)):
            print(f"Argument '{arg_name}' (positional): {get_size(arg)}")

        # Print keyword arguments
        for key, value in kwargs.items():
            print(f"Argument '{key}' (keyword): {get_size(value)}")

        return func(*args, **kwargs)

    return wrapper


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__}, elapsed time: {elapsed_time:.6f} seconds")
        return result

    return wrapper
