from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh, inv, eig
from matplotlib.pylab import Axes


import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch, find_peaks


def detect_peaks(x: np.ndarray, threshold_factor: float = 0.02):
    """Detect peaks in the PSD."""
    threshold = threshold_factor * np.std(x)
    peaks, _ = find_peaks(x, threshold=threshold)
    return peaks


def analyze_timeseries(
    x: np.ndarray, dt: float, compute_psd=True, detect_psd_peaks=False, **welch_kwargs
):
    """
    Analyze time series with FFT and optionally PSD.
    Returns a list of dictionaries, one per column.
    """
    results = []
    for i in range(x.shape[1]):
        ts = x[:, i]
        freq = rfftfreq(len(ts), dt)
        yfft = rfft(ts)
        result = {"time_series": ts, "fft_freq": freq, "fft": yfft}

        if compute_psd:
            f_psd, pxx = welch(ts, 1 / dt, **welch_kwargs)
            result.update({"psd_freq": f_psd, "psd": pxx})
            if detect_psd_peaks:
                peaks = detect_peaks(pxx)
                result["psd_peaks"] = peaks

        results.append(result)

    return results


def plot_timeseries(ax: Axes, x, dt: float = None):
    if dt is None:
        ax.plot(x)
        ax.set_xlabel("t (-)")
    else:
        t = np.arange(len(x)) * dt
        ax.plot(t, x)
        ax.set_xlabel("t (s)")


def plot_fft(ax: Axes, freq, yfft, log_y=False):
    ax.plot(freq, np.abs(yfft))
    ax.set_xlabel("f (Hz)")
    if log_y:
        ax.set_yscale("log")


def plot_psd(ax: Axes, f, pxx, peaks=None, log_y=False):
    ax.plot(f, pxx)
    ax.set_xlabel("f (Hz)")
    if peaks is not None:
        ax.plot(f[peaks], pxx[peaks], "x")
    if log_y:
        ax.set_yscale("log")


def plot_timeseries_analysis(
    results,
    figsize=(8, 16),
    log_y=False,
    titles: list = [],
    max_freq: float = None,
    dt: float = None,
):
    """
    Plot time series, FFT, and optionally PSD (with peaks) from structured results.

    Parameters:
    - results: list of dicts from `analyze_timeseries`
    - figsize: figure size
    - log_y: apply log scale to y-axis for FFT and PSD
    - titles: list of titles for the first row
    - max_freq: restrict plots to frequencies <= max_freq
    """
    ncols = 3 if "psd" in results[0] else 2
    fig, axes = plt.subplots(len(results), ncols, figsize=figsize)

    if len(results) == 1:
        axes = [axes]  # wrap single row

    for i, data in enumerate(results):
        ax = axes[i]

        if i == 0:  # set titles only for the first row
            try:
                for j, t in enumerate(titles):
                    ax[j].set_title(t)
            except:
                pass

        plot_timeseries(ax[0], data["time_series"], dt)

        # Apply max frequency filter if provided
        fft_freq = data["fft_freq"]
        fft_vals = data["fft"]
        if max_freq is not None:
            mask = fft_freq <= max_freq
            fft_freq = fft_freq[mask]
            fft_vals = fft_vals[mask]

        plot_fft(ax[1], fft_freq, fft_vals, log_y=log_y)

        if "psd" in data:
            psd_freq = data["psd_freq"]
            psd_vals = data["psd"]
            if max_freq is not None:
                mask = psd_freq <= max_freq
                psd_freq = psd_freq[mask]
                psd_vals = psd_vals[mask]
                # Adjust peaks if available
                peaks = data.get("psd_peaks")
                if peaks is not None:
                    peaks = [p for p in peaks if data["psd_freq"][p] <= max_freq]
            else:
                peaks = data.get("psd_peaks")

            plot_psd(ax[2], psd_freq, psd_vals, peaks, log_y=log_y)

        for a in ax:
            a.tick_params(labelsize=8)

    return fig, axes


class SSM:
    def __init__(self, M: np.ndarray, C: np.ndarray, K: np.ndarray, B2=None, Ca=None):
        self.n = M.shape[0]
        zeros = np.zeros((self.n, self.n))
        self.Ca = Ca if Ca is not None else np.eye(self.n)
        self.B2 = B2 if B2 is not None else np.eye(self.n)

        self.P = np.block([[C, M], [M, zeros]])
        self.Q = np.block([[K, zeros], [zeros, -M]])

        self.Ac = np.block([[zeros, np.eye(M.shape[0])], [-inv(M) @ K, -inv(M) @ C]])
        self.Bc = np.concatenate([np.zeros_like(self.B2), inv(M) @ self.B2])
        self.Cc = np.block([-self.Ca @ inv(M) @ K, -self.Ca @ inv(M) @ C])
        self.Dc = self.Ca @ inv(M) @ self.B2

        self.eigvals, self.eigvecs = eig(self.Ac)
        self.phi = self.eigvecs[: self.n, : self.n]

    def get_discrete_system(self, dt: float, method="zoh"):
        return signal.cont2discrete((self.Ac, self.Bc, self.Cc, self.Dc), dt, method)

    #
    def plot_phi(self, dof: list[int] | None = None, figsize=(6, 4)):
        """
        Plot mode shapes as points in the complex plane (Im vs Re).
        Each point corresponds to a DOF in one mode shape.

        Parameters:
        - dof: list of DOF indices to include (default: all)
        - figsize: figure size
        """
        if dof is None:
            dof = range(self.n)

        fig, ax = plt.subplots(figsize=figsize)

        for mode_idx in range(self.n):
            # phi_mode = self.phi[:, mode_idx]

            # Select only requested DOFs
            phi_selected = self.phi[dof, mode_idx]
            re = np.real(phi_selected)
            im = np.imag(phi_selected)
            ax.scatter(re, im, label=f"Mode {mode_idx+1}", s=45)
            coeffs = np.polyfit(re, im, 1)  # Linear fit: im = m*r + c
            r_f = np.linspace(min(re), max(re), 100)
            im_f = np.polyval(coeffs, r_f)
            ax.plot(r_f, im_f, "k--", alpha=0.75)

        ax.set_xlabel("Re(ϕ)")
        ax.set_ylabel("Im(ϕ)")
        ax.set_title("Mode shapes in complex plane")
        ax.grid(True)
        ax.legend(fontsize=8)

        return fig, ax

    # def plot_phi(self, dof: int | list[int] | None = None, figsize=(6, 10)):

    #     if dof is None:
    #         dof = range(self.n)
    #     elif isinstance(dof, int):
    #         dof = [dof]

    #     fig, ax = plt.subplots(len(dof), 1, figsize=figsize)
    #     if len(dof) == 1:
    #         ax = [ax]

    #     for i, d in enumerate(dof):
    #         # r = np.real(self.phi[d, :])
    #         # im = np.imag(self.phi[d, :])
    #         r = np.real(self.phi[:, d])
    #         im = np.imag(self.phi[:, d])
    #         ax[i].scatter(r, im, label='Eigenvector')

    #         # Fit and plot a straight line
    #         if len(r) > 1:
    #             coeffs = np.polyfit(r, im, 1)  # Linear fit: im = m*r + c
    #             r_fit = np.linspace(min(r), max(r), 100)
    #             im_fit = np.polyval(coeffs, r_fit)
    #             ax[i].plot(r_fit, im_fit, 'k--',alpha=0.8, label=f'Fit: im = {coeffs[0]:.1f}*r + {coeffs[1]:.1f}')

    #         ax[i].set_title(f"DOF {d}",fontsize=10)
    #         ax[i].set_xlabel("Re(ϕ)")
    #         ax[i].set_ylabel("Im(ϕ)")
    #         ax[i].grid(True)
    #         ax[i].legend()

    #     fig.subplots_adjust(hspace=0.35)
    #     return fig, ax
