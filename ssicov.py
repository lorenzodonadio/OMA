import numpy as np
import pandas as pd

from numpy.typing import NDArray
import matplotlib.pyplot as plt

from numba import jit, prange
from sklearn.utils.extmath import randomized_svd


@jit(nopython=True, parallel=True)
def blockToeplitz_jit(IRF: NDArray) -> NDArray:
    """Constructs a block Toeplitz matrix from the impulse response function (IRF)."""
    N1 = round(IRF.shape[2] / 2)
    M = IRF.shape[1]
    T1 = np.zeros(((N1) * M, (N1) * M), dtype=np.float32)

    for i in prange(N1):
        for j in prange(N1):
            T1[i * M : (i + 1) * M, j * M : (j + 1) * M] = IRF[:, :, N1 + i - j]

    return T1


def bit_stab_stat(freq_stable: bool, damp_stable: bool, mac_stable: bool):
    """Returns an integer encoding the stability status using bit flags."""
    status = 0
    if freq_stable:
        status |= 1  # bit 0
    if damp_stable:
        status |= 2  # bit 1
    if mac_stable:
        status |= 4  # bit 2
    return status


def stability_check(x: float, x0arr: NDArray, treshold=0.01, is_perc_mode=True):
    """Checks if x is stable with respect to x0arr within a threshold."""
    eps = treshold * x if is_perc_mode else treshold
    if np.min(np.abs(x0arr - x)) <= eps:
        return True
    return False


def mac(x0: NDArray, x1: NDArray):
    """Computes the Modal Assurance Criterion (MAC) between two vectors."""
    x0f, x1f = x0.flatten(), x1.flatten()
    return np.abs(np.dot(x0f, x1f) ** 2 / (np.dot(x0f, x0f) * np.dot(x1f, x1f)))


def mac_stability_check(phi: NDArray, phi0: NDArray, treshold=0.85):
    """Checks if phi is stable with respect to phi0 using MAC threshold."""
    for i in range(phi0.shape[1]):
        if mac(phi, phi0[:, i]) >= treshold:
            return True
    return False


def stability_status(
    fn0,
    zeta0,
    phi0,
    fn1,
    zeta1,
    phi1,
    freq_treshold=0.01,
    damp_treshold=0.05,
    mac_treshold=0.85,
    is_freq_perc=True,
    is_damp_perc=True,
):
    """Returns stability status for each mode between two model orders."""
    frq_s = [stability_check(f, fn0, freq_treshold, is_freq_perc) for f in fn1]
    dmp_s = [
        stability_check(z, zeta0, damp_treshold, is_damp_perc) if z > 0 else False for z in zeta1
    ]
    mac_s = [mac_stability_check(phi1[:, j], phi0, mac_treshold) for j in range(phi1.shape[1])]
    return [bit_stab_stat(frq_s[i], dmp_s[i], mac_s[i]) for i in range(len(frq_s))]


class SSICOV:
    """Implements the SSI-COV algorithm for modal parameter identification."""

    def __init__(self, acc: NDArray, fs: float, time_sample: float, nmin: int, nmax: int):
        """
        Initializes the SSICOV class.

        Parameters
        ----------
        acc : numpy.ndarray
            Acceleration measurements array with shape (samples × channels).
        fs : float
            Sampling frequency in Hz.
        time_sample : float
            Time interval in seconds for impulse response function calculation.
        nmax : int
            Maximum model order (number of states) to consider.
        nmin : int
            Minimum model order (number of states) to consider.
        """
        assert isinstance(acc, np.ndarray), "acc must be a numpy array"
        assert acc.ndim == 2, "acceleration data (acc) must be a 2D array"
        assert fs > 0, "Sampling frequency (fs) must be positive"
        assert time_sample > 0, "Time interval (time_sample) must be positive"
        assert isinstance(nmax, int) and nmax > 0, "nmax must be a positive integer"
        assert isinstance(nmin, int) and nmin > 0, "nmin must be a positive integer"
        assert nmax >= nmin, "nmax must be greater than or equal to nmin"

        self.acc = acc
        self.fs = fs
        self.time_sample = time_sample
        self.n0 = acc.shape[1]
        self.nmax = nmax
        self.nmin = nmin
        self.dt = 1 / self.fs

    def calcIRF(self) -> NDArray:
        """Calculates the impulse response function (IRF) from acceleration data."""
        M = int(self.time_sample / self.dt)
        IRF = np.zeros((self.n0, self.n0, M), dtype=np.float32)
        for i in range(self.n0):
            y1 = np.fft.rfft(self.acc[:, i])
            for j in range(i, self.n0):
                y2 = np.fft.rfft(self.acc[:, j])
                h0 = np.fft.irfft(y1 * y2.conj())
                # impulse response function
                IRF[i, j, :] = np.real(h0[0:M])
                if i != j:
                    IRF[j, i, :] = IRF[i, j, :]
        return IRF

    def _obsMatrix(self, U: NDArray, S: NDArray, model_order: int):
        """Constructs the observability matrix and system matrices A, C."""
        o = U[:, :model_order] * np.sqrt(S[:model_order])
        A = np.linalg.pinv(o[self.n0 :, :]) @ o[: -self.n0, :]
        C = o[: self.n0, :]
        return o, A, C

    def modalParameters(self, U: NDArray, S: NDArray, model_order):
        """Extracts modal parameters (frequencies, damping, mode shapes)."""
        o, A, C = self._obsMatrix(U, S, model_order)
        eigvals, eigvects = np.linalg.eig(A)
        phi: NDArray[np.complex64] = (C @ eigvects)[:, ::2]
        lambda_k = np.log(eigvals[::2]) / self.dt
        resonance_freqs = np.abs(lambda_k) / (2 * np.pi)
        zeta = np.real(lambda_k) / np.abs(lambda_k)

        return resonance_freqs, zeta, phi

    def calc_stability_diagram(self, U: NDArray, S: NDArray, **kwargs):
        """Calculates the stability diagram for a range of model orders.
        kwargs are passed to stability_status :
                        freq_treshold=0.01,
                        damp_treshold=0.05,
                        mac_treshold=0.85,
                        is_freq_perc=True,
                        is_damp_perc=True,
        """
        stability = []

        fn0, zeta0, phi0 = self.modalParameters(U, S, self.nmin)
        stability.append(
            {
                "mo": self.nmin,
                "freqs": fn0,
                "zeta": zeta0,
                "phi": phi0,
                "status": [0] * len(fn0),
            }
        )

        for mo in range(self.nmin + 2, self.nmax + 2, 2):

            fn1, zeta1, phi1 = self.modalParameters(U, S, mo)
            status = stability_status(fn0, zeta0, phi0, fn1, zeta1, phi1, **kwargs)
            stability.append({"mo": mo, "freqs": fn1, "zeta": zeta1, "phi": phi1, "status": status})
            fn0, zeta0, phi0 = fn1, zeta1, phi1

        return pd.DataFrame(stability)

    def run(self, random_svd=False, **kwargs):
        """Runs the SSI-COV algorithm and returns the stability diagram.

        kwargs for stability status:
            freq_treshold=0.01,
            damp_treshold=0.05,
            mac_treshold=0.85,
            is_freq_perc=True,
            is_damp_perc=True,

        returns: pd.DataFrame
        """
        IRF = self.calcIRF()
        T = blockToeplitz_jit(IRF)

        if random_svd:
            rank_value = int(max(30 - 0.00156 * T.shape[0], 25) * T.shape[0] / 100)

            if rank_value < self.nmax:
                rank_value = self.nmax + 1

            U, S, Vh = randomized_svd(T, rank_value)
        else:
            U, S, Vh = np.linalg.svd(T)

        return self.calc_stability_diagram(U, S, **kwargs)


def plot_stabilization_diagram(df, cmap="Accent", figsize=(10, 6)):
    """
    Plots a stabilization diagram (frequency vs model order).

    Parameters:
    - df: DataFrame with columns ['mo', 'freqs', 'status']
    - cmap: Matplotlib colormap name
    """
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, 4))

    fig, ax = plt.subplots(figsize=figsize)
    # plt.figure(figsize=(10, 6))
    for _, row in df.iterrows():
        mo = row["mo"]
        for freq, status in zip(row["freqs"], row["status"]):
            ax.scatter(freq, mo, 90, marker=".", color=colors[0], label="All Poles")
            if status & 1:  # frequency stable
                ax.scatter(
                    freq,
                    mo,
                    100,
                    marker="o",
                    linewidths=3,
                    edgecolors=colors[1],
                    facecolors="none",
                    label="Stable Frequency",
                )
            if status & 2:  # damp stable
                ax.scatter(freq, mo, 100, marker="+", color=colors[2], label="Stable Damping")
            if status & 4:  # MAC stable
                ax.scatter(freq, mo, 86, marker="x", color=colors[3], label="MAC")

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_ylabel("Model Order")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_title("Stabilization Diagram")
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def explode_stability_df(df: pd.DataFrame):
    """
    Expands the stability DataFrame so each row corresponds to a single mode.

    Parameters:
    - df: DataFrame with columns ['mo', 'freqs', 'zeta', 'phi', 'status']

    Returns:
    - Expanded DataFrame with one mode per row.
    """
    try:
        newphicol = [p[:, j] for p in df["phi"] for j in range(p.shape[1])]
        df_sl = df.drop("phi", axis=1).explode(["status", "freqs", "zeta"], ignore_index=True)
        df_sl["phi"] = newphicol
        return df_sl
    except Exception as e:
        print(e)
        return pd.DataFrame([], columns=df.columns)
