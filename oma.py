import numpy as np
import matplotlib.pylab as plt
from numpy.linalg import eigh, inv, eig
from matplotlib.pylab import Axes


def plot_response_magnitude(
    H: np.ndarray | list[np.ndarray],
    freq: np.ndarray,
    vertlines: None | np.ndarray = None,
    figsize=(8, 8),
):

    if isinstance(H, list):
        assert all([H[0].shape == h.shape for h in H])
        fdim, rows, cols = H[0].shape
        assert cols == rows
        assert fdim == len(freq)

        h_mag = [np.abs(h) for h in H]

        hmin = 0.95 * np.min(h_mag)
        hmax = 1.05 * np.max(h_mag)
    else:
        fdim, rows, cols = H.shape
        assert cols == rows
        assert fdim == len(freq)

        h_mag = np.abs(H)
        hmin = 0.95 * np.min(h_mag)
        hmax = 1.05 * np.max(h_mag)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i in range(cols):
        for j in range(rows):
            ax: Axes = axes[i][j]
            ax.set_ylim(hmin, hmax)
            if vertlines is not None:
                for f in vertlines:
                    ax.plot([f, f], [hmin, hmax], c="#FF9008", linestyle="-.", alpha=0.5)

            if isinstance(H, list):
                for h in h_mag:
                    ax.plot(freq, h[:, i, j])
            else:
                ax.plot(freq, h_mag[:, i, j])

            ax.set_yscale("log")
            ax.set_title(f"H {i+1}{j+1}", fontsize=9)
            ax.grid(True, linestyle="--")

            if j == 0:
                ax.set_ylabel("m/N")

            if i == cols - 1:
                ax.set_xlabel("f(Hz)")

            ax.tick_params(labelsize=8)

    fig.subplots_adjust(hspace=0.3)
    return fig, axes


def plot_response_phase(
    H: np.ndarray | list[np.ndarray],
    freq: np.ndarray,
    vertlines: None | np.ndarray = None,
    figsize=(8, 8),
):

    ymin, ymax = -185, 185

    if isinstance(H, list):
        assert all([H[0].shape == h.shape for h in H])
        fdim, rows, cols = H[0].shape
        assert cols == rows
        assert fdim == len(freq)

        phase = [np.angle(h, deg=True) for h in H]

    else:
        fdim, rows, cols = H.shape
        assert cols == rows
        assert fdim == len(freq)

        phase = np.angle(H, deg=True)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            ax: Axes = axes[i][j]
            ax.set_ylim(ymin, ymax)

            if vertlines is not None:
                for f in vertlines:
                    ax.plot([f, f], [ymin, ymax], c="#FF9008", linestyle="-.", alpha=0.5)

            if isinstance(H, list):
                for h in phase:
                    ax.plot(freq, h[:, i, j])
            else:
                ax.plot(freq, phase[:, i, j])

            ax.set_title(f"Phase H {i+1}{j+1}", fontsize=9)
            ax.grid(True, linestyle="--")

            if j == 0:
                ax.set_ylabel("deg")

            if i == cols - 1:
                ax.set_xlabel("f(Hz)")

            ax.tick_params(labelsize=8)

    fig.subplots_adjust(hspace=0.3)
    return fig, axes


def plot_white_noise_response(
    sq: np.ndarray,
    freq: np.ndarray,
    figsize=(8, 8),
):
    if len(sq.shape) == 4:
        kk, fdim, rows, cols = sq.shape
    elif len(sq.shape) == 3:
        kk = 0
        fdim, rows, cols = sq.shape
    else:
        raise ValueError("sq must be a 3d or 4d array")

    assert fdim == len(freq)
    sqmag = np.abs(sq)
    ymin = 0.95 * np.min(sqmag)
    ymax = 1.05 * np.max(sqmag)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            ax: Axes = axes[i][j]
            ax.set_ylim(ymin, ymax)
            if kk > 0:
                for k in range(kk):
                    ax.plot(freq, sqmag[k, :, i, j])
            else:
                ax.plot(freq, sqmag[:, i, j])

            ax.grid(True, linestyle="--")
            ax.set_yscale("log")
            ax.tick_params(labelsize=8)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    return fig, ax


class OperationalModalAnalysis:
    def __init__(
        self,
        nf=4,
        m: float | np.ndarray = 150,
        k: float | np.ndarray = 30000,
        zeta=0.01,
        solve_eig_init=True,
    ):
        # model parameters
        self.nf = nf  # number of floors
        self.m = m  # kg
        self.k = k  # N/m
        self.zeta = zeta

        # set up matrices
        # this works for either float or ndarray
        self.M = m * np.eye(nf)  # mass matrix
        # stiffness matrix
        if isinstance(k, np.ndarray):
            self.K = np.diag(k + np.append(k[1:], 0)) - np.diag(k[1:], k=-1) - np.diag(k[1:], k=1)
        else:
            self.K = (
                2 * k * np.eye(nf)
                - k * np.diag(np.ones(nf - 1), k=-1)
                - k * np.diag(np.ones(nf - 1), k=1)
            )
            self.K[-1, -1] = k

        c = zeta * 2 * m * np.sqrt(k / m)

        # Damp factor matrix
        if isinstance(c, np.ndarray):
            self.C = np.diag(c + np.append(c[1:], 0)) - np.diag(c[1:], k=-1) - np.diag(c[1:], k=1)
        else:
            self.C = (
                2 * c * np.eye(nf)
                - c * np.diag(np.ones(nf - 1), k=-1)
                - c * np.diag(np.ones(nf - 1), k=1)
            )
            self.C[-1, -1] = c

        self.sdof_freq = np.sqrt(np.sum(self.k) / np.sum(self.m)) / (2 * np.pi)

        if solve_eig_init:
            self.solve_eig()

    def solve_eig(self):
        # Solve the generalized eigenvalue problem
        # self.eigvals, self.eigvecs = eigh(inv(self.M) @ self.K)
        eigvals, eigvecs = eig(inv(self.M) @ self.K)
        idx = np.argsort(eigvals)
        self.eigvals, self.eigvecs = eigvals[idx], eigvecs[:, idx]

        # Natural frequencies (rad/s)
        self.omega_k = np.sqrt(self.eigvals)
        # Convert to Hz
        self.f_k = self.omega_k / (2 * np.pi)
        print("1 DOF frequency (Hz):", self.sdof_freq)
        print("Natural frequencies (Hz):", self.f_k)

    def response_H_inv(self, w: float | int | np.ndarray):
        try:
            if isinstance(w, np.ndarray):
                return np.array([inv(-(ww**2) * self.M + 1.0j * ww * self.C + self.K) for ww in w])
            else:
                return inv(-(w**2) * self.M + 1.0j * w * self.C + self.K)
        except Exception as e:
            raise (e)

    def response_H_explicit(self, w: float | int | np.ndarray, ret_split=False):
        try:
            if not isinstance(w, np.ndarray):
                w = float(w)
            if ret_split:
                H = np.zeros((len(self.omega_k), len(w), self.nf, self.nf), dtype=np.complex128)
            else:
                H = np.zeros((len(w), self.nf, self.nf), dtype=np.complex128)
            # for wi,w in enumerate(ww):
            for i in range(self.nf):
                for j in range(self.nf):
                    for k, eigval in enumerate(self.eigvals):
                        phi = self.eigvecs[:, k]
                        num = phi[i] * phi[j]
                        den = eigval - w**2 + 2 * 1.0j * self.zeta * w * self.omega_k[k]
                        if ret_split:
                            H[k, :, i, j] = num / den
                        else:
                            H[:, i, j] += num / den
            return H / self.m
        except Exception as e:
            raise (e)

    def spectrum_response(self, w: float | int | np.ndarray, rp: float | int | np.ndarray):
        try:
            if not isinstance(w, np.ndarray):
                w = float(w)

            if isinstance(rp, np.ndarray):
                rp = rp.flatten()
                assert len(rp) == len(self.eigvals)
            else:
                rp = float(rp) * np.ones(len(self.eigvals))

            sq = np.zeros((len(self.omega_k), len(w), self.nf, self.nf), dtype=np.complex128)
            # for wi,w in enumerate(ww):
            for i in range(self.nf):
                for j in range(self.nf):
                    for k, eigval in enumerate(self.eigvals):
                        phi = self.eigvecs[:, k]
                        num = (phi[i] * phi[j]) ** 2
                        den1 = eigval - w**2 + 2 * 1.0j * self.zeta * w * self.omega_k[k]
                        den2 = eigval - w**2 - 2 * 1.0j * self.zeta * w * self.omega_k[k]

                        sq[k, :, i, j] = num * rp[k] / (den1 * den2)
            return sq / self.m
        except Exception as e:
            raise (e)

    def get_modes(self, normalize=False):
        modes = np.concat([np.zeros((1, self.eigvecs.shape[0])), self.eigvecs])
        if normalize:
            for i, m in enumerate(modes[-1, :]):
                if np.abs(m) < 1e-3:
                    modes[:, i] /= m + np.sign(m) * 1e-3
                else:
                    modes[:, i] /= m
        return modes

    def plot_modes(self, normalize=False):
        modes = self.get_modes(normalize)

        fig, axes = plt.subplots(1, modes.shape[1], figsize=(2 * modes.shape[1], 6))
        yy = range(len(modes[:, 0]))

        for i, ax in enumerate(axes):

            ax.set_title(f"f :{self.omega_k[i]/(2*np.pi):2.2f}Hz", fontsize=10)
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
