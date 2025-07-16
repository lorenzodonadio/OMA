import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

def sy_svd(Sy:np.ndarray):
    """Sy:np.ndarray, with 1st dimension equal to the frequency domain"""
    s_w = []
    u_w = []
    for i in range(Sy.shape[0]):
        U, S, Vh = svd(Sy[i,:,:])
        s_w.append(S)
        u_w.append(U[:,0])
    return np.array(s_w), np.array(u_w)

def SDOFbells_to_timedomain(svd_arr:np.ndarray,bell_indices:list[np.ndarray],zeropad_factor=5,normalize=True):
    assert zeropad_factor >=1
    assert len(bell_indices) == svd_arr.shape[1]
    time_signal = []
    freq_len = svd_arr.shape[0]
    for i,sel in enumerate(bell_indices):
        svsel = np.zeros(shape=(freq_len))
        svsel[sel] = svd_arr[sel,0]

        tmp = np.fft.irfft(svsel,n=2*zeropad_factor*freq_len)
        # tmp = np.fft.irfft(svsel)
        tmp = tmp[:len(tmp)//2]
        if normalize:
            tmp = tmp/np.max(tmp)
        time_signal.append(tmp)

    return np.array(time_signal)

def get_longest_cont_segment(arr:np.ndarray):
    """arr: 1D numpy array ints
    returns longest continous segment"""
    assert len(arr.shape) == 1,"Provide a 1D array"
    # Indices where the difference is not 1 (breaks in continuity)
    breaks = np.where(np.diff(arr) != 1)[0]

    # Add -1 and len(arr)-1 to mark the start and end of the ranges
    start_indices = np.insert(breaks + 1, 0, 0)
    end_indices = np.append(breaks, len(arr) - 1)

    # 
    max_len = 0
    best_segment = np.array([])
    for start, end in zip(start_indices, end_indices):
        if end - start + 1 > max_len:
            max_len = end - start + 1
            best_segment = arr[start:end+1]
    return best_segment

def get_SDOFbell_indices(arr:np.ndarray,mac_treshold=0.9):
    fdd_selections = []
    for j in range(arr.shape[1]):
        tmp = np.argwhere(arr[:,j]>mac_treshold).flatten()
        fdd_selections.append(get_longest_cont_segment(tmp))
    return fdd_selections

def max_min_envelope(x:np.ndarray):
    idxcross = np.where(np.diff(np.sign(x)))[0]
    x_diff_abs = np.abs(np.diff(np.append(x,0)))

    max_indices = []
    min_indices = []

    for i,j in zip(np.insert(idxcross[:-1],0,0),idxcross):
        x[i:j]
        maxidx = i + np.argmax(x[i:j])
        minidx = i + np.argmin(x[i:j])
    # the maxindex derivative is higher than the min index derivative
    # then the point is a minimum and we reject the max index
        if x_diff_abs[maxidx] > x_diff_abs[minidx]:
            min_indices.append(minidx)
        else:
            max_indices.append(maxidx)
    return np.array(max_indices),np.array(min_indices)


def mac_matrix(Fi1: np.ndarray, Fi2: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of MAC between columns of `Fi1` and rows of `u`.

    Parameters
    ----------
    Fi1 : array of shape (n, m)
        Reference mode shapes (e.g., Fi1s).
    Fi2 : array of shape (k, n)
        Mode shape vectors to compare (e.g., u_w).

    Returns
    -------
    mac_arr : array of shape (k, m)
        MAC values for each pair.
    """
    # Fi1: (n, m), u: (k, n) -> transpose u to (n, k)
    Fi1_H_Fi1 = np.sum(np.abs(Fi1)**2, axis=0)  # (m,)
    Fi2_H_Fi2 = np.sum(np.abs(Fi2)**2, axis=1)        # (k,)

    # (k, m): dot product for all combinations
    dot_products = np.abs(Fi2 @ Fi1)**2

    # Denominator: outer product (k, m)
    denom = np.outer(Fi2_H_Fi2, Fi1_H_Fi1)

    return dot_products / denom

def mac(Fi1:np.ndarray,Fi2:np.ndarray):
    '''
    This function returns the Modal Assurance Criterion (MAC) for two mode 
    shape vectors.
    
    If the input arrays are in the form (n,) (1D arrays) the output is a 
    scalar, if the input are in the form (n,m) the output is a (m,m) matrix
    (MAC matrix).
    
    ----------
    Parameters
    ----------
    Fi1 : array (1D or 2D)
        First mode shape vector (or matrix).
    Fi2 : array (1D or 2D)
        Second mode shape vector (or matrix). 
        
    -------
    Returns
    -------
    MAC : float or (2D array)
        Modal Assurance Criterion.
    '''
        
    return np.abs((Fi1.conj().T @ Fi2)**2/((Fi1.conj().T @ Fi1)*(Fi2.conj().T @ Fi2)))

def plot_time_signals_with_envelopes(time_signal, envelopes:list[np.ndarray], t_vect:np.ndarray, show_minima=False,ncols = 2):
    """
    Plot multiple time signals with their max (and optionally min) envelope points in subplots.

    Parameters:
        time_signal (list or array): List of 1D arrays, each representing a time-domain signal.
        envelopes (list of tuples): Each tuple contains (max_indices, min_indices) for the corresponding signal.
        t_vect (array): Time vector corresponding to the signals.
        show_minima (bool): Whether to also show minima envelope points.
    """
    n_signals = len(time_signal)
    nrows = (n_signals + 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), sharex=True)
    axes = axes.flatten()  # Flatten to simplify indexing

    for i, sdof in enumerate(time_signal):
        ax = axes[i]
        maxidx, minidx = envelopes[i]

        ax.plot(t_vect, sdof, label='Signal')
        ax.plot(t_vect[maxidx], sdof[maxidx], 'r.--', label='Maxima')
        if show_minima:
            ax.plot(t_vect[minidx], sdof[minidx], 'b.--', label='Minima')

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Reconstructed Signal {i+1}")
        ax.legend()

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig,axes
