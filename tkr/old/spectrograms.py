from imports import *


def resample(x_in, rate_in, rate_out, mode='fourier'):
    '''Resample a signal.

    Args:
        x_in (array): Input signal, sampled at rate_in hertz.
        rate_in (float): Sampling rate (in hertz) of input signal.
        rate_out (float): Sampling rate (in hertz) of output signal.
        mode (str): Choose between Fourier method ('fourier') and polyphase filter bank method.

    Returns:
        x_out (array): Output signal, sampled at rate_out hertz.
    '''

    if mode == 'polyphase':
        rate_ratio = rate_out / rate_in
        num, den = (rate_ratio).as_integer_ratio()
        x_out = dsp.resample_poly(x_in, up=num, down=den)

    else:
        rate_ratio = rate_out / rate_in
        num_samples = np.ceil(x_in.size * rate_ratio).astype(int)
        x_out = dsp.resample(x_in, num_samples)

    return x_out


def stft(x, dft_len=1024, step=512, window='hann', win_len=None, rate=None):
    '''Compute the short-time Fourier transform (STFT) of a real-valued array.
    
    Args:
        x (array): Discrete-time signal of interest. Assumed to be real-valued.
        dft_len (int): Number of samples to use in computing the discrete Fourier transform.
        step (int): Number of samples between adjacent blocks; also called hop size or hop length.
        window (str): Type of window function to use (e.g., rectangular, Hann, ...).
        win_len (int): Window function length. If None, win_len is set equal to dft_len.
        rate (int): Sampling rate, as measured in hertz.

    Returns:
        time (array): Array of sampling times. Only returned if rate is specified.
        freq (array): Array of sampling frequencies. Only returned if rate is specified.
        transform (array): Short-time Fourier transform (STFT) of input array.
    '''

    transform = librosa.stft(x, n_fft=dft_len, hop_length=step, window=window, win_length=win_len,
                             center=True, pad_mode='constant')

    if rate is not None:
        num_freqs, num_times = transform.shape # Number of rows, columns in STFT
        timestep = step / rate # Time resolution
        time = np.arange(0, timestep * num_times, timestep)
        freqstep = 0.5 * rate / num_freqs # Frequency resolution
        freq = np.arange(0, freqstep * num_freqs, freqstep)
        return time, freq, transform

    else:
        return transform

    
def make_spectrogram(x, dft_len=1024, step=512, window='hann', win_len=None, rate=None, norm=True):
    '''Compute the spectrogram of an array.
    
    Args:
        x (array): Discrete-time signal of interest.
        dft_len (int): Number of samples to use in computing the discrete Fourier transform.
        step (int): Number of samples between adjacent blocks; also called hop size or hop length.
        window (str): Type of window function to use (e.g., rectangular, Hann, ...).
        win_len (int): Window function length. If None, win_len is set equal to dft_len.
        rate (int): Sampling rate, as measured in hertz.
        norm (bool): If norm = True, normalize spectrogram to unit magnitude, i.e., 0 dB maximum.

    Returns:
        time (array): Array of sampling times. Only returned if rate is specified.
        freq (array): Array of sampling frequencies. Only returned if rate is specified.
        spec (array): Spectrogram of input array.        
    '''

    if rate is not None:
        time, freq, transform = stft(x, dft_len=dft_len, step=step, window=window, win_len=win_len, rate=rate)
        spec = np.square(np.abs(transform))
        if norm:
            spec /= np.max(spec)
        return time, freq, spec

    else:
        transform = stft(x, dft_len=dft_len, step=step, window=window, win_len=win_len, rate=rate)
        spec = np.square(np.abs(transform))
        if norm:
            spec /= np.max(spec)
        return spec


def show_spectrogram(time, freq, spec, figsize=(8, 4), colormap='magma', filename=None, title='spectrogram',
                     xlabel='elapsed time (seconds)', ylabel='frequency (hertz)', zlabel='power (dB)'):
    '''Display a spectrogram.
    
    Args:
        time (array): Sampling times, as measured in seconds.
        freq (array): Sampling frequencies, as measured in cycles per second, i.e., hertz.
        spec (array): Spectrogram.
        figsize (array): Array, list, or tuple of form (figure_width, figure_height).
        colormap (str): Colormap (e.g., 'viridis', 'magma', ...)
        filename (str): Filename to save figure under. If not specified, nothing is saved.
        title (str): Title. Defaults to 'spectrogram'.
        xlabel (str): x-axis label. Defaults to 'elapsed time (seconds)'.
        ylabel (str): y-axis label. Defaults to 'frequency (hertz)'.
        zlabel (str): z-axis (colorbar) label. Defaults to 'power (dB)'.
    '''

    num_rows, num_cols = 1, 1
    fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
    im = ax.pcolormesh(time, freq, spec, cmap=colormap)

    fig.colorbar(im, label=zlabel)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename + '.svg', bbox_inches='tight') # Save as SVG
        fig.savefig(filename + '.png', dpi=300, bbox_inches='tight') # Save as PNG

    plt.show()
    return


def make_mel_filter_bank(rate, dft_len, num_mels=128, freq_min=0.0, freq_max=None):
    '''Generate a mel filter bank.

    Args:
        rate (float): Sampling rate, as measured in hertz.
        dft_len (int): Length of discrete Fourier transform, i.e., number of frequency bins.
        num_mels (int): Number of bins in mel filter bank.
        freq_min (float): Minimum frequency in mel filter bank.
        freq_max (float): Maximum frequency in mel filter bank.

    Returns:
        mel_filter_bank (array): Use for converting frequency from hertz to mels.

    '''

    mel_filter_bank = librosa.filters.mel(sr=rate, n_fft=dft_len, n_mels=num_mels, fmin=freq_min, fmax=freq_max)
    return mel_filter_bank


