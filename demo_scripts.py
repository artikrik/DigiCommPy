import numpy as np
from numpy import log10, hanning
import matplotlib.pyplot as plt  # library for plotting
from signalgen import sine_wave  # import the function
from scipy import signal
from scipy.linalg import toeplitz
from scipy.signal import chirp, welch, hilbert
from scipy.fftpack import fft, fftshift, ifft

from essentials import analytic_signal, my_convolve, conv_brute_force


def sine_wave_demo():
    """
    Simulate a sinusoidal signal with given sampling rate
    """
    f = 10  # frequency = 10 Hz
    overSampRate = 30  # oversammpling rate
    phase = 1 / 3 * np.pi  # phase shift in radians
    nCyl = 5  # desired number of cycles of the sine wave
    (t, g) = sine_wave(f, overSampRate, phase, nCyl)  # function call
    plt.plot(t, g)  # plot using pyplot library from matplotlib package
    plt.title('Sine wave f=' + str(f) + ' Hz')  # plot title
    plt.xlabel('Time (s)')  # x-axis label
    plt.ylabel('Amplitude')  # y-axis label
    plt.show()  # display the figure


def scipy_square_wave():
    """
    Generate a square wave with given sampling rate
    """

    f = 10  # f = 10Hz
    overSampRate = 30  # oversampling rate
    nCyl = 5  # number of cycles to generate
    fs = overSampRate * f  # sampling frequency
    t = np.arange(start=0, stop=nCyl * 1 / f, step=1 / fs)  # time base
    g = signal.square(2 * np.pi * f * t, duty=0.2)
    plt.plot(t, g)
    plt.show()


def chirp_demo():
    """
    Generating and plotting a chirp signal
    """

    fs = 500  # sampling frequency in Hz
    t = np.arange(start=0, stop=1, step=1 / fs)  # total time base from 0 to 1 second
    g = chirp(t, f0=1, t1=0.5, f1=20, phi=0, method='linear')
    plt.plot(t, g)
    plt.show()


def fft_demo():
    fc = 10  # frequency of the carrier
    fs = 32 * fc  # sampling frequency with oversampling factor=32
    t = np.arange(start=0, stop=2, step=1 / fs)  # 2 seconds duration
    x = np.cos(2 * np.pi * fc * t)  # time domain signal (real number)
    np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    ax1.plot(t, x)  # plot the signal
    ax1.set_title("$x[n]= cos(2 \\pi 10 t)$")
    ax1.set_xlabel('$t=nT_s$')
    ax1.set_ylabel('$x[n]$')
    N = 256  # FFT size
    X = fft(x, N)  # N-point complex DFT, output contains DC at index 0
    # Nyquist frequency at N/2 th index positive frequencies from
    # index 2 to N/2-1 and negative frequencies from index N/2 to N-1
    # calculate frequency bins with FFT
    df = fs / N  # frequency resolution
    sampleIndex = np.arange(start=0, stop=N)  # raw index for FFT plot
    f = sampleIndex * df  # x-axis index converted to frequencies

    ax2.stem(sampleIndex, abs(X))  # sample values on x-axis
    ax2.set_title('X[k]')
    ax2.set_xlabel('k')
    ax2.set_ylabel('|X(k)|')
    ax3.stem(f, abs(X))  # x-axis represent frequencies
    ax3.set_title('X[f]')
    ax3.set_xlabel('frequencies (f)')
    ax3.set_ylabel('|X(f)|')
    fig.show()

    # re-order the index for emulating fftshift
    sampleIndex = np.arange(start=-N // 2, stop=N // 2)  # // for integer division
    X1 = X[sampleIndex]  # order frequencies without using fftShift
    X2 = fftshift(X)  # order frequencies by using fftshift
    df = fs / N  # frequency resolution
    f = sampleIndex * df  # x-axis index converted to frequencies
    # plot ordered spectrum using the two methods
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)  # subplots creation
    ax1.stem(sampleIndex, abs(X1))  # result without fftshift
    ax1.stem(sampleIndex, abs(X2), 'r')  # result with fftshift
    ax1.set_xlabel('k');
    ax1.set_ylabel('|X(k)|')
    ax2.stem(f, abs(X1))
    ax2.stem(f, abs(X2), 'r', use_line_collection=True)
    ax2.set_xlabel('frequencies (f)'), ax2.set_ylabel('|X(f)|');
    fig.show()


def compare_convolutions():
    """
    Comparing different methods for computing convolution
    """

    x = np.random.normal(size=7) + 1j * np.random.normal(size=7)  # normal random complex vectors
    h = np.random.normal(size=3) + 1j * np.random.normal(size=3)  # normal random complex vectors
    L = len(x) + len(h) - 1  # length of convolution output
    y1 = my_convolve(h, x)  # Convolution Using Toeplitz matrix
    y2 = ifft(fft(x, L) * (fft(h, L))).T  # Convolution using FFT
    y3 = np.convolve(h, x)  # Numpy's standard function
    print(f' y1 : {y1} \n y2 : {y2} \n y3 : {y3} \n')


def analytic_signal_demo():
    """
    Investigate components of an analytic signal
    """

    # import our function from essentials.py
    import matplotlib.pyplot as plt

    t = np.arange(start=0, stop=0.5, step=0.001)  # time base
    x = np.sin(2 * np.pi * 10 * t)  # real-valued f = 10 Hz
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(t, x)  # plot the original signal
    ax1.set_title('x[n] - real-valued signal')
    ax1.set_xlabel('n')
    ax1.set_ylabel('x[n]')
    z = analytic_signal(x)  # construct analytic signal
    ax2.plot(t, np.real(z), 'k', label='Real(z[n])')
    ax2.plot(t, np.imag(z), 'r', label='Imag(z[n])')
    ax2.set_title('Components of Analytic signal')
    ax2.set_xlabel('n')
    ax2.set_ylabel(r'$z_r[n]$ and $z_i[n]$')
    ax2.legend()
    plt.show()


def extract_envelope_phase():
    """
    Demonstrate extraction of instantaneous amplitude and phase from
    the analytic signal constructed from a real-valued modulated signal
    """

    from essentials import analytic_signal

    fs = 600  # sampling frequency in Hz
    t = np.arange(start=0, stop=1, step=1 / fs)  # time base
    a_t = 1.0 + 0.7 * np.sin(2.0 * np.pi * 3.0 * t)  # information signal
    c_t = chirp(t, f0=20, t1=t[-1], f1=80, phi=0, method='linear')
    x = a_t * c_t  # modulated signal
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(x)  # plot the modulated signal
    z = analytic_signal(x)  # form the analytical signal
    inst_amplitude = abs(z)  # envelope extraction
    inst_phase = np.unwrap(np.angle(z))  # inst phase
    inst_freq = np.diff(inst_phase) / (2 * np.pi) * fs  # inst frequency
    # Regenerate the carrier from the instantaneous phase
    extracted_carrier = np.cos(inst_phase)
    ax1.plot(inst_amplitude, 'r')  # overlay the extracted envelope
    ax1.set_title('Modulated signal and extracted envelope')
    ax1.set_xlabel('n');
    ax1.set_ylabel(r'x(t) and $|z(t)|$')
    ax2.plot(extracted_carrier)
    ax2.set_title('Extracted carrier or TFS')
    ax2.set_xlabel('n');
    ax2.set_ylabel(r'$cos[\omega(t)]$')
    plt.show()


def hilbert_phase_demod():
    """
    Demonstrate simple Phase Demodulation using Hilbert transform
    """

    fc = 210  # carrier frequency
    fm = 10  # frequency of modulating signal
    alpha = 1  # amplitude of modulating signal
    theta = np.pi / 4  # phase offset of modulating signal
    beta = np.pi / 5  # constant carrier phase offset
    # Set True if receiver knows carrier frequency & phase offset

    receiverKnowsCarrier = False
    fs = 8 * fc  # sampling frequency
    duration = 0.5  # duration of the signal
    t = np.arange(start=0, stop=duration, step=1 / fs)  # time base

    # Phase Modulation
    m_t = alpha * np.sin(2 * np.pi * fm * t + theta)  # modulating signal
    x = np.cos(2 * np.pi * fc * t + beta + m_t)  # modulated signal

    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(t, m_t)  # plot modulating signal
    ax1.set_title('Modulating signal')
    ax1.set_xlabel('t')
    ax1.set_ylabel('m(t)')

    ax2.plot(t, x)  # plot modulated signal
    ax2.set_title('Modulated signal')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x(t)')

    # Add AWGN noise to the transmitted signal
    mu = 0
    sigma = 0.1  # noise mean and sigma
    n = np.random.normal(mu, sigma, len(t))  # awgn noise
    r = x + n  # noisy received signal
    # Demodulation of the noisy Phase Modulated signal
    z = hilbert(r)  # form the analytical signal from the received vector
    inst_phase = np.unwrap(np.angle(z))  # instaneous phase

    if receiverKnowsCarrier:  # If receiver knows the carrier freq/phase perfectly
        offsetTerm = 2 * np.pi * fc * t + beta
    else:  # else, estimate the subtraction term
        p = np.polyfit(x=t, y=inst_phase, deg=1)  # linear fit instantaneous phase
        # re-evaluate the offset term using the fitted values
        estimated = np.polyval(p, t)
        offsetTerm = estimated

    demodulated = inst_phase - offsetTerm

    fig2, ax3 = plt.subplots()
    ax3.plot(t, demodulated)  # demodulated signal
    ax3.set_title('Demodulated signal')
    ax3.set_xlabel('n')
    ax3.set_ylabel(r'$\hat{m(t)}$')

    plt.show()
