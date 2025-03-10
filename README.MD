для встановлення залежностей спочатку запустити в терміналі:

``pip install -r requirements.txt``

* Program 1: DigiCommPy\signalgen.py: Function to generate sine wave
* Program 2: DigiCommPy\demo scripts.py: Call the function and plot output

``python -c "import demo_scripts; demo_scripts.sine_wave_demo()"``
* Program 3: DigiCommPy\signalgen.py: Function to generate a square wave
* Program 4: DigiCommPy\demo scripts.py: Generate square wave using SciPy

``python -c "import demo_scripts; demo_scripts.scipy_square_wave()"``
* Program 5: DigiCommPy\signalgen.py: Generating rectangular pulse with desired pulse width
* Program 6: DigiCommPy\signalgen.py: Generating Gaussian pulse with desired pulse width
* Program 7: DigiCommPy\demo scripts.py: Generating and plotting a chirp signal

``python -c "import demo_scripts; demo_scripts.chirp_demo()"``
* Program 8: DigiCommPy\essentials.py: Estimate and plot Welch PSD for a carrier modulated signal
* Program 9: DigiCommPy\essentials.py: Brute force method to compute convolution
* Program 10: DigiCommPy\essentials.py: Construct Toeplitz matrix of size (N+p-1)xp
* Program 11: DigiCommPy\essentials.py: Computing convolution of two sequences
* Program 12: DigiCommPy\chapter 1\demo scripts.py: Comparing different methods for convolution

``python -c "import demo_scripts; demo_scripts.compare_convolutions()"``
* Program 13: DigiCommPy\essentials.py: Generating an analytic signal for a given real-valued signal
* Program 14: DigiCommPy\demo scripts.py: Investigate components of an analytic signal

``python -c "import demo_scripts; demo_scripts.analytic_signal_demo()"``
* Program 15: DigiCommPy\demo scripts.py: Envelope and instantaneous phase extraction

``python -c "import demo_scripts; demo_scripts.extract_envelope_phase()"``
* Program 16: DigiCommPy\demo scripts.py: Demodulate phase modulated signal

``python -c "import demo_scripts; demo_scripts.hilbert_phase_demod()"``