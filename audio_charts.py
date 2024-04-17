import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft

# Wczytanie pliku WAV
czestotliwosc_probkowania, dane = wavfile.read(r'Data/Audio/audio.wav')

# Wyznaczenie liczby próbek i czasu trwania dźwięku
liczba_probek = len(dane)
czas_trwania = liczba_probek / czestotliwosc_probkowania

# Obliczanie FFT
widmo = fft(dane)
widmo_moc = np.abs(widmo)[:liczba_probek // 2] * 1 / liczba_probek  # jednostronne widmo mocy

# Oś częstotliwości dla widma
czestotliwosci = np.linspace(0, czestotliwosc_probkowania / 2, liczba_probek // 2)

# Wykres amplitudy
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(np.linspace(0, czas_trwania, liczba_probek), dane)
plt.title('Amplituda w funkcji czasu')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')

# Wykres widma mocy
plt.subplot(212)
plt.plot(czestotliwosci, widmo_moc)
plt.title('Widmo mocy')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Moc')

plt.tight_layout()
plt.show()
