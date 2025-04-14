# --- IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt

# --- INDLÆS OG KLARGØR DATA ---
df = pd.read_csv('AABA_2006-01-01_to_2018-01-01.csv')

# Konvert dato kolonne og sorter
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

# Vælg 'lukkeprisen' som time series signal
signal = df['Close'].copy()

# --- VISUALIEr ORIGINALT (STØJET) SIGNAL ---
plt.figure(figsize=(12, 5))
plt.plot(signal, label='Original (Noisy) Signal', color='gray')
plt.title('Original Time Series Signal (AABA Close Price)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- EXPONENTIAL MOVING AVERAGE (EMA) FILTERING ---
# Apply Exponential Moving Average instead of simple MA
ema_span = 10  # Controls how "responsive" the EMA is
ema = signal.ewm(span=ema_span, adjust=False).mean()

# Visualiser resultatet
plt.figure(figsize=(12, 5))
plt.plot(signal, label='Original', alpha=0.4)
plt.plot(ema, label=f'Exponential Moving Average (span={ema_span})', color='teal')
plt.title('Signal Before and After EMA Filter')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- FREQUENCY DOMAIN FILTERING ---
# Fjerner NaNs hvis der er nogen
clean_signal = signal.dropna()
n = len(clean_signal)
fs = 1  # 1 sample per day
frequencies = fftfreq(n, d=1/fs)
X_freq = fft(clean_signal)

# Anvender low-pass filtering
cutoff = 0.05  # Lower = more aggressive filtering
X_freq_filtered = X_freq.copy()
X_freq_filtered[np.abs(frequencies) > cutoff] = 0

# Inverse FFT #
filtered_signal = ifft(X_freq_filtered).real

# Visualiser frequency-filtered resultat
plt.figure(figsize=(12, 5))
plt.plot(clean_signal.index, clean_signal, label='Original', alpha=0.4)
plt.plot(clean_signal.index, filtered_signal, label='Frequency Domain Filtered', color='green')
plt.title('Signal Before and After Frequency Domain Filtering')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---  SEABORN VISUALIZATION --- #

# Reset index to use Seaborn (Foretrækker kolonner)
df_reset = df.reset_index()

# BOXPLOT: Viser outliers i lukkeprisen
plt.figure(figsize=(10, 5))
sns.boxplot(y='Close', data=df_reset, color='skyblue')
plt.title('Boxplot of AABA Close Price (Outlier Detection)')
plt.grid(True)
plt.tight_layout()
plt.show()

# HISTOGRAM + KDE: Prisdistribution
plt.figure(figsize=(10, 5))
sns.histplot(df_reset['Close'], bins=50, kde=True, color='steelblue')
plt.title('Histogram and KDE of AABA Close Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
