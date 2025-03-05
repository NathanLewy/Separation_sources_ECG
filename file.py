import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct



# Choose simulation
'''
0	Baseline (no events) + noise
1	Foetal movement + noise
2	MHR /FHR acceleration / decelerations + noise
3	Uterine contraction + noise
4	Ectopic beats (for both foetus and mother) + noise
5	Additional NI-FECG (twin pregnancy) + noise


XX	=	simulated pregnancy number [01 to 10].
YY	=	SNR level [00, 03, 06, 09, 12] dB.
Z	=	repetition number [1 to 5].
WW	=	[c0 to c5], or empty for the baseline case, representing the cases shown in the above section.
VVVV	=	Fetal ECG [fecgN] where N is the fetus number, maternal ECG [mecg], or [noise].


.hea for header files
.dat for signal files. 
.qrs give machine generated QRS locations of their respective signals.
channels 1 to 32 are the abdominal FECG channels
channels 33 to 34 are the maternal reference ECG channels.
'''

XX  = '08'
YY = '06'
Z = '1'
WW = 'c1'
VVVV = 'fecg1'

def extract_signal(xx, yy, z, ww, vvvv):
    if ww == '':
        filename = 'sub'+xx+'_snr'+yy+'dB_l'+z+'_'+vvvv
    else:   
        filename = 'sub'+xx+'_snr'+yy+'dB_l'+z+'_'+ww+'_'+vvvv
    filename = filename + '.dat'
    # Set working directory
    os.chdir('D:\\datacset_ecg\\fetal-ecg-synthetic-database-1.0.0' + '\\sub'+xx+'\\snr'+yy+'dB')
    # Load data
    filepath = os.getcwd()+'\\'+filename
    data = np.fromfile(filepath, dtype=np.int16)
    return data




# Load data
data_fecg = extract_signal(XX, YY, Z, WW, 'fecg1')
data_mecg = extract_signal(XX, YY, Z, WW, 'mecg')
data_noisem = extract_signal(XX, YY, Z, WW, 'noise1')
data_noisef = extract_signal(XX, YY, Z, WW, 'noise2')
data = data_fecg + data_mecg + data_noisem +data_noisef
f_ech = 250 #Hz

# Plot data
plt.figure()
for i in range(34):
    n_ecg = i
    ecg_n  = data[n_ecg::34]
    plt.plot(ecg_n)
plt.show()