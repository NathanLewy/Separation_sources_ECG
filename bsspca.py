import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
import wfdb
import re

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

XX  = f'0{np.random.randint(1,9)}'
YY = '09'
Z = f'{np.random.randint(1,5)}'
WW = f'c{np.random.randint(1,5)}'
print(XX, Z, WW)


def extract_signal(xx, yy, z, ww, vvvv):
    try:
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
    except Exception as e:
        return []


def extract_qrs(xx, yy, z, ww, vvvv):
        if ww == '':
            filename = 'sub'+xx+'_snr'+yy+'dB_l'+z+'_'+vvvv
        else:   
            filename = 'sub'+xx+'_snr'+yy+'dB_l'+z+'_'+ww+'_'+vvvv
        os.chdir('D:\\datacset_ecg\\fetal-ecg-synthetic-database-1.0.0' + '\\sub'+xx+'\\snr'+yy+'dB')
        filepath = os.getcwd()+'\\'+filename
        record = wfdb.rdann(filepath, 'qrs')  # Charge les annotations QRS

        return record.sample

def extract_hr(xx, yy, z, ww, vvvv):
        if ww == '':
            filename = 'sub'+xx+'_snr'+yy+'dB_l'+z+'_'+vvvv
        else:   
            filename = 'sub'+xx+'_snr'+yy+'dB_l'+z+'_'+ww+'_'+vvvv
        filename = filename
        os.chdir('D:\\datacset_ecg\\fetal-ecg-synthetic-database-1.0.0' + '\\sub'+xx+'\\snr'+yy+'dB')
        filepath = os.getcwd()+'\\'+filename+'.hea'

        with open(filepath, 'r') as file:  # 'r' pour mode lecture
            data = file.read()
            fhr_match = re.search(r'#fhr:([-\d.]+)', data)
            mhr_match = re.search(r'#mhr:([-\d.]+)', data)
        # Afficher les premières annotations
        if fhr_match:
            return (fhr_match.group(1), mhr_match.group(1))
        else:
            return (0,0)
        



# Load data
data_fecg = extract_signal(XX, YY, Z, WW, 'fecg1')
data_fecg_twin = extract_signal(XX, YY, Z, WW, 'fecg2')
data_mecg = extract_signal(XX, YY, Z, WW, 'mecg')
data_noisem = extract_signal(XX, YY, Z, WW, 'noise1')
data_noisef = extract_signal(XX, YY, Z, WW, 'noise2')
qrs_m = extract_qrs(XX, YY, Z, WW, 'mecg')
qrs_f = extract_qrs(XX, YY, Z, WW, 'fecg1')
fhr, mhr = extract_hr(XX, YY, Z, WW, 'fecg1')
data = data_fecg + data_mecg + data_noisem +data_noisef
if len(data_fecg_twin)>0:
    data+=data_fecg_twin
    qrs_f2 = extract_qrs(XX, YY, Z, WW, 'fecg2')


#signal to process
f_ech = 250 #Hz
T = 30 #durée de fenetre en secondes
T0 = 15 #début de l'analyse en secondes
N_ech = f_ech*T
N_signaux = 34
signals=[]
data=data[34*int(T0*f_ech):]
qrs_m = [int(i-T0*f_ech) for i in qrs_m if T0 < i/f_ech < T + T0]
qrs_f = [int(i-T0*f_ech) for i in qrs_f if T0 < i/f_ech < T + T0]
for i in range(N_signaux):
    signal_i = data[i::34][:N_ech]
    signals.append((signal_i-np.mean(signal_i))/np.std(signal_i,ddof=1))
signals_matrix = np.array(signals)



#parameters
min_heartrate = 50 #bpm
max_heartrate = 170 #bpm
min_period = int(f_ech * 60/max_heartrate)
max_period = int(f_ech * 60/min_heartrate)



# Apply PCA for BSS
n_pca = 3
pca = PCA(n_components=n_pca)
signals_pca = pca.fit_transform(signals_matrix.T)  # Transpose to have signals as rows

# Reconstruct the separated signals from the PCA components
reconstructed_signals = pca.inverse_transform(signals_pca).T

# Plot original and separated signals
plt.figure()

# Plot original mixed signals (first 5 signals)
plt.subplot(2, 1, 1)
for i in range(34):
    plt.plot(signals_matrix[i], label=f"Original Signal {i+1}")
plt.title("Original Mixed Signals")


# Plot separated signals using PCA (first 5 components)
plt.subplot(2, 1, 2)

plt.plot(signals_pca, label=f"Separated Signal")
plt.title("Separated Signals using PCA")
plt.legend()

plt.tight_layout()
plt.show()




def bartlett_estimator(signal, min_period, max_period):
    """Calcule les coefficients de corrélation selon l'estimateur de Bartlett."""
    autocorr=[]
    for k in range(min_period,max_period):
        autocorr.append(np.sum(signal[k:]*signal[:-k])/(len(signal)-k))
    return (np.arange(min_period,max_period),np.array(autocorr))

def bt_estimator(signal, min_period, max_period):
    """Calcule les coefficients de corrélation selon l'estimateur de Bartlett."""
    autocorr=[]
    for k in range(min_period,max_period):
        autocorr.append(np.sum(signal[k:]*signal[:-k])/(len(signal)))
    return (np.arange(min_period,max_period),np.array(autocorr))

def are_close(A,B):
    return np.sum([np.min([np.abs(a-b) for b in B]) for a in A])/np.sqrt(len(A)*len(B))


print(f'frequence de la mere: {mhr} bpm')
print(f'frequence du foetus: {fhr} bpm')
list_peaks=[]

for i in range(n_pca):
    current_signal = signals_pca[:,i].T/signals_pca[:,i].T[np.argmax(np.abs(signals_pca[:,i].T))]
    x,autocorr = bt_estimator(current_signal, min_period, max_period)
    peaks_autocorr, _ = find_peaks(autocorr, prominence=np.max(autocorr)*0.5)
    if len(peaks_autocorr)==0:
        peaks_autocorr=[np.argmax(autocorr)]
    period_main = x[peaks_autocorr[0]]
    plt.figure(2)
    plt.subplot(n_pca,1,i+1)
    plt.plot(x, autocorr, color='blue',label = f'autocorrélation du composant n°{i+1}')
    plt.scatter(x[peaks_autocorr], autocorr[peaks_autocorr], color='green', label = f'pic du composant n°{i+1}', marker = 'o')
    plt.legend()
    print(f'frequence a priori de la composante n° {i+1}: {1/((period_main/f_ech)/60)} bpm')

    peaks_signal, _ = find_peaks(current_signal, distance = int(period_main*0.8), prominence=0.45)
    list_peaks.append(peaks_signal)

    #composantes principales
    plt.figure(1)
    plt.subplot(n_pca,1,i+1)
    plt.plot(current_signal, label=f"composant principal n: "+str(i), color='b')
    plt.scatter(peaks_signal, current_signal[peaks_signal], label=f"composant principal n: "+str(i),  marker='+', color='r')
    plt.legend()


plt.tight_layout()
plt.show()


d01 = are_close(list_peaks[0],list_peaks[1])
d02 = are_close(list_peaks[0],list_peaks[2])
d12 = are_close(list_peaks[0],list_peaks[2])

found_ecg_m = list_peaks[0]
if d01>d12: 
    found_ecg_f = list_peaks[1]
    print('took component 2')
else:
    if d02>d01:
        found_ecg_f = list_peaks[2]
        print('took component 3')
    else:
        found_ecg_f = list_peaks[1]
        print('took component 2')

interesting_channels=[0, 2, 5, 16, 33, 1, 7, 24, 29, 14]
for c in interesting_channels:
    plt.plot(signals_matrix[c],label='signal d origine', color='black', alpha=0.6/len(interesting_channels))

plt.vlines(found_ecg_m, 0, signals_matrix[i,:][found_ecg_m], label=f"pics n: "+str(i), linewidth=1, color='black')
plt.vlines(found_ecg_f, 0, signals_matrix[i,:][found_ecg_f], label=f"pics n: "+str(i), linewidth=1, color='black')
plt.vlines(qrs_m, 0, signals_matrix[i, qrs_m], color='red', linewidth=1, label='pics réels mere', linestyles='--')
plt.vlines(qrs_f, 0, signals_matrix[i, qrs_f], color='blue', linewidth=1, label='pics réels foetus', linestyles='--')
plt.legend()
plt.show()


def classif_report(predicted, real, margin, sr):
    tolerance = int(margin * sr)
    fp = 0
    tp = 0
    fn = 0
    for p in predicted:
        if np.min([np.abs(p-r) for r in real])<=tolerance:
            tp+=1
        else:
            fp+=1
    for r in real:
        if np.min([np.abs(p-r) for p in predicted])>tolerance:
            fn += 1
    sensitivity = tp/(fn + tp)
    positive_predictive_value = tp/(fp + tp)
    f1_score = 2*(positive_predictive_value * sensitivity)/(positive_predictive_value + sensitivity)
    return sensitivity, positive_predictive_value, f1_score


margin_m = 0.15 #s
margin_f = 0.05 #s
print(classif_report(found_ecg_m,qrs_m, margin_m, f_ech))
print(classif_report(found_ecg_f,qrs_f, margin_f, f_ech))