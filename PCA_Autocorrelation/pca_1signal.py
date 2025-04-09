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


def bartlett_estimator(signal, min_period, max_period):
    """Calcule les coefficients de corrélation selon l'estimateur de Bartlett."""
    autocorr=[]
    for k in range(min_period,max_period):
        autocorr.append(np.sum(signal[k:]*signal[:-k])/(len(signal)))
    return (np.arange(min_period,max_period),np.array(autocorr))



# Load data
XX  = f'0{np.random.randint(1,10)}'
YY = '03'
Z = f'{np.random.randint(1,5)}'
WW = f'c0'
print(f'extrait de l\'echantillon : {XX, Z, WW}')

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
    signals.append(signal_i-np.mean(signal_i))
signals_matrix = np.array(signals)



#parameters
min_heartrate = 50 #bpm
max_heartrate = 170 #bpm
margin_m = 0.15 #s
margin_f = 0.05 #s
min_period = int(f_ech * 60/max_heartrate)
max_period = int(f_ech * 60/min_heartrate)



# Apply PCA for BSS
n_pca = 3
pca = PCA(n_components=n_pca)
signals_pca = pca.fit_transform(signals_matrix.T)
reconstructed_signals = pca.inverse_transform(signals_pca).T

# Plot original and PCA signals
plt.figure()
plt.subplot(2, 1, 1)
for i in range(34):
    plt.plot(signals_matrix[i], label=f"Original Signal {i+1}")
plt.title("Original Mixed Signals")
plt.xlabel('indice de la mesure')
plt.ylabel('amplitude d\'ECG')

plt.subplot(2, 1, 2)
for i in range(n_pca):
    plt.plot(signals_pca[:,i], label=f"Separated Signal {i}")
plt.title("Separated Signals using PCA")
plt.legend()
plt.xlabel('indice de la mesure')
plt.ylabel('amplitude d\'ECG')
plt.tight_layout()
plt.show()

#recherche des complexes QRS 
print(f'frequence réelle de la mere: {mhr} bpm')
print(f'frequence réelle du foetus: {fhr} bpm')
list_peaks=[]

for i in range(n_pca):
    #recherche des pics de l'autocorrélation
    current_signal = signals_pca[:,i].T/signals_pca[:,i].T[np.argmax(np.abs(signals_pca[:,i].T))]
    x,autocorr = bartlett_estimator(current_signal, min_period, max_period)
    peaks_autocorr, _ = find_peaks(autocorr, prominence=np.max(autocorr)*0.5)
    if len(peaks_autocorr)==0:
        peaks_autocorr=[np.argmax(autocorr)]
    period_main = x[peaks_autocorr[0]]
    print(f'frequence estimée de la composante n° {i+1}: {1/((period_main/f_ech)/60)} bpm')
    
    #recherche des pics sur la composante principale avec la fréquence associée à l'autocorrélation
    peaks_signal, _ = find_peaks(current_signal, distance = int(period_main*0.8), prominence=np.max(np.abs(current_signal))*0.45)
    if len(peaks_signal)==0:
        peaks_signal = find_peaks(current_signal, distance = int(period_main*0.8))
    list_peaks.append(peaks_signal)

    # plot des pics de l'autocorrelation
    plt.figure(2)
    plt.subplot(n_pca,1,i+1)
    plt.plot(x, autocorr, color='blue',label = f'autocorrélation du composant n°{i+1}')
    plt.scatter(x[peaks_autocorr], autocorr[peaks_autocorr], color='green', label = f'pic du composant n°{i+1}', marker = 'o')
    plt.xlabel('Ordre d\'autocorrelation')
    plt.ylabel(f"Autocorrelation")
    plt.legend()
    

    # indices des pics reportés sur les composantes principales
    plt.figure(1)
    plt.subplot(n_pca,1,i+1)
    plt.plot(current_signal, label=f"composant principal n: "+str(i), color='b')
    plt.scatter(peaks_signal, current_signal[peaks_signal], label=f"pics détectés",  marker='+', color='r')
    plt.legend()
    plt.xlabel('indice de la mesure')
    plt.ylabel('amplitude')
plt.tight_layout()
plt.show()

#on suppose qu'un superviseur est capable de reperer quelle composante
#de la pca est redondante
_,_,s1  = classif_report(list_peaks[1],qrs_f, margin_f, f_ech)
_,_,s2  = classif_report(list_peaks[2],qrs_f, margin_f, f_ech)
found_ecg_m = list_peaks[0]
if s1 > s2:
    found_ecg_f = list_peaks[1]
else:
    found_ecg_f = list_peaks[2]   
print(f'sensitivity, ppv, f1-score mere: {classif_report(found_ecg_m,qrs_m, margin_m, f_ech)}')
print(f'sensitivity, ppv, f1-score foetus: {classif_report(found_ecg_f,qrs_f, margin_f, f_ech)}')


# comparaison des complexes QRS trouvés et réels
interesting_channels=[0, 2, 5, 16, 33, 1, 7, 24, 29, 14]
for c in interesting_channels:
    plt.plot(signals_matrix[c],label=f'signal de la sonde {c}', color='black', alpha=0.6/len(interesting_channels))
plt.vlines(found_ecg_m, 0, signals_matrix[i,:][found_ecg_m], label=f"pics trouvés mere", linewidth=1, color='black')
plt.vlines(found_ecg_f, 0, signals_matrix[i,:][found_ecg_f], label=f"pics trouvés foetus", linewidth=1, color='black')
plt.vlines(qrs_m, 0, signals_matrix[i, qrs_m], color='red', linewidth=1, label='pics réels mere', linestyles='--')
plt.vlines(qrs_f, 0, signals_matrix[i, qrs_f], color='blue', linewidth=1, label='pics réels foetus', linestyles='--')
plt.xlabel('indice de la mesure')
plt.ylabel('amplitude d\'ECG')
plt.title('Reconstruction des complexes QRS')
plt.legend()
plt.show()

