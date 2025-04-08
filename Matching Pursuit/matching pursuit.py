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

def extract_signal(xx, yy, z, ww, vvvv):
    if ww == '':
        filename = 'sub'+xx+'_snr'+yy+'dB_l'+z+'_'+vvvv
    else:   
        filename = 'sub'+xx+'_snr'+yy+'dB_l'+z+'_'+ww+'_'+vvvv
    filename = filename + '.dat'
    # Set working directory (change if needed !!!!!!!!)
    os.chdir('D:\\datacset_ecg\\fetal-ecg-synthetic-database-1.0.0' + '\\sub'+xx+'\\snr'+yy+'dB')
    filepath = os.getcwd()+'\\'+filename
    data = np.fromfile(filepath, dtype=np.int16)
    return data


def peigne(N_,periode, phase, largeur=6):
    P=np.zeros(N_)
    i=0
    while i+phase<N_:
        for j in range(i+phase-largeur,i+phase+largeur+1):
            if j>=0 and j<N_:
                P[j]=1 - abs(i+phase - j)/largeur
        i+=periode
    return P/np.sqrt(np.dot(P,P))


def project(signal_, min_period_, max_period_):
    max_proj = -float('inf')
    best_k = 0
    best_phi = 0

    for k in range(min_period_,max_period_):
        for phi in range(k):
            g_k_phi = peigne(len(signal_), k, phi)
            proj = np.abs(np.dot(signal_,g_k_phi))
            if proj > max_proj:
                max_proj = proj
                best_k = k
                best_phi = phi
    return peigne(len(signal_),best_k,best_phi)


#choose data
XX  = '07'
YY = '06'
Z = '1'
WW = 'c1'
VVVV = 'fecg1'

# Load whole data without the noise to show it cannot work at all
f_ech = 250 #Hz
data_fecg = extract_signal(XX, YY, Z, WW, 'fecg1')
data_mecg = extract_signal(XX, YY, Z, WW, 'mecg')
data_noisem = extract_signal(XX, YY, Z, WW, 'noise1')
data_noisef = extract_signal(XX, YY, Z, WW, 'noise2')
data = data_fecg + data_mecg #+ data_noisem +data_noisef


# Plot data
plt.figure()
for i in range(34):
    n_ecg = i
    ecg_n  = data[n_ecg::34]
    plt.plot(np.linspace(0,len(ecg_n)/f_ech, len(ecg_n)),ecg_n)
plt.xlabel('temps')
plt.ylabel('amplitude de l\'ECG')
plt.title('Tous les relevés d\'ECG')
plt.show()


#signal to process
T = 5 #s
N_ech = f_ech*T
signal = data[20::34][:N_ech]
signal = signal/max(signal)
N=len(signal)

#parameters (limited range for calculation time)
min_heartrate = 40 #bpm
max_heartrate = 100 #bpm
min_period = int(f_ech * 60/max_heartrate)
max_period = int(f_ech * 60/min_heartrate)

#matching pursuit algorithm
r = [float('inf') for i in range(len(signal))]
seuil = 0.5*np.sqrt(np.dot(signal,signal))
liste_proj = []
liste_u = []
meilleur_atome = project(signal, min_period, max_period)
liste_proj.append(meilleur_atome)
liste_u.append(meilleur_atome)
r = signal - np.dot(meilleur_atome, signal) * meilleur_atome

#plot the first residual
plt.figure()
plt.plot(signal, label = 'une composante du signal original')
plt.plot(r, label = 'residu d\'ordre 1')
plt.plot(np.dot(meilleur_atome, signal) * meilleur_atome, label = 'meilleur atome ordre 1')
plt.legend()
plt.xlabel('temps')
plt.ylabel('amplitude de l\'ecg')
plt.title('Ordre 1 du matching pursuit')
plt.show()

#itération de l'algorithme de matching pursuit
while np.sqrt(np.dot(r, r)) > seuil:  
    meilleur_atome = project(r, min_period, max_period)
    proj_residu_sur_atome = np.dot(meilleur_atome,r)*meilleur_atome
    proj_orthogonal = np.zeros_like(liste_u[0])
    for i in range(len(liste_u)):
        proj_orthogonal += np.dot(proj_residu_sur_atome, liste_u[i]) * liste_u[i]

    vecteur_innovation = proj_residu_sur_atome - proj_orthogonal
    vecteur_orthogonal = vecteur_innovation/np.sqrt(np.dot(vecteur_innovation, vecteur_innovation))
    r = r - np.dot(vecteur_orthogonal,r)*vecteur_orthogonal
    print(f'norme du résidu : {np.sqrt(np.dot(r,r))}')

    liste_proj.append(meilleur_atome)
    liste_u.append(vecteur_orthogonal)

#résolution du système matriciel
M = np.matmul(np.array(liste_u), np.array(liste_proj).T)
m = np.array([np.dot(signal, i) for i in liste_u])
x =np.linalg.solve(M,m)
approx = np.sum([abs(x[i])*liste_u[i] for i in range(len(x))], axis = 0)

#plot last residual
plt.figure()
plt.plot(approx, label='approx')
plt.plot(signal, label='signal')
plt.plot(r, label='residu')
plt.legend()
plt.xlabel('temps')
plt.ylabel('amplitude de l\'ecg')
plt.title(f'Ordre {len(liste_u)} du matching pursuit')
plt.show()
