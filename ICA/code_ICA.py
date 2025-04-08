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

XX  = '03'
YY = '12'
Z = '1'
WW = 'c1'
vvvv = 'fecg1' #on laisse un seul foetus pour l'instant


#en premier la fhr, en deuxieme la mhr
dico = {'01' : ( 143.0077, 107.7705), 

        '03' : (137.1695 , 77.3140),

        '07' : (115.5789,59.6965) }



def extract_signal(xx, yy, z, ww, vvvv):
    if ww == '':
        filename = 'sub'+xx+'_snr'+yy+'dB_l'+z+'_'+vvvv
    else:   
        filename = 'sub'+xx+'_snr'+yy+'dB_l'+z+'_'+ww+'_'+vvvv
    filename = filename + '.dat'
    # Set working directory
    os.chdir('D:\\datacset_ecg\\fetal-ecg-synthetic-database-1.0.0' + '\\sub'+xx+'\\snr'+yy+'dB')
    # Load data
    filepath = os.getcwd()+'/'+filename
    data = np.fromfile(filepath, dtype=np.int16)
    return data



def signal(XX,YY,Z,WW,vvvv):
    # Load data
    data_fecg = extract_signal(XX, YY, Z, WW, 'fecg1')
    data_mecg = extract_signal(XX, YY, Z, WW, 'mecg')
    data_noisem = extract_signal(XX, YY, Z, WW, 'noise1')
    data_noisef = extract_signal(XX, YY, Z, WW, 'noise2')
    data = data_fecg + data_mecg + data_noisem +data_noisef
    f_ech = 250 #Hz
         
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import FastICA
    from numpy import linspace
    X = np.array([np.mean([(data[i::34]) for i in range(1,6)],axis=0),
              np.mean([(data[i::34]) for i in range(8,14)],axis=0),
              np.mean([(data[i::34]) for i in range(16,23)],axis=0),
              np.mean([(data[i::34]) for i in range(26,32)],axis=0)])

    X = X.astype(np.float64)
    X = X.transpose(1,0)
    X /= X.std(axis = 0)
    ica = FastICA(n_components = 2, whiten = "arbitrary-variance")
    S_ = ica.fit_transform(X)
    S_ = S_.transpose(1,0)


    #détection des fréquences des marqueurs
    from scipy.signal import find_peaks
    peaks1, _ = find_peaks(abs(S_[0]), height=abs(S_[0]).max()/3 ,width = [1,10],distance = 40)
    peaks2, _ = find_peaks(abs(S_[1]), height=abs(S_[1]).max()/3,width = [1,10],distance = 40)



    plt.figure()
    plt.plot(data[2::34])
    plt.title("le cas " + str(dico[XX]) + " de bpm et " + str(YY) + " de bruit")
    plt.plot(peaks1, S_[0][peaks1], "x")
    plt.plot(peaks2, S_[1][peaks2], "o")



    #plot results
    plt.figure()
    plt.subplot(4,1,2)
    plt.plot(S_[0])
    plt.plot(peaks1, S_[0][peaks1], "x")
    plt.plot(S_[1])    
    plt.title("le cas " + str(dico[XX]) + " de bpm et " + str(YY) + " de bruit")
    plt.plot(peaks2, S_[1][peaks2], "o")

    plt.show()




#différentes fréquences


XX  = '03'
YY = '12'
Z = '1'
WW = 'c1'
vvvv = 'fecg1' #on laisse un seul foetus pour l'instant

signal(XX,YY,Z,WW,vvvv)


XX  = '07' # mettre à 07
YY = '12'
Z = '1'
WW = 'c1'
vvvv = 'fecg1' #on laisse un seul foetus pour l'instant

signal(XX,YY,Z,WW,vvvv)


XX  = '01' # mettre à 01
YY = '12'
Z = '1'
WW = 'c1'
vvvv = 'fecg1' #on laisse un seul foetus pour l'instant

signal(XX,YY,Z,WW,vvvv)


#différents niveaux de bruits 
XX  = '03' # mettre à 01
YY = '12'
Z = '1'
WW = 'c1'
vvvv = 'fecg1' #on laisse un seul foetus pour l'instant

signal(XX,YY,Z,WW,vvvv)


XX  = '03' # mettre à 01
YY = '09'
Z = '1'
WW = 'c1'
vvvv = 'fecg1' #on laisse un seul foetus pour l'instant

signal(XX,YY,Z,WW,vvvv)


XX  = '03' # mettre à 01
YY = '06'
Z = '1'
WW = 'c1'
vvvv = 'fecg1' #on laisse un seul foetus pour l'instant

signal(XX,YY,Z,WW,vvvv)

XX  = '03' # mettre à 01
YY = '03'
Z = '1'
WW = 'c1'
vvvv = 'fecg1' #on laisse un seul foetus pour l'instant

signal(XX,YY,Z,WW,vvvv)

