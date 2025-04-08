## Project Overview

This project focuses on the analysis and separation of fetal and maternal ECG signals using various signal processing techniques. The workspace contains implementations of Independent Component Analysis (ICA), Matching Pursuit, and Principal Component Analysis (PCA) with autocorrelation to extract and analyze ECG signals from synthetic datasets.

The project is structured into multiple directories, each implementing a specific signal processing technique. The goal is to separate fetal ECG signals from maternal ECG signals and noise, and to evaluate the performance of these methods.

---

## Project Structure

```
slides_soutenance.pdf
docs/
	Andreotti_2016_Physiol._Meas._37_627 (1).pdf
	demos.pdf
	pbp_11.pdf
ICA/
	code_ICA.py
Matching Pursuit/
	matching pursuit.py
PCA_Autocorrelation/
	pca_1signal.py
	pca_allsignals.py
```

### Key Components

1. **ICA (Independent Component Analysis)**:
   - File: code_ICA.py
   - Implements ICA to separate fetal and maternal ECG signals from synthetic datasets.
   - Uses the `FastICA` algorithm from `sklearn` to extract independent components.
   - Visualizes the results and detects peaks in the separated signals.

2. **Matching Pursuit**:
   - File: matching pursuit.py
   - Implements the Matching Pursuit algorithm to iteratively decompose ECG signals into a set of basis functions.
   - Detects periodic patterns in the signal using a custom projection method.
   - Visualizes the residuals and approximations at each iteration.

3. **PCA with Autocorrelation**:
   - Files:
     - pca_1signal.py
     - pca_allsignals.py
   - Uses PCA to reduce dimensionality and separate signals.
   - Applies autocorrelation to detect periodic patterns in the principal components.
   - Evaluates the performance of the separation using metrics like sensitivity, positive predictive value, and F1-score.

4. **Documentation**:
   - The docs folder contains relevant research papers and references, such as:
     - Andreotti et al. (2016) on fetal ECG analysis.
     - Additional PDFs for demos and related studies.

5. **Slides and Handout**:
   - slides_soutenance.pdf: A presentation summarizing the project and its results.
   - CR.pdf: A more formal presentation of the underlying tools and methods used throughout the project.

---

## Requirements

Install the dependencies using:
```bash
pip install -r requirements.txt
```

---

## Dataset

The project uses a synthetic fetal ECG dataset from https://physionet.org/static/published-projects/fecgsyndb/fetal-ecg-synthetic-database-1.0.0.zip which you should unzip at:
```
D:\datacset_ecg\fetal-ecg-synthetic-database-1.0.0
```
You should end up with :
```
D:\datacset_ecg\fetal-ecg-synthetic-database-1.0.0\sub03\snr06dB\sub03_snr06dB_l1_c0_fecg1.dat 
```
being a path that exists on your device. If you fail to load this path, you can always change the path in use inside the python code.

