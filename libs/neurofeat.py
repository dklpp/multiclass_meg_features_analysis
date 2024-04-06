import scipy.io
import numpy as np
import h5py as h5

import mat73 

from scipy.signal import hilbert
import pandas as pd
from scipy.stats import zscore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from fooof.plts.spectra import plot_spectrum
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic

import networkx as nx # graphs

from frites.utils import parallel_func


def threshold_mat(data,thresh=2):
    current_data=data
    binarized_data=np.where(np.abs(current_data)>thresh,1,0)
    return (binarized_data)

def transprob(aval,nregions): # (t,r)
    mat = np.zeros((nregions, nregions))
    norm = np.sum(aval, axis=0)
    for t in range(len(aval) - 1):
        ini = np.where(aval[t] == 1)
        mat[ini] += aval[t + 1]
    mat[norm != 0] = mat[norm != 0] / norm[norm != 0][:, None]
    return mat

def Transprob(ZBIN,nregions, val_duration): # (t,r)
    mat = np.zeros((nregions, nregions))
    A = np.sum(ZBIN, axis=1)
    a = np.arange(len(ZBIN))
    idx = np.where(A != 0)[0]
    aout = np.split(a[idx], np.where(np.diff(idx) != 1)[0] + 1)
    ifi = 0 # ifi = 0
    for iaut in range(len(aout)):
        if len(aout[iaut]) > val_duration:
            mat += transprob(ZBIN[aout[iaut]],nregions)
            ifi += 1
    mat = mat / ifi
    return mat,aout

def find_avalanches(data,thresh=2, val_duration=2):
    binarized_data=threshold_mat(data,thresh=thresh)
    N=binarized_data.shape[1]
    mat, aout = Transprob(binarized_data,N,val_duration)
    aout=np.array(aout,dtype=object)
    list_length=[len(i) for i in aout]
    unique_sizes=set(list_length)
    min_size,max_size=min(list_length),max(list_length)
    list_avalanches_bysize={i:[] for i in unique_sizes}
    for s in aout:
        n=len(s)
        list_avalanches_bysize[n].append(s)
    return(aout,min_size,max_size,list_avalanches_bysize, mat)

def _plv(phi_x, phi_y):
    X = np.vstack((phi_x, phi_y))
    complex_phase_diff = np.exp(1j * (np.diff(X, axis=0).squeeze()))
    plv = np.abs(np.sum(complex_phase_diff)) / len(complex_phase_diff)
    return plv

class Matr(object):
    def __init__(self, matr, objtype):
        self.matr = matr
        self.type = objtype # 'ATM', 'CC', 'PLV', 'AEC'
        self.graph = nx.from_numpy_array(self.matr, parallel_edges=False)
        
    def flatten(self):
        """Returns a flattened array without symmetric values and main diagonal elements."""
        colnames = [f'{self.type}_{i}-{j}' for i in range(self.matr.shape[0]) # PLV_1-10
                         for j in range(i+1, self.matr.shape[1])]
        data = [self.matr[i, j] for i in range(self.matr.shape[0]) 
                         for j in range(i+1,self.matr.shape[1])]
        df = pd.DataFrame([data], columns=colnames)
        return df
        
    def betw_centr(self):
        """Betweenness centrality. Returns a df row with value for each region."""
        btw_cent = nx.betweenness_centrality(self.graph, weight='weight')
        indexes = [f'centr_{i}' for i in list(btw_cent.keys())]
        df = pd.DataFrame(data=btw_cent.values(), index=indexes).T
        return df
    
    def eign_centr(self):
        """Eigenvector centrality. Returns a df row with value for each region."""
        egn_cent = nx.eigenvector_centrality(self.graph, weight='weight', max_iter=200)
        indexes = [f'centr_{i}' for i in list(egn_cent.keys())]
        df = pd.DataFrame(data=egn_cent.values(), index=indexes).T
        return df
    
    def degree(self):
        """Compute weighted degree for each region."""
        weighted_degrees = dict(self.graph.degree(weight='weight'))
        indexes = [f'degr_{i}' for i in list(weighted_degrees.keys())]
        df = pd.DataFrame(data=weighted_degrees.values(), index=indexes).T
        return df 
    
    
class Patient(object):
    def __init__(self, timeser, patient_class, ID, sfreq=1024.0, n_ch=116, mindim=200000):
        """
        :timeser: n_channels x n_times time series MEG signal (type: np.array)
        :sfreq: sampling frequency (type: float)
        :patient_class: class of a patient: PD, MCI, SLA, MS (type: string)
        :ID: fake variable for looping over the same class patients 
        :mindim: Number of data points to threshold 
        :n_ch: Number of channels to be used
        """
        self.sfreq = sfreq
        self.timeser = timeser[:n_ch, :mindim] # timeser[:, :mindim] for thresholding 
        self.patient_class = patient_class
        self.id = ID
        
    def calcAvalMatrix(self, returnFlatten=False, returnAsBetweenness=False, 
                       returnAsEigenCentr=False, returnAsDegree=False):
        """
        Number of output features: (n_ch * n_ch - 116 ) / 2
        """
        num_time_points, num_channels = (self.timeser.shape[1], self.timeser.shape[0])
        
        zsc_timeseries = pd.DataFrame(self.timeser.T).apply(zscore).values
        
        zscore_epoched_time_series = zsc_timeseries.reshape((1, zsc_timeseries.shape[0], zsc_timeseries.shape[1]))
        
        ntrials, ntimes, nchannels = zscore_epoched_time_series.shape
        transition_matrices_list = []
        for trial in range(ntrials):
            avalanches, _, _, _, mat = find_avalanches(zscore_epoched_time_series[trial, :, :],
                                                      thresh=2, val_duration=2)
            for avalanche in avalanches:
                transition_matrices_list.append(mat)
        transition_matrix = np.stack(transition_matrices_list, axis=2)
        transition_matrix = transition_matrix.mean(axis=2) # (n_ch, n_ch)
        
        matr = Matr(transition_matrix, objtype='ATM')

        if returnAsBetweenness:
            return matr.betw_centr()
        elif returnAsEigenCentr:
            return matr.eign_centr()
        elif returnAsDegree:
            return matr.degree()
        elif returnFlatten:
            return matr.flatten()
        else:
            return transition_matrix # nroi x nroi
        
    def calcPhaseLockValue(self, n_jobs=1, verbose=False, returnFlatten=False, returnAsBetweenness=False, 
                           returnAsEigenCentr=False, returnAsDegree=False):
        """Paralellized, much faster version."""
        # Get data dimensions
        nroi, ntimes = self.timeser.shape
        # Get all edge pairs indexes
        npairs = nroi * nroi
        phases = np.angle(scipy.signal.hilbert(self.timeser, axis=-1))
        # define the function to compute in parallel
        parallel, p_fun = parallel_func(_plv, n_jobs=n_jobs, verbose=verbose, total=npairs)
        # Compute the single trial coherence
        out = parallel(p_fun(phases[i], phases[j]) for i in range(0,nroi)
                                                   for j in range(0,nroi))
        out = np.stack(out)

        matr = Matr(out.reshape((116, 116)), objtype='PLV')

        if returnAsBetweenness:
            return matr.betw_centr()
        elif returnAsEigenCentr:
            return matr.eign_centr()
        elif returnAsDegree:
            return matr.degree()
        elif returnFlatten:
            return matr.flatten()
        else:
            return out.reshape((116, 116)) # adjacency matrix nroi x nroi 

        
#     def calcPhaseLockValue(self, returnAsBetweenness=False, returnAsEigenCentr=False, returnAsDegree=False):
#         """
#         Number of output features: (n_ch * n_ch - 116 ) / 2
#         """
#         probe = hilbert(self.timeser.T, axis=0) # (n_times, n_channels)
#         out = np.zeros((probe.shape[1], probe.shape[1]), dtype=float)
#         for idx in range(probe.shape[1]):
#             temp = np.conj(probe[:, idx])[:, np.newaxis] * np.ones((1, probe.shape[1]))
#             e = np.exp(1j * np.angle(temp * probe))
#             out[idx, :] = np.abs(np.sum(e, axis=0)) / probe.shape[0]
            
            
#         matr = Matr(out, objtype='PLV')    
            
#         if returnAsBetweenness:
#             return matr.betw_centr()
#         elif returnAsEigenCentr:
#             return matr.eign_centr()
#         elif returnAsDegree:
#             return matr.degree()
#         else:
#             return matr.flatten()
            
    
    def calcCrossCorrelations(self, returnFlatten=False, returnAsBetweenness=False, 
                              returnAsEigenCentr=False, returnAsDegree=False):
        """
        Number of output features: (n_ch * n_ch - 116 ) / 2
        """
        corr_matrix = round(pd.DataFrame(self.timeser).T.iloc[:, :].corr(), 2) # (n_times, n_channels)
        matr = Matr(corr_matrix.values, objtype='CC')
        
        if returnAsBetweenness:
            return matr.betw_centr()
        elif returnAsEigenCentr:
            return matr.eign_centr()
        elif returnAsDegree:
            return matr.degree()
        elif returnFlatten:
            return matr.flatten()
        else:
            return corr_matrix # adjacency matrix nroi x nroi
    
    def calcFoofParameters(self):
        """
        Number of output features: n_channels 
        """
        win = int(10 * self.sfreq)
        fs = self.sfreq
        meg_bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 48)}
        params_list = []
        
        for i in range(116):
            data = self.timeser[i, :]
            freqs, psd = signal.welch(data, fs, nperseg=win, average='median')
            fm = FOOOF(min_peak_height=0.05)
            freq_range = [0.5, 48]
            fm.fit(freqs, psd, freq_range)
            fres = fm.get_results()
            params_list.append(fres.aperiodic_params[1])
            
        colnames = [f'foof_{i}' for i in range(116)]
        df = pd.DataFrame([params_list], columns=colnames)
        return df
    
    def calcPowerSpectrBands(self):
        """
        Number of output features: n_ch * n_freq_bands
        """
        fs = self.sfreq
        win = int(10 * fs)
        meg_power_fft = dict()
        
        # Standartize all signals across channels 
        zscored_sign = pd.DataFrame(self.timeser.T).apply(zscore).values
        
        for i in range(0, 116):
            data = zscored_sign[:, i]
            freqs, psd = signal.welch(data, fs, nperseg=win)
            fft_vals = np.absolute(np.fft.rfft(data))
            power_vals = np.square(fft_vals)
            fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)
            meg_bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 48)}
            meg_band_fft = dict()
            for band in meg_bands:  
                freq_ix = np.where((fft_freq >= meg_bands[band][0]) & 
                                   (fft_freq <= meg_bands[band][1]))[0]
                meg_band_fft[band] = np.mean(fft_vals[freq_ix])

            for band in meg_bands:  
                freq_ix = np.where((fft_freq >= meg_bands[band][0]) & 
                                   (fft_freq <= meg_bands[band][1]))[0]
                meg_power_fft['PS_{}_ch{}'.format(band, i)] = np.mean(power_vals[freq_ix])
        
        df = pd.DataFrame(meg_power_fft.values(), meg_power_fft.keys()).T
        # df['PATIENT_ID'] = PATIENT_ID
        return df 
    
    def calcAmplEnvCorr(self, returnFlatten=False, returnAsBetweenness=False, 
                        returnAsEigenCentr=False, returnAsDegree=False):
        """
        Number of output features: (n_ch * n_ch - 116 ) / 2
        """
        an_sign = hilbert(self.timeser.T, axis=0) # (n_times, n_channels)
        an_sign = np.abs(an_sign)
        an_sign_z = pd.DataFrame(an_sign).apply(zscore).values # z_standardized hilbert signals
        aec_matrix = round(pd.DataFrame(an_sign_z).corr(), 2)  # pearson corr
        matr = Matr(aec_matrix.values, objtype='AEC')
        
        if returnAsBetweenness:
            return matr.betw_centr()
        elif returnAsEigenCentr:
            return matr.eign_centr()
        elif returnAsDegree:
            return matr.degree()
        elif returnFlatten:
            return matr.flatten()
        else:
            return aec_matrix # adjacency matrix nroi x nroi
        
# _____________________________________________________________________________________________
# _____________________________________________________________________________________________
#                                     FEATURE SELECTION
#                      (Kruskal-Wallis Test + FDR test (multicomparison problem))
# _____________________________________________________________________________________________
# _____________________________________________________________________________________________
        
        
def select_features_kruskal_Wallis(df, num_feat, feat, sel_feat):
    """Takes a df with many features. Performs Kruskal-Wallis Test. Performs FDR test.
    
    Inputs:
    : df : input dataframe
    : num_feat : total number of features 
    : feat : type of feature, ['plv', 'aec', 'foof', 'cc', 'pw', 'atm'] _eign _betw 
    : sel_feat : number of features to select
    
    Returns:
    df - dataframe with top features selected
    """
    # p values for Kruskal-Wallis test
    classes = list(np.unique(df['class'].values)) # list of classes 
    
    p_vals = {f'{feat}_{i}': kruskal(*[df[df['class'] == cls].iloc[:, i] 
                                for cls in classes]).pvalue for i in range(num_feat)}
    pvals = np.array(list(p_vals.values())) # num_feat p-values

    # FDR correction
    pvals_corrected = fdrcorrection(pvals)[1]
    pvals_fdr = {f'{feat}_{i}': pvals_corrected[i] for i in range(0, num_feat)}
    top_features = [x[0] for x in sorted(pvals_fdr.items(), key=lambda x:x[1])[:sel_feat]]
    
    # Rename columns 
    cols = [f'{feat}_{i}' for i in range(0, num_feat)]
    cols = cols + ['ID', 'class']
    df.columns = cols
    
    df = df[top_features + ['class']]
    
    return df


# _____________________________________________________________________________________________
# _____________________________________________________________________________________________
#                             MACHINE LEARNING EVALUATION METHODS
#                               (SVM, LDA, XGBoost, Naive Bayes)
# _____________________________________________________________________________________________
# _____________________________________________________________________________________________

# Hyperparameters for tuning
param_grid_lda = {'solver': ['svd', 'lsqr', 'eigen']} 
param_grid_svm = {'C': [0.1,1, 10, 100], 
              'gamma': [1,0.1,0.01,0.001],
               'kernel': ['rbf', 'poly']}
param_grid_nb = {'var_smoothing': np.logspace(0, -9, num=100)}

# input datasets
all_datasets = ['df_plv_flat.csv', 'df_aec_flat.csv', 'df_aec_eign.csv',
                'df_aec_betw.csv', 'df_cc_betw.csv', 'df_plv_betw.csv', 'df_plv_eign.csv',
                'df_foof_4.csv', 'df_ps_5fb.csv', 'df_atm.csv', 'df_cc.csv', 'df_atm_betwcentr.csv',
                'df_atm_eigncentr.csv', 'df_plv_degr.csv', 'df_aec_degr.csv', 'df_atm_degr.csv',
                'df_cc_degr.csv']

# output dataframe with evaluation metrics
final_ml = pd.DataFrame(columns=['SVM_balanced_accuracy', 'LDA_balanced_accuracy', 'XGB_balanced_accuracy', 
                                 'NB_balanced_accuracy', 'SVM_accuracy', 'LDA_accuracy', 'XGB_accuracy', 
                                 'NB_accuracy', 'SVM_f1_weighted', 'LDA_f1_weighted', 'XGB_f1_weighted', 
                                 'NB_f1_weighted', 'SVM_f1_macro', 'LDA_f1_macro', 'XGB_f1_macro', 
                                 'NB_f1_macro'], index=all_datasets)

# list for plotting balanced accuracy for each classifier as bar plots 
plotting_list = []

def eval_ml(ds: str, n_repeats: int) -> dict:
    """Without feature scaling.
    
    Input:
    :ds: dataset path to import as dataframe
    :n_repeats: number of repeats for RepeatedStratifiedKFolds
    
    Returns:
    :metrics: dict with different metrics for all classifiers
    """
    print(ds)
    df = pd.read_csv(ds)
    print(df.shape)
    
    if ds == 'df_ps_5fb.csv':
        df.iloc[:,:-2] = df.iloc[:,:-2].apply(zscore) # PS has very large values here => z-score
        
    df = select_features_kruskal_Wallis(df, feat=ds[3:6], num_feat=len(df.columns)-2,
                                           sel_feat=20)
    
    X = df.drop('class', axis=1)
    le = LabelEncoder()
    y = le.fit_transform(df['class'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle=True)
    
    cv_grid_search = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=43) # for GridSearch CV
    
    grid_lda = GridSearchCV(LinearDiscriminantAnalysis(), param_grid_lda, scoring='balanced_accuracy', cv=cv_grid_search) 
    grid_svm = GridSearchCV(SVC(random_state=102), param_grid_svm, scoring='balanced_accuracy', cv=cv_grid_search, 
                            refit=True, verbose=0)
    grid_nb = GridSearchCV(GaussianNB(random_state=303), param_grid_nb, scoring='balanced_accuracy', cv=cv_grid_search)
    
    grid_lda.fit(X_train, y_train)
    grid_svm.fit(X_train, y_train)
    grid_nb.fit(X_train, y_train)
    
    clf_lda = grid_lda.best_estimator_
    clf_svm = grid_svm.best_estimator_
    clf_xgb = XGBClassifier(objective='multi:softmax')
    clf_nb = grid_nb.best_estimator_
    
    all_bal_accs, all_accs, all_f1w, all_f1m = list(), list(), list(), list()
    cv_eval = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_repeats, random_state=17)  # for evaluative CV
    metrics = dict()
    
    scoring_metrics = ['balanced_accuracy', 'accuracy', 'f1_weighted', 'f1_macro']
    
    for metric in scoring_metrics:
        scores_svm = cross_val_score(clf_svm, X, y, cv=cv_eval, scoring=metric)
        scores_lda = cross_val_score(clf_lda, X, y, cv=cv_eval, scoring=metric)
        scores_xgb = cross_val_score(clf_xgb, X, y, cv=cv_eval, scoring=metric)
        scores_nb = cross_val_score(clf_nb, X, y, cv=cv_eval, scoring=metrics)
        
        metrics['SVM_' + metric] = scores_svm.mean()
        metrics['LDA_' + metric] = scores_lda.mean()
        metrics['XGB_' + metric] = scores_xgb.mean()
        metrics['NB_' + metrics] = scores_nb.mean()
        
        if metric == 'balanced_accuracy':
            plotting_list.append([ds, scores_svm.mean(), 'SVM'])
            plotting_list.append([ds, scores_lda.mean(), 'LDA'])
            plotting_list.append([ds, scores_xgb.mean(), 'XGB'])
    
    return metrics