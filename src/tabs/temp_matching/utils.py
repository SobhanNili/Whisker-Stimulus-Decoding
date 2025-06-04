import numpy as np
from enum import Enum
from scipy.spatial.distance import cosine,euclidean
from sklearn.metrics import brier_score_loss,f1_score
from scipy.ndimage import gaussian_filter1d
from scipy.signal import ellip, sosfiltfilt, zpk2sos,find_peaks
from pymatreader import read_mat
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import os.path
import gdown.download
import pickle
# helper funcs

def normalize_mat_by_row(mat:np.ndarray):
    return (mat.T/mat.sum(axis=1)).T

def modified_cosine(a,b):
    if np.all(a==0) and np.all(b==0):
        return 0
    if np.all(a==0) ^ np.all(b==0):
        return 1
    return cosine(a,b)

def calc_logits(temp_distance:np.ndarray) -> np.ndarray:
    d_nostim = temp_distance[0]
    d_stim = np.min(temp_distance[1:])
    logits = np.array([-d_nostim, -d_stim])
    return logits

def confidence_calc_from_distance(temp_distance:np.ndarray,T:float)->float:
    logits = calc_logits(temp_distance)
    exps = np.exp((logits - np.max(logits))/T)
    probs = exps / np.sum(exps)
    return probs[1]
    # return np.exp(-d_stim/T)/(np.exp(-d_stim/T)+np.exp(-d_nostim/T))

class TemplateMatching:
    def __init__(self,templates:np.ndarray,similarity_metric):
        # templates: 1st axis: temp_idx, 2nd axis: neuron_idx
        self.templates = templates
        self.similarity_metric = similarity_metric 

    def decode(self,population_vec):
        templates_count = self.templates.shape[0]
        projection_vals = np.zeros(shape=templates_count)
        # TODO: NORMALIZATION?!
        for template_idx in range(templates_count):
            projection_vals[template_idx] = self.similarity_metric(self.templates[template_idx,:],population_vec)
        return np.argmin(projection_vals)
    
    def decode_soft(self,population_vec)->np.ndarray:
        templates_count = self.templates.shape[0]
        projection_vals = np.zeros(shape=templates_count)
        for template_idx in range(templates_count):
            projection_vals[template_idx] = self.similarity_metric(self.templates[template_idx,:],population_vec)
        return projection_vals
    
def optimize_T(temprature_candidates:np.ndarray,trial_templates:np.ndarray,template_matcher:TemplateMatching,stims_binary):
    best_T = 1.0
    best_score = float('inf')
    trials_count = trial_templates.shape[0]
    soft_decode_result = np.zeros(trials_count)
    for T in temprature_candidates:
        for trial_idx in range(trials_count):
            sample_distances = template_matcher.decode_soft(trial_templates[trial_idx,:])
            soft_decode_result[trial_idx] = confidence_calc_from_distance(sample_distances,T)
        score = brier_score_loss(stims_binary, soft_decode_result)
        if score < best_score:
            best_score = score
            best_T = T
    return best_T

def get_spike_range(spike_train:np.ndarray,onset:float,bounds:tuple):
    aligned_spike_train = spike_train - onset
    time_filt = (aligned_spike_train >= bounds[0]) & (aligned_spike_train <= bounds[1])
    return aligned_spike_train[time_filt]

def get_psth(neruon_spike_times,trial_onsets,bin_size,time_bound,smooth=False):
    bin_edges = np.arange(start=time_bound[0],stop=time_bound[1],step=bin_size)
    hist = np.zeros(shape=len(bin_edges)-1)
    trial_onsets = trial_onsets if isinstance(trial_onsets,np.ndarray) or isinstance(trial_onsets,list) else [trial_onsets]
    trial_counts = len(trial_onsets) 
    for trial_idx in range(trial_counts):
        trial_spikes = get_spike_range(neruon_spike_times,trial_onsets[trial_idx],time_bound)
        hist += np.histogram(trial_spikes,bin_edges)[0]
    normalized_hist = hist / (trial_counts * bin_size)
    if smooth:
        return gaussian_filter1d(normalized_hist,sigma=5)
    return normalized_hist

def get_psth_sum(neurons_spike_times,trial_onsets,bin_size,time_bound,smooth=False):
    bin_edges = np.arange(start=time_bound[0],stop=time_bound[1],step=bin_size)
    trial_onsets = trial_onsets if isinstance(trial_onsets,np.ndarray) or isinstance(trial_onsets,list) else [trial_onsets]
    trial_counts = len(trial_onsets) 

    neurons_count = len(neurons_spike_times)
    normalized_hist_sum = np.zeros(shape=len(bin_edges)-1)

    for neruon_spike_times in neurons_spike_times:
        hist = np.zeros(shape=len(bin_edges)-1)
        for trial_idx in range(trial_counts):
            trial_spikes = get_spike_range(neruon_spike_times,trial_onsets[trial_idx],time_bound)
            hist += np.histogram(trial_spikes,bin_edges)[0]
        normalized_hist = hist / (trial_counts * bin_size)
        normalized_hist_sum += normalized_hist

    normalized_hist_sum /= neurons_count
    if smooth:
        return gaussian_filter1d(normalized_hist_sum,sigma=5)
    return normalized_hist_sum

class FeatureType(Enum):
    PEAK = 1,
    FULL = 2,
    FULL_NORMALIZED = 3,
    SUM = 4
    # SUM_NORMALIZED = 5
    

def feature_extractor(neuron_psth,bin_edges,feature_type:FeatureType,time_bound:tuple):
    time_filt = [True if bin_edges[i] >= time_bound[0] and bin_edges[i] <= time_bound[1] else False for i in range(len(neuron_psth))]
    match feature_type:
        case FeatureType.PEAK:
            # TODO: normalization by dividing by baseline?
            return neuron_psth[time_filt].max()
        case FeatureType.FULL_NORMALIZED:     
            time_filtered_psth = neuron_psth[time_filt]
            if time_filtered_psth.std() == 0:
                return np.zeros_like(time_filtered_psth)
            else:
                return (time_filtered_psth - time_filtered_psth.mean())/time_filtered_psth.std()
        case FeatureType.FULL:
            return neuron_psth[time_filt]
        case FeatureType.SUM:
            return neuron_psth[time_filt].sum()

def get_template(neural_activity,onsets,bin_size,time_bound,feature_type:FeatureType):
    neurons_count = len(neural_activity)
    bin_edges = np.arange(start=time_bound[0],stop=time_bound[1],step=bin_size) 
    template = []
    for neuron_idx in range(neurons_count):
        neuron_activity = neural_activity[neuron_idx]
        neuron_psth = get_psth(neuron_activity,onsets,bin_size,time_bound)
        extracted_feature = feature_extractor(neuron_psth,bin_edges,feature_type,time_bound)
        if isinstance(extracted_feature,np.ndarray):
            template += extracted_feature.tolist()
        elif isinstance(extracted_feature,list):
            template += extracted_feature
        else:
            template += [extracted_feature]
    return np.array(template)

def calc_session_lick_times(Lick_Data,Trial_Times,stimIndices):
    SR=1000; # Lick_Data sampling rate
    Gain_Thrs=1.5 # Gain to adjust the threshold
    merge_on=0.01 # time interval (s) between two lick event to merge
    Trial_pts=np.floor(Trial_Times*SR)
    Stim_Times = Trial_Times[stimIndices]
    Stim_pts=np.floor(Stim_Times*SR).astype(int)

    # Remove the coil stim artifacts
    Lick_Clean=Lick_Data
    dur=5 # TODO: only five ms?!

    for t in range(len(Stim_pts)):
        pt1=Stim_pts[t]
        pt2=min(len(Lick_Clean),pt1+dur)
        
        if pt1 > len(Lick_Clean):
            continue

        Val1=Lick_Clean[pt1]
        Val2=Lick_Clean[pt2]
        Delta_Vm=Val2-Val1
        
        in_val = np.arange(pt2 - pt1 + 1)
        in_val = (in_val/(pt2-pt1))*Delta_Vm
        in_val= in_val+Val1
            
        Lick_Clean[pt1:pt2+1]=in_val # array bounds

    # Subtract the median
    Lick_1 = Lick_Clean - np.median(Lick_Clean)

    # Filter design: Elliptic filter between 1 and 40 Hz
    Zs, Ps, K = ellip(4, 0.1, 20, [1, 40], btype='bandpass', fs=SR, output='zpk')
    SOS = zpk2sos(Zs, Ps, K)

    # Apply the filter
    Lick_1 = sosfiltfilt(SOS, Lick_1)

    # Square the signal
    Lick_2 = Lick_1 ** 2

    # Initialize Lick_baseline
    Lick_baseline = np.zeros((len(Trial_pts), 3))

    # Calculate baseline metrics for each trial
    for t, pt in enumerate(Trial_pts):
        pt1 = int(max(pt - 2 * SR, 0))
        pt2 = int(pt) - 1
        baseline_segment = Lick_2[pt1:pt2 + 1]  # Include pt2
        if len(baseline_segment) != 0:
            Lick_baseline[t, 0] = np.mean(baseline_segment)
            Lick_baseline[t, 1] = np.std(baseline_segment)
            Lick_baseline[t, 2] = np.max(baseline_segment)

    # Calculate threshold
    Thrs = np.mean(Lick_baseline[:, 0]) + Gain_Thrs * np.mean(Lick_baseline[:, 2])
    # Subtract threshold
    Lick_2 -= Thrs

    # Calculate Lick_Stat1
    Lick_Stat1 = Lick_2 / np.abs(Lick_2)
    Lick_Stat1 = (Lick_Stat1 + 1) / 2

    # Set first and last points to 0
    Lick_Stat1[:5] = 0
    Lick_Stat1[-5:] = 0

    # Calculate derivative and find peaks
    Lick_Stat1_der = np.diff(Lick_Stat1)
    Time_On, _ = find_peaks(Lick_Stat1_der, height=0.1)
    Lick_Stat1_der = -Lick_Stat1_der
    Time_Off, _ = find_peaks(Lick_Stat1_der, height=0.1)

    # Handle merging of activation periods
    merge_on_pts = int(merge_on * SR)

    if len(Time_On) > 0:
        Time_On = Time_On[1:]  # Remove first
    if len(Time_Off) > 0:
        Time_Off = Time_Off[:-1]  # Remove last

    Delta_Act = Time_On - Time_Off
    Lick_Stat2 = Lick_Stat1.copy()

    for int_idx in range(len(Delta_Act)):
        if Delta_Act[int_idx] < merge_on_pts:
            Lick_Stat2[Time_Off[int_idx]:Time_On[int_idx]] = 1
    
    return Lick_Stat2.astype(bool)

def load_data():
    # dowload data if not already
    drive_url = "https://drive.google.com/file/d/1lA4qiZn9uXAdUULGuZMiU8d-QgleZorA/view?usp=sharing"
    output_path = "paper_data.pkl"  # Adjust the file extension as per the actual file type
    file_id = drive_url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        print('Downloading Data For the First Time \n ------')
        gdown.download(download_url, output_path, quiet=False)
        print('Downloading Completed \n ------')

    with open(output_path, 'rb') as f:
        data_struct = pickle.load(f)
    return data_struct

def find_optimal_threshold(y_true, y_probs):
    thresholds = np.arange(0, 1.01, 0.01)  # Define thresholds from 0 to 1
    best_threshold = 0
    best_f1 = 0

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)  # Convert probabilities to binary predictions
        f1 = f1_score(y_true, y_pred)  # Calculate F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

def cont_calc(estim_timepoints, temp_matcher, neural_activity, BIN_SIZE, TIME_BOUND, FEATURE_TYPE,progress,n_jobs=-1):
    def cont_calc_iteration(estim_time, temp_matcher, neural_activity, BIN_SIZE, TIME_BOUND, FEATURE_TYPE):
        extracted_temp = get_template(neural_activity, estim_time, BIN_SIZE, TIME_BOUND, FEATURE_TYPE)
        return calc_logits(temp_matcher.decode_soft(extracted_temp))
    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        results = parallel(
            delayed(cont_calc_iteration)(estim_time, temp_matcher, neural_activity, BIN_SIZE, TIME_BOUND, FEATURE_TYPE)
            for estim_time in tqdm(estim_timepoints)
        )
    results = np.array(results)
    return results
