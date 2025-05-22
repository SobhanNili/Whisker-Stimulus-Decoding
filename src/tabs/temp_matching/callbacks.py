from ...utils.temp_matching import *
from .plots import *
import gradio as gr

PARALLEL_RUN = True


def update_neurons_count(data_struct,session_idx):
    return f"{(data_struct['trials'][session_idx]['brain_region'] == 'wS1').sum()} WS1 Neurons" if data_struct is not None else ''
def update_trial_type_count(data_struct,session_idx,trial_type):
    if data_struct is None:
        return ''
    trial_results = np.array(data_struct['trials'][session_idx]['result'])
    return f'{(trial_results == trial_type).sum()} Trials'
def toggle_confidence_slider(optimize_threshold):
    return gr.update(interactive=not optimize_threshold)
def run_analysis(data_struct,session_idx, similarity_metric, time_bound, multi_level,stim_temp_trial_type,nostim_temp_trial_type,bin_size,confidence_threshold,optimize_threshold,show_wrong_labels,feature_type,save_results_indicator,estim_bin_size):
    # input processing
    SESSION_IDX = session_idx
    SIMILIARITY_METRIC = modified_cosine if similarity_metric == 'cosine' else euclidean
    TIME_BOUND = (time_bound[0]/1000,time_bound[1]/1000)
    TWO_TEMP_MODE = not multi_level
    BIN_SIZE = bin_size/1000
    ESTIMATION_BIN = estim_bin_size/1000
    CONFIDENCE_THRESHOLD = confidence_threshold
    FEATURE_TYPE = FeatureType[feature_type]
    TEMPRATURE_CANDIDATES = [0.01,0.1,0.5,1,10]

    region_filt = data_struct['trials'][SESSION_IDX]['brain_region'] == 'wS1'

    trial_onsets = data_struct['trial_onset'][SESSION_IDX]
    neural_activity = np.array(data_struct['trials'][SESSION_IDX]['spikes'],dtype=object)[region_filt]
    stim_amps = data_struct['trials'][SESSION_IDX]['stim_amp']
    real_stims_binary = stim_amps.astype(bool)
    trial_results = np.array(data_struct['trials'][session_idx]['result'])

    template_stim_amps = [0,4] if TWO_TEMP_MODE else range(5)
    trial_filt = np.vstack([data_struct['trials'][SESSION_IDX]['stim_amp'] == i for i in template_stim_amps])
    temps_count = trial_filt.shape[0]
    stim_templates = []

    for temp_idx in range(temps_count):
        if temp_idx == 0:
            match nostim_temp_trial_type:
                case 'All':
                    filtered_onsets = trial_onsets[trial_filt[temp_idx,:]]
                case 'CR':
                    filtered_onsets = trial_onsets[trial_filt[temp_idx,:] & (trial_results=='CR')]
                case 'FA': 
                    filtered_onsets = trial_onsets[trial_filt[temp_idx,:] & (trial_results=='FA')]
        else:
            match stim_temp_trial_type:
                case 'All':
                    filtered_onsets = trial_onsets[trial_filt[temp_idx,:]]
                case 'Hit':
                    filtered_onsets = trial_onsets[trial_filt[temp_idx,:] & (trial_results=='hit')] 
                case 'Miss':
                    filtered_onsets = trial_onsets[trial_filt[temp_idx,:] & (trial_results=='miss')] 
        stim_templates.append(get_template(neural_activity,filtered_onsets,BIN_SIZE,TIME_BOUND,FEATURE_TYPE))
    stim_templates = np.array(stim_templates)

    temp_matcher = TemplateMatching(templates=stim_templates,similarity_metric=SIMILIARITY_METRIC)
    
    trial_templates = []
    trials_count = len(trial_onsets)
    for trial_idx in range(trials_count):
        trial_templates.append(get_template(neural_activity,[trial_onsets[trial_idx]],BIN_SIZE,TIME_BOUND,FEATURE_TYPE))
    trial_templates = np.array(trial_templates)

    T_optim = optimize_T(TEMPRATURE_CANDIDATES,trial_templates,temp_matcher,stim_amps.astype(bool)) # optimize temprature param

    soft_decode_result = np.zeros(trials_count)
    for trial_idx in range(trials_count):
        sample_distances = temp_matcher.decode_soft(trial_templates[trial_idx,:])
        soft_decode_result[trial_idx] = confidence_calc_from_distance(sample_distances,T_optim)
    if optimize_threshold:
        CONFIDENCE_THRESHOLD = find_optimal_threshold(real_stims_binary,soft_decode_result)
    hard_decode_result = soft_decode_result >= CONFIDENCE_THRESHOLD
    wrong_classification_indices = hard_decode_result != real_stims_binary

    output_filename = None
    if save_results_indicator:
        LICK_SR = 1000
        lick_indices = data_struct['trials'][SESSION_IDX]['lick_indices']
        lick_times = np.where(lick_indices)[0]/LICK_SR
        stim_trial_indices = data_struct['trials'][SESSION_IDX]['stim']

        session_time = trial_onsets.max() + 5
        estim_timepoints = np.arange(start=ESTIMATION_BIN,stop=session_time,step=ESTIMATION_BIN)
        cont_estimations = np.zeros_like(estim_timepoints)
        cont_estimations = cont_calc(estim_timepoints,temp_matcher,neural_activity,BIN_SIZE,TIME_BOUND,FEATURE_TYPE,T_optim,parallel_calc=PARALLEL_RUN,n_jobs=-1)

        digitized_licktimes = np.digitize(lick_times, estim_timepoints)
        # lick_response = np.bincount(digitized_licktimes, minlength=len(estim_timepoints) - 1) > LICK_RESPONSE_THRESHOLD
        lick_response = np.isin(np.arange(len(estim_timepoints) - 1), digitized_licktimes)

        digitized_trialtimes = np.digitize(trial_onsets[stim_trial_indices], estim_timepoints)
        stimulus_presence = np.isin(np.arange(len(estim_timepoints) - 1), digitized_trialtimes)

        output_filename = os.path.join('outputs',f'continuous_perception.npz')
        np.savez(output_filename,perception=cont_estimations[:-1],timepoints=estim_timepoints[:-1],lick_response=lick_response,stim_presence=stimulus_presence)


    # gen outputs
    temp_plot = generate_psth_plot(stim_amps,template_stim_amps,neural_activity,trial_onsets,TIME_BOUND,BIN_SIZE)
    roc_plot = generate_roc_plot(real_stims_binary, soft_decode_result,CONFIDENCE_THRESHOLD)
    temp_distance_plot = generate_temp_distance_plot(stim_templates,SIMILIARITY_METRIC)
    temp_clustering_plot = generate_clustering_plot(trial_templates,stim_amps,SIMILIARITY_METRIC,stim_templates,template_stim_amps,wrong_classification_indices if show_wrong_labels else None)
    confmat_plot = plot_confusion_matrix(stim_amps,hard_decode_result)
    behavior_corr_plot_output = generate_behavior_corr_plot(data_struct['trials'][SESSION_IDX]['result'],hard_decode_result)
    perception_overtime_plot_output = generate_softscore_overtime_plot(soft_decode_result,stim_amps)
    perception_hist_plot_output = generate_softscore_hist_plot(soft_decode_result,stim_amps)
    f1_score_val = f'{f1_score(real_stims_binary,hard_decode_result):0.2f}'
    return temp_plot,gr.update(value=temp_distance_plot, visible=multi_level),temp_clustering_plot,roc_plot,confmat_plot,behavior_corr_plot_output,perception_overtime_plot_output,perception_hist_plot_output,f1_score_val,output_filename
