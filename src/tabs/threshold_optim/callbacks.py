import gradio as gr
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os.path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def threshold_optimizer_for_decoder(stim_presence, perception, timepoints, beta, DETECTION_WINDOW=0.5): # beta: how much more penalize missing stim over hallucinating stim
    possible_thresholds = np.arange(0, 1, 0.01)
    stim_presence_times = timepoints[stim_presence]

    def compute_decoder_loss(threshold):
        # Determine perception crosses and subthresholds
        perception_cross_mask = perception >= threshold
        perception_subthreshold_mask = ~perception_cross_mask

        # Window definitions
        cross_window_starts = timepoints[perception_cross_mask] - DETECTION_WINDOW
        cross_window_ends = timepoints[perception_cross_mask]

        subthreshold_window_starts = timepoints[perception_subthreshold_mask] - DETECTION_WINDOW
        subthreshold_window_ends = timepoints[perception_subthreshold_mask]

        # Calculate false stim counts
        false_stim_count = np.sum([
            not np.any((stim_presence_times >= ws) & (stim_presence_times <= we))
            for ws, we in zip(cross_window_starts, cross_window_ends)
        ])
        true_stim_count = len(cross_window_starts) - false_stim_count

        # Calculate false no-stim counts
        false_nostim_count = np.sum([
            np.any((stim_presence_times >= ws) & (stim_presence_times <= we))
            for ws, we in zip(subthreshold_window_starts, subthreshold_window_ends)
        ])
        true_nostim_count = len(subthreshold_window_starts) - false_nostim_count


        # Loss calculation
        fp = false_stim_count; fn = false_nostim_count
        N_neg = false_stim_count + true_nostim_count; N_pos = true_stim_count + false_nostim_count
        
        normalized_fp = fp / N_neg if N_neg > 0 else 0
        normalized_fn = fn / N_pos if N_pos > 0 else 0

        return beta * normalized_fn + normalized_fp

    # Parallel processing over thresholds
    losses = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_decoder_loss)(threshold) for threshold in possible_thresholds
    )

    # Optimal threshold selection
    optimal_threshold = possible_thresholds[np.argmin(losses)]
    return optimal_threshold

def threshold_optimizer_for_behavior(lick_response, perception, timepoints, beta=0.5, RESPONSE_WINDOW=0.5): # beta: how much more penalize not predicting lick over hallucinating nessicity of lick
    possible_thresholds = np.arange(0, 1, 0.01)
    lick_times = timepoints[lick_response]

    def compute_behavior_loss(threshold):
        # Identify perception crosses
        perception_cross_mask = perception >= threshold
        perception_cross_times = timepoints[perception_cross_mask]

        # Window definitions
        cross_window_starts = perception_cross_times
        cross_window_ends = perception_cross_times + RESPONSE_WINDOW

        lick_window_starts = lick_times - RESPONSE_WINDOW
        lick_window_ends = lick_times

        # Wrongly high perception: perception above threshold but no lick in window
        false_high_perception_count = np.sum([
            not np.any((lick_times > ws) & (lick_times <= we))
            for ws, we in zip(cross_window_starts, cross_window_ends)
        ])
        true_high_perception_count = len(cross_window_starts) - false_high_perception_count

        # Wrongly low perception: lick but no perception crossing in preceding window
        false_low_perception_count = np.sum([
            not np.any((perception_cross_times >= ws) & (perception_cross_times < we))
            for ws, we in zip(lick_window_starts, lick_window_ends)
        ])
        true_low_perception_count = len(lick_window_starts) - false_low_perception_count

        # Loss calculation
        fp = false_high_perception_count; fn = false_low_perception_count
        N_neg = false_high_perception_count + true_low_perception_count; N_pos = true_high_perception_count + false_low_perception_count
        
        normalized_fp = fp / N_neg if N_neg > 0 else 0
        normalized_fn = fn / N_pos if N_pos > 0 else 0

        return beta * normalized_fn + normalized_fp

    # Parallelized computation of losses
    losses = Parallel(n_jobs=-1, backend="loky")(
        delayed(compute_behavior_loss)(threshold) for threshold in possible_thresholds
    )

    # Determine optimal threshold
    optimal_threshold = possible_thresholds[np.argmin(losses)]
    return optimal_threshold

def calc_optim_thresholds(session_data,sliding_window_length,sliding_window_sep):
    global mapped_decoder_thresholds,mapped_behavior_thresholds

    timefilt_start = np.arange(start=0,stop=session_data['timepoints'][-1]-sliding_window_length,step=sliding_window_sep)
    timefilt_end = timefilt_start + sliding_window_length

    full_timepoints = session_data['timepoints']
    full_stim_presence = session_data['stim_presence']
    full_perception = session_data['perception']
    full_lick_response = session_data['lick_response']

    optimized_behavior_thresholds = np.zeros_like(timefilt_start)
    optimized_decoder_thresholds = np.zeros_like(timefilt_start)
    for idx in tqdm(range(len(timefilt_start))):
        timefilt = (full_timepoints >= timefilt_start[idx]) & (full_timepoints <= timefilt_end[idx])
        stim_presence = full_stim_presence[timefilt]
        perception = full_perception[timefilt]
        timepoints = full_timepoints[timefilt]
        lick_response = full_lick_response[timefilt]
        optimized_decoder_thresholds[idx] = threshold_optimizer_for_decoder(stim_presence,perception,timepoints,beta=2)
        optimized_behavior_thresholds[idx] = threshold_optimizer_for_behavior(lick_response,perception,timepoints,beta=2)
    threshold_estim_timepoints = (timefilt_start + timefilt_end)/2


    mapped_decoder_thresholds = np.zeros_like(full_timepoints, dtype=float)
    mapped_behavior_thresholds = np.zeros_like(full_timepoints, dtype=float)

    # Iterate over each window and assign the corresponding threshold
    for idx in range(len(timefilt_start)):
        # Create a mask for timepoints within the current window
        time_mask = (full_timepoints >= timefilt_start[idx]) & (full_timepoints <= timefilt_end[idx])
        
        # Assign the thresholds to the corresponding timepoints
        mapped_decoder_thresholds[time_mask] = optimized_decoder_thresholds[idx]
        mapped_behavior_thresholds[time_mask] = optimized_behavior_thresholds[idx]

    return optimized_decoder_thresholds,optimized_behavior_thresholds,threshold_estim_timepoints

def plot_threshold_evo(sliding_window_length,sliding_window_sep):
    optimized_decoder_thresholds,optimized_behavior_thresholds,threshold_estim_timepoints = calc_optim_thresholds(session_data,sliding_window_length,sliding_window_sep)
    # Fit lines to the data
    decoder_fit = np.polyfit(threshold_estim_timepoints, optimized_decoder_thresholds, 1)
    behavior_fit = np.polyfit(threshold_estim_timepoints, optimized_behavior_thresholds, 1)

    # Generate fitted lines
    decoder_line = np.polyval(decoder_fit, threshold_estim_timepoints)
    behavior_line = np.polyval(behavior_fit, threshold_estim_timepoints)

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

    # Plot for optimized decoder thresholds
    axes[0].plot(threshold_estim_timepoints, optimized_decoder_thresholds, label='Threshold', color='blue')
    axes[0].plot(threshold_estim_timepoints, decoder_line, label=f'Linear Fit (slope={decoder_fit[0]:.2f})', color='red', linestyle='--')
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    axes[0].set_ylim([0, 1])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Optimum Threshold')
    axes[0].set_title('Decoding Matching Optimization')
    axes[0].legend()

    # Plot for optimized behavior thresholds
    axes[1].plot(threshold_estim_timepoints, optimized_behavior_thresholds, label='Threshold', color='green')
    axes[1].plot(threshold_estim_timepoints, behavior_line, label=f'Linear Fit (slope={behavior_fit[0]:.2f})', color='orange', linestyle='--')
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    axes[1].set_ylim([0, 1])
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Optimum Threshold')
    axes[1].set_title('Behavior Matching Optimization')
    axes[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.close()
    return fig

def save_thresholds2file():
        output_filename = os.path.join('outputs',f'continuous_perception_with_thresholds.npz')
        np.savez(output_filename,perception=session_data['perception'],timepoints=session_data['timepoints'],lick_response=session_data['lick_response'],stim_presence=session_data['stim_presence'],
                 decoder_threshold=mapped_decoder_thresholds,behavior_threshold=mapped_behavior_thresholds)
        return output_filename

session_data = None
mapped_decoder_thresholds = None; mapped_behavior_thresholds = None
def load_file(file):
    global session_data
    if file is None:
        return gr.update(visible=False)
    else:
        session_data = np.load(file.name)
        return gr.update(visible=True)
    
def run_analysis(sliding_window_length,sliding_window_sep):
    return plot_threshold_evo(sliding_window_length,sliding_window_sep)

def format_time(secs,pos):
    minutes = int(secs // 60)
    seconds = int(secs % 60)
    return f"{minutes}:{seconds:02d}"