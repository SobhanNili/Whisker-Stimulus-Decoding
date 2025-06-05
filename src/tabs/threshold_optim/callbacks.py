import gradio as gr
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os.path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim

# Params
RESPONSE_WINDOW = 0.5           # seconds after stimulus to expect perception threshold crossing
SAFE_WINDOW = RESPONSE_WINDOW   # seconds before perception threshold crossing to check for stimulus
PERCEPT_DELAY = 1               # seconds before lick to expect perception threshold crossing
LICK_WINDOW = PERCEPT_DELAY     # seconds after perception threshold crossing to expect lick
K = 10.0                        # steepness for sigmoid approximation

# def threshold_optimizer_for_decoder(stim_presence, perception, timepoints, beta, detection_window=DETECTION_WINDOW): # beta: how much more penalize missing stim over hallucinating stim
#     possible_thresholds = np.arange(0, 1, 0.01)
#     stim_presence_times = timepoints[stim_presence]

#     def compute_decoder_loss(threshold):
#         # Determine perception crosses and subthresholds
#         perception_cross_mask = perception >= threshold
#         perception_subthreshold_mask = ~perception_cross_mask

#         # Window definitions
#         cross_window_starts = timepoints[perception_cross_mask] - detection_window
#         cross_window_ends = timepoints[perception_cross_mask]

#         subthreshold_window_starts = timepoints[perception_subthreshold_mask] - detection_window
#         subthreshold_window_ends = timepoints[perception_subthreshold_mask]

#         # Calculate false stim counts
#         false_stim_count = np.sum([
#             not np.any((stim_presence_times >= ws) & (stim_presence_times <= we))
#             for ws, we in zip(cross_window_starts, cross_window_ends)
#         ])
#         true_stim_count = len(cross_window_starts) - false_stim_count # maybe replace len(cross_window_starts) with number of total stimulus

#         # Calculate false no-stim counts
#         false_nostim_count = np.sum([
#             np.any((stim_presence_times >= ws) & (stim_presence_times <= we))
#             for ws, we in zip(subthreshold_window_starts, subthreshold_window_ends)
#         ])
#         true_nostim_count = len(subthreshold_window_starts) - false_nostim_count


#         # Loss calculation
#         fp = false_stim_count; fn = false_nostim_count
#         N_neg = false_stim_count + true_nostim_count; N_pos = true_stim_count + false_nostim_count
        
#         normalized_fp = fp / N_neg if N_neg > 0 else 0
#         normalized_fn = fn / N_pos if N_pos > 0 else 0

#         return beta * normalized_fn + normalized_fp

#     # Parallel processing over thresholds
#     losses = Parallel(n_jobs=-1, backend="loky")(
#         delayed(compute_decoder_loss)(threshold) for threshold in possible_thresholds
#     )

#     # Optimal threshold selection
#     optimal_threshold = possible_thresholds[np.argmin(losses)]
#     return optimal_threshold

# def threshold_optimizer_for_behavior(lick_response, perception, timepoints, beta=0.5, response_window=RESPONSE_WINDOW): # beta: how much more penalize not predicting lick over hallucinating nessicity of lick
#     possible_thresholds = np.arange(0, 1, 0.01)
#     lick_times = timepoints[lick_response]

#     def compute_behavior_loss(threshold):
#         # Identify perception crosses
#         perception_cross_mask = perception >= threshold
#         perception_cross_times = timepoints[perception_cross_mask]

#         # Window definitions
#         cross_window_starts = perception_cross_times
#         cross_window_ends = perception_cross_times + response_window

#         lick_window_starts = lick_times - response_window
#         lick_window_ends = lick_times

#         # Wrongly high perception: perception above threshold but no lick in window
#         false_high_perception_count = np.sum([
#             not np.any((lick_times > ws) & (lick_times <= we))
#             for ws, we in zip(cross_window_starts, cross_window_ends)
#         ])
#         true_high_perception_count = len(cross_window_starts) - false_high_perception_count

#         # Wrongly low perception: lick but no perception crossing in preceding window
#         false_low_perception_count = np.sum([
#             not np.any((perception_cross_times >= ws) & (perception_cross_times < we))
#             for ws, we in zip(lick_window_starts, lick_window_ends)
#         ])
#         true_low_perception_count = len(lick_window_starts) - false_low_perception_count

#         # Loss calculation
#         fp = false_high_perception_count; fn = false_low_perception_count
#         N_neg = false_high_perception_count + true_low_perception_count; N_pos = true_high_perception_count + false_low_perception_count
        
#         normalized_fp = fp / N_neg if N_neg > 0 else 0
#         normalized_fn = fn / N_pos if N_pos > 0 else 0

#         return beta * normalized_fn + normalized_fp

#     # Parallelized computation of losses
#     losses = Parallel(n_jobs=-1, backend="loky")(
#         delayed(compute_behavior_loss)(threshold) for threshold in possible_thresholds
#     )

#     # Determine optimal threshold
#     optimal_threshold = possible_thresholds[np.argmin(losses)]
#     return optimal_threshold

# def calc_optim_thresholds(session_data,sliding_window_length,sliding_window_sep,decoding_beta,behavior_beta,progress):
#     timefilt_start = np.arange(start=0,stop=session_data['timepoints'][-1]-sliding_window_length,step=sliding_window_sep)
#     timefilt_end = timefilt_start + sliding_window_length

#     full_timepoints = session_data['timepoints']
#     full_stim_presence = session_data['stim_presence']
#     full_perception = session_data['perception']
#     full_lick_response = session_data['lick_response']

#     optimized_behavior_thresholds = np.zeros_like(timefilt_start)
#     optimized_decoder_thresholds = np.zeros_like(timefilt_start)
#     for idx in progress.tqdm(range(len(timefilt_start))):
#         timefilt = (full_timepoints >= timefilt_start[idx]) & (full_timepoints <= timefilt_end[idx])
#         stim_presence = full_stim_presence[timefilt]
#         perception = full_perception[timefilt]
#         timepoints = full_timepoints[timefilt]
#         lick_response = full_lick_response[timefilt]
#         optimized_decoder_thresholds[idx] = threshold_optimizer_for_decoder(stim_presence,perception,timepoints,beta=decoding_beta)
#         optimized_behavior_thresholds[idx] = threshold_optimizer_for_behavior(lick_response,perception,timepoints,beta=behavior_beta)
#     threshold_estim_timepoints = (timefilt_start + timefilt_end)/2

#     return optimized_decoder_thresholds,optimized_behavior_thresholds,threshold_estim_timepoints

#%% utils
def optimize_parameters(loss_fn, n_epochs=600, lr=0.01):
    """Optimize T and theta for given loss function"""
    alpha = torch.tensor(0.0, requires_grad=True)  # T = exp(alpha)
    beta = torch.tensor(0.0, requires_grad=True)   # theta = sigmoid(beta)
    
    optimizer = optim.Adam([alpha, beta], lr=lr)
    
    for epoch in range(n_epochs):
        T = torch.exp(alpha)
        theta = torch.sigmoid(beta)
        
        optimizer.zero_grad()
        loss = loss_fn(T, theta)
        loss.backward()
        optimizer.step()
    
    return T.item(), theta.item()

def calc_optimal_thresh_and_temp(timepoints:torch.tensor, logits:torch.tensor, stim_presence:torch.tensor, lick_response:torch.tensor, lambda_fa, lambda_up):

    dt = (timepoints[1] - timepoints[0]).item()

    def create_mask_from_events(event_tensor, window_start, window_end):
        event_indices = torch.where(event_tensor)[0]  # indices of True events
        if len(event_indices) == 0:
            return torch.zeros_like(event_tensor, dtype=torch.bool)

        start_offset = int(window_start / dt)
        end_offset = int(window_end / dt)
        window_range = torch.arange(start_offset, end_offset + 1)

        expanded_indices = event_indices.unsqueeze(1) + window_range.unsqueeze(0)  # shape (num_events, window_length)
        expanded_indices = expanded_indices.clamp(min=0, max=len(event_tensor)-1)
        flat_indices = expanded_indices.view(-1).unique()

        mask = torch.zeros_like(event_tensor, dtype=torch.bool,requires_grad=False)
        mask[flat_indices] = True

        return mask

    safe_stim_mask = create_mask_from_events(stim_presence, -SAFE_WINDOW, 0)
    future_lick_mask = create_mask_from_events(lick_response, 0, LICK_WINDOW)
    stim_indices = torch.where(stim_presence)[0]
    lick_indices = torch.where(lick_response)[0]

    def loss_fn_decoding(T, theta):
        s = F.softmax(logits.detach() / T, dim=-1)[..., 0]
        percept_indicators = torch.sigmoid(K * (s - theta))

        if len(stim_indices) > 0:
            # Create window offsets (in indices)
            window_len = int(RESPONSE_WINDOW / dt) + 1
            window_offsets = torch.arange(window_len)

            windows_idx = stim_indices.unsqueeze(1) + window_offsets.unsqueeze(0)
            windows_idx = windows_idx.clamp(max=len(s) - 1)
            windows = percept_indicators[windows_idx]

            # Compute max per window, then average the miss losses
            max_vals, _ = torch.max(windows, dim=1)
            miss_loss = torch.mean(1 - max_vals)
        else:
            miss_loss = torch.tensor(0.0, device=s.device)

        false_alarms = percept_indicators * (~safe_stim_mask).float()
        fa_loss = torch.sum(false_alarms) / torch.sum(~safe_stim_mask)

        return miss_loss + lambda_fa * fa_loss

    def loss_fn_behavior(T, theta):
        s = F.softmax(logits.detach() / T, dim=-1)[..., 0]
        percept_indicators = torch.sigmoid(K * (s - theta))

        if len(lick_indices) > 0:
            window_len = int(PERCEPT_DELAY / dt) + 1
            window_offsets = torch.arange(-window_len + 1, 1)
            windows_idx = lick_indices.unsqueeze(1) + window_offsets.unsqueeze(0)
            windows_idx = windows_idx.clamp(min=0, max=len(s) - 1)
            windows = percept_indicators[windows_idx]

            # Compute max per window, then average the fa_lick_loss
            max_vals, _ = torch.max(windows, dim=1)
            fa_lick_loss = torch.mean(1 - max_vals)
        else:
            fa_lick_loss = torch.tensor(0.0, device=s.device)

        no_lick_percepts = percept_indicators * (~future_lick_mask).float()
        no_lick_loss = torch.sum(no_lick_percepts) / torch.sum(~future_lick_mask)

        return fa_lick_loss + lambda_up * no_lick_loss

    T_dec_soft, theta_dec_soft = optimize_parameters(loss_fn_decoding)
    T_beh_soft, theta_beh_soft = optimize_parameters(loss_fn_behavior)

    return T_dec_soft, theta_dec_soft, T_beh_soft, theta_beh_soft

def calc_optim_params_per_session(session_data, sliding_window_length, sliding_window_sep, decoding_beta, behavior_beta, progress): 
    timefilt_start = np.arange(start=0, stop=session_data['timepoints'][-1] - sliding_window_length, step=sliding_window_sep)
    timefilt_end = timefilt_start + sliding_window_length

    full_timepoints = torch.tensor(session_data['timepoints'], dtype=torch.float32,requires_grad=False)
    full_stim_presence = torch.tensor(session_data['stim_presence'], dtype=torch.bool,requires_grad=False)
    full_logits = torch.tensor(session_data['logits'], dtype=torch.float32,requires_grad=False)
    full_lick_response = torch.tensor(session_data['lick_response'], dtype=torch.bool,requires_grad=False)

    optimized_decoder_temps = np.zeros_like(timefilt_start)
    optimized_decoder_thresholds = np.zeros_like(timefilt_start)
    optimized_behavior_temps = np.zeros_like(timefilt_start)
    optimized_behavior_thresholds = np.zeros_like(timefilt_start)

    iterator = tqdm(range(len(timefilt_start))) if progress is None else progress.tqdm(range(len(timefilt_start)))

    for idx in iterator:
        timefilt = (full_timepoints >= timefilt_start[idx]) & (full_timepoints <= timefilt_end[idx])
        stim_presence = full_stim_presence[timefilt].detach()
        logits = full_logits[timefilt].detach()
        timepoints = full_timepoints[timefilt].detach()
        lick_response = full_lick_response[timefilt].detach()

        optimized_decoder_temps[idx], optimized_decoder_thresholds[idx], optimized_behavior_temps[idx], optimized_behavior_thresholds[idx] = \
            calc_optimal_thresh_and_temp(timepoints, logits, stim_presence, lick_response, lambda_fa=1/decoding_beta, lambda_up=1/behavior_beta)

    estim_timepoints = (timefilt_start + timefilt_end) / 2

    return optimized_decoder_temps, optimized_decoder_thresholds, optimized_behavior_temps, optimized_behavior_thresholds, estim_timepoints

def calc_perception(logits:np.ndarray,T:np.ndarray):
    scaled_logits = np.zeros_like(logits)
    scaled_logits[:,0] = logits[:,0]/T
    scaled_logits[:,1] = logits[:,1]/T

    exps = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
    softmax = exps / np.sum(exps, axis=-1, keepdims=True)

    return softmax[..., 0]
#%% plotting

def calc_perception_with_behavior_correspondance(perception):
    timepoints = session_data['timepoints']
    decoder_thresold = decoder_thresholds
    lick_response = session_data['lick_response']

    lick_times = timepoints[lick_response]

    # Identify perception crosses
    perception_cross_mask = perception >= decoder_thresold
    perception_cross_times = timepoints[perception_cross_mask]

    # Window definitions
    cross_window_starts = perception_cross_times
    cross_window_ends = perception_cross_times + RESPONSE_WINDOW

    lick_window_starts = lick_times - RESPONSE_WINDOW
    lick_window_ends = lick_times

    # Wrongly high perception: perception above threshold but no lick in window
    false_high_perception_count = 0
    true_high_perception_count = 0
    for ws, we in zip(cross_window_starts, cross_window_ends):
        if not np.any((lick_times > ws) & (lick_times <= we)):
            false_high_perception_count += 1
        else:
            true_high_perception_count += 1


    # Wrongly low perception: lick but no perception crossing in preceding window
    false_low_perception_count = 0
    true_low_perception_count= 0
    for ws, we in zip(lick_window_starts, lick_window_ends):
        if not np.any((perception_cross_times >= ws) & (perception_cross_times < we)):
            false_low_perception_count += 1
        else:
            true_low_perception_count += 1

    results = (false_low_perception_count/len(lick_times),
               false_high_perception_count/len(perception_cross_times))

    return [gr.update(value=f'{results[i]:0.2f}') for i in range(2)]


def plot_perception_dist_based_on_behavior(): # if the wrong behavior is a result of perception, or perception could've prevented it
    # find windows before a wrong lick
    # find windows before a non-detected stim
    timepoints = session_data['timepoints']
    lick_response = session_data['lick_response']
    stim_presence = session_data['stim_presence']

    lick_times = timepoints[lick_response]
    stim_times = timepoints[stim_presence]

    lick_window_starts = lick_times - 1 # time to get reward
    lick_window_ends = lick_times

    stim_window_starts = stim_times
    stim_window_ends = stim_times + 0.5 # time that the stim data is still preserved

    fig = plt.figure(dpi=300)
    perception = behavior_perception
    wrong_lick_windows_dist = [] # fa
    true_lick_window_dist = [] # hit
    for ws, we in zip(lick_window_starts, lick_window_ends):
        window_filt = (timepoints >= ws) & (timepoints <= we)
        perception_window = perception[window_filt]
        window_stat = np.max(perception_window)
        if not np.any((stim_times > ws) & (stim_times <= we)):
            wrong_lick_windows_dist.append(window_stat)
        else:
            true_lick_window_dist.append(window_stat)
    
    wrong_nolick_windows_dist = [] # miss
    true_nolick_windows_dist = [] # cr
    for ws, we in zip(stim_window_starts, stim_window_ends):
        window_filt = (timepoints >= ws) & (timepoints <= we)
        perception_window = perception[window_filt]
        window_stat = np.max(perception_window)
        if not np.any((lick_times > ws) & (lick_times <= we)):
            wrong_nolick_windows_dist.append(window_stat)
        else:
            true_nolick_windows_dist.append(window_stat)

    sns.kdeplot(wrong_lick_windows_dist, label="Before an extra lick (FA)")
    sns.kdeplot(true_lick_window_dist, label="Before a correct lick (Hit)")
    sns.kdeplot(wrong_nolick_windows_dist, label="After missing a stimulus (Miss)")
    sns.kdeplot(true_nolick_windows_dist, label="Before not licking in catch trials (CR)")


    plt.title("Behaviorally-Optimized Perception")
    plt.legend(loc='upper left')
    plt.xlabel("Max Perception in Window")
    plt.ylabel("Density")
    plt.grid(True) 
        
    plt.suptitle("KDE of Max Perception Surronding Different Behavioral Windows",fontweight='bold')

    return fig

def plot_threshold_temp_evo(sliding_window_length,sliding_window_sep,decoding_beta,behavior_beta,progress):
    global behavior_perception,decoder_perception,decoder_temps,decoder_thresholds,behavior_temps,behavior_thresholds

    optim_decoder_temps, optim_decoder_thresholds, optim_behavior_temps, optim_behavior_thresholds, estim_timepoints = calc_optim_params_per_session(session_data,sliding_window_length,sliding_window_sep,decoding_beta,behavior_beta,progress)
    # Fit lines to the data
    decoder_temp_fit = np.polyfit(estim_timepoints, optim_decoder_temps, 1)
    decoder_threshold_fit = np.polyfit(estim_timepoints, optim_decoder_thresholds, 1)
    behavior_temp_fit = np.polyfit(estim_timepoints, optim_behavior_temps, 1)
    behavior_threshold_fit = np.polyfit(estim_timepoints, optim_behavior_thresholds, 1)

    # Generate fitted lines
    decoder_temp_line = np.polyval(decoder_temp_fit, estim_timepoints)
    decoder_threshold_line = np.polyval(decoder_threshold_fit, estim_timepoints)
    behavior_temp_line = np.polyval(behavior_temp_fit, estim_timepoints)
    behavior_threshold_line = np.polyval(behavior_threshold_fit, estim_timepoints)

    # For saving to file
    decoder_temps = np.polyval(decoder_temp_fit, session_data['timepoints'])
    decoder_thresholds = np.polyval(decoder_threshold_fit, session_data['timepoints'])
    behavior_temps = np.polyval(behavior_temp_fit, session_data['timepoints'])
    behavior_thresholds = np.polyval(behavior_threshold_fit, session_data['timepoints'])
    decoder_perception = calc_perception(session_data['logits'],decoder_temps)
    behavior_perception = calc_perception(session_data['logits'],behavior_temps)

    fig, axes = plt.subplots(2, 2, figsize=(12,12), dpi=300)
    axes = axes.flatten()

    # Plot for optimized decoder thresholds
    axes[0].plot(estim_timepoints, optim_decoder_thresholds, label='Threshold', color='blue')
    axes[0].plot(estim_timepoints, decoder_threshold_line, label=f'Linear Fit (slope={decoder_threshold_fit[0]*60:.2f}) per min', color='red', linestyle='--')
    axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    axes[0].set_ylim([0, 1])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Optimum Threshold')
    axes[0].set_title('Decoding Matching Optimization')
    axes[0].legend()

    # Plot for optimized behavior thresholds
    axes[1].plot(estim_timepoints, optim_behavior_thresholds, label='Threshold', color='green')
    axes[1].plot(estim_timepoints, behavior_threshold_line, label=f'Linear Fit (slope={behavior_threshold_fit[0]*60:.2f}) per min', color='orange', linestyle='--')
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    axes[1].set_ylim([0, 1])
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Optimum Threshold')
    axes[1].set_title('Behavior Matching Optimization')
    axes[1].legend()

    # Plot for optimized decoder temps
    axes[2].plot(estim_timepoints, optim_decoder_temps, label='Temperature', color='blue')
    axes[2].plot(estim_timepoints, decoder_temp_line, label=f'Linear Fit (slope={decoder_temp_fit[0]*60:.2f}) per min', color='red', linestyle='--')
    axes[2].xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Optimum Temprature')
    axes[2].set_title('Decoding Matching Optimization')
    axes[2].legend()

    # Plot for optimized behavior thresholds
    axes[3].plot(estim_timepoints, optim_behavior_temps, label='Temperature', color='green')
    axes[3].plot(estim_timepoints, behavior_temp_line, label=f'Linear Fit (slope={behavior_temp_fit[0]*60:.2f}) per min', color='orange', linestyle='--')
    axes[3].xaxis.set_major_formatter(ticker.FuncFormatter(format_time))
    axes[3].set_xlabel('Time')
    axes[3].set_ylabel('Optimum Temprature')
    axes[3].set_title('Behavior Matching Optimization')
    axes[3].legend()

    # axes[2].get_shared_y_axes().join(axes[2], axes[3])

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.close()
    return fig

def save_results2file():
        output_filename = os.path.join('outputs',f'perception.npz')
        np.savez(output_filename,behavior_perception=behavior_perception,decoder_perception=decoder_perception,timepoints=session_data['timepoints'],lick_response=session_data['lick_response'],stim_presence=session_data['stim_presence'],
                 decoder_thresholds=decoder_thresholds,behavior_thresholds=behavior_thresholds,stim_amps=session_data['stim_amps'])
        return output_filename

session_data = None
# mapped_decoder_thresholds = None; mapped_behavior_thresholds = None
behavior_perception = None; decoder_perception = None; decoder_temps = None; decoder_thresholds = None; behavior_temps = None; behavior_thresholds = None
def load_file(file):
    global session_data
    if file is None:
        return gr.update(visible=False)
    else:
        session_data = np.load(file.name)
        return gr.update(visible=True)
    
def run_analysis(sliding_window_length,sliding_window_sep,decoding_beta,behavior_beta,progress=gr.Progress()):
    return plot_threshold_temp_evo(sliding_window_length,sliding_window_sep,decoding_beta,behavior_beta,progress),plot_perception_dist_based_on_behavior(),gr.update(interactive=True),*calc_perception_with_behavior_correspondance(decoder_perception),*calc_perception_with_behavior_correspondance(behavior_perception)

def format_time(secs,pos):
    minutes = int(secs // 60)
    seconds = int(secs % 60)
    return f"{minutes}:{seconds:02d}"