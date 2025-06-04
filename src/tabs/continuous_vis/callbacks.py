RESPONSE_WINDOW = 1
# FIXED_THRESHOLD = 0.6

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter

session_data = None
def load_file(file):
    global session_data
    if file is None:
        return gr.update(visible=False),gr.update(maximum=int(session_data['timepoints'][-1]))
    else:
        session_data = np.load(file.name)
        return gr.update(visible=True),gr.update(maximum=int(session_data['timepoints'][-1]))

def get_next_stim_time(current_value,interval):
    upcoming_stims_time = (session_data['timepoints'] > current_value + 2) & (session_data['stim_presence'] == True)
    if upcoming_stims_time.any:
        next_stim_time = session_data['timepoints'][upcoming_stims_time][0]
        return gr.update(value=int(max(0,next_stim_time-interval/10)))
    else:
        return gr.update(value=current_value)
    
def format_time(secs,pos):
    minutes = int(secs // 60)
    seconds = int(secs % 60)
    return f"{minutes}:{seconds:02d}"

def get_wrong_indices(timepoints,perception,lick_response,timefilt,threshold):
    timepoints = timepoints[timefilt]
    perception = perception[timefilt]
    lick_response = lick_response[timefilt]
    threshold = threshold[timefilt]

    time_res = timepoints[1]-timepoints[0]
    response_window_indices_len = int(RESPONSE_WINDOW // time_res)

    perception_cross_indices = np.where(perception >= threshold)[0]
    lick_indices = np.where(lick_response)[0]

    # perception and licking misalignment
    false_high_perception_times = [] # marked perception time that was expected to elicit a lick
    false_low_perception_times = [] # mark the RESPONSE_WINDOW before an unanticipated lick

    for idx in perception_cross_indices:
        window_end = timepoints[idx] + RESPONSE_WINDOW
        if not np.any((timepoints[lick_indices] > timepoints[idx]) & (timepoints[lick_indices] <= window_end)):
            false_high_perception_times.append(timepoints[idx])
    for idx in lick_indices:
        window_start = timepoints[idx] - RESPONSE_WINDOW
        if not np.any((timepoints[perception_cross_indices] >= window_start) & (timepoints[perception_cross_indices] < timepoints[idx])):
            false_low_perception_times += timepoints[max(0,idx-response_window_indices_len):idx].tolist()

    # output is bin start times
    return np.array(false_high_perception_times), np.array(false_low_perception_times)

def plot_data(start_time, interval, show_fr,threshold_type,show_smoothed):
    plotting_range = (start_time, min(start_time + interval, session_data['timepoints'][-1]))  # in sec
    timebin = session_data['timepoints'][1] - session_data['timepoints'][0]
    time_filt = (session_data['timepoints'] >= plotting_range[0]) & (session_data['timepoints'] <= plotting_range[1])
    stim_times = session_data['timepoints'][(session_data['stim_presence'] == True) & time_filt]
    lick_times = session_data['timepoints'][(session_data['lick_response'] == True) & time_filt]

    match threshold_type:
        case 'Match with Behavior': threshold = session_data['behavior_thresholds']; perception = session_data['behavior_perception']
        case 'Optimize Decoding': threshold = session_data['decoder_thresholds']; perception = session_data['decoder_perception']

    # Create subplots: one for perception and one for lick responses
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), dpi=200, sharex=True, gridspec_kw={'height_ratios': [20, 1]}
    )

    # Main plot: Perception
    window_length = int(1//timebin)
    polyorder = 3

    if show_smoothed:
        smoothed_perception = savgol_filter(perception[time_filt], window_length, polyorder)
        ax1.plot(session_data['timepoints'][time_filt], smoothed_perception, label='Smoothed Perception')
    else:
        ax1.plot(session_data['timepoints'][time_filt], perception[time_filt], label='Perception')
    ax1.vlines(x=stim_times, ymin=0, ymax=1, colors='green', linestyles='dashed', label='Stimulus Times')
    ax1.plot(session_data['timepoints'][time_filt], threshold[time_filt], linestyle='--', label='Threshold')

    if show_fr:
        false_high_perception_times, false_low_perception_times = get_wrong_indices(session_data['timepoints'],perception if not show_smoothed else smoothed_perception,session_data['lick_response'], time_filt, threshold)
        for i in range(0, len(false_high_perception_times)):
            ax1.fill_betweenx([0, 1], false_high_perception_times[i], false_high_perception_times[i] + timebin, color='pink',
                              alpha=0.5, label='Wrongly High Perception' if i == 0 else '')
        for i in range(0, len(false_low_perception_times)):
            ax1.fill_betweenx([0, 1], false_low_perception_times[i], false_low_perception_times[i] + timebin, color='yellow',
                              alpha=0.5, label='Wrongly Low Perception' if i == 0 else '')

    ax1.set_ylim([0, 1])
    ax1.set_xlim(plotting_range)
    ax1.set_ylabel('Perception of Stim Presence')
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))

    # Secondary plot: Lick Responses
    lick_presence = np.zeros_like(session_data['timepoints'][time_filt], dtype=bool)
    lick_presence[np.isin(session_data['timepoints'][time_filt], lick_times)] = True  # Mark bins with licks

    ax2.fill_between(
        session_data['timepoints'][time_filt],
        0,
        lick_presence,
        step='pre',
        color='blue',
        alpha=0.5,
        label='Lick Presence'
    )
    ax2.set_xlim(plotting_range)
    ax2.set_ylim([0, 1])
    ax2.set_yticks([])
    ax2.set_xlabel('Time (sec)')
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_time))

    # Merge legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), ncol=2, fontsize='small')

    plt.tight_layout(pad=2.5)

    # Save and close the plot
    plt.close()
    return fig, gr.update(value="Plot", variant="secondary")  # Reset button appearance
