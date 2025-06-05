from ..temp_matching.utils import normalize_mat_by_row,get_psth_sum
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import umap
import warnings

# Suppress UMAP warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="'force_all_finite' was renamed to 'ensure_all_finite'")
warnings.filterwarnings("ignore", category=UserWarning, message="using precomputed metric; inverse_transform will be unavailable")
warnings.filterwarnings("ignore", category=UserWarning, message="Transforming new data with precomputed metric. We are assuming the input data is a matrix of distances from the new points to the points in the training set.")

def generate_behavior_corr_plot(trial_results, binary_pred):
    trial_results = pd.Series(trial_results)

    hit_stim = np.sum((trial_results == 'hit') & binary_pred)
    hit_nostim = np.sum((trial_results == 'hit') & (~binary_pred))

    miss_stim = np.sum((trial_results == 'miss') & binary_pred)
    miss_nostim = np.sum((trial_results == 'miss') & (~binary_pred))

    fa_stim = np.sum((trial_results == 'FA') & binary_pred)
    fa_nostim = np.sum((trial_results == 'FA') & (~binary_pred))

    behavior_decoding_corr = np.array([[hit_stim,hit_nostim],
                                    [miss_stim,miss_nostim],
                                    [fa_stim,fa_nostim]])
    fig, ax = plt.subplots()
    sns.heatmap(normalize_mat_by_row(behavior_decoding_corr),square=True,annot=True,cmap='Blues',xticklabels=['Stim','No Stim'],yticklabels=['Hit','Miss','False Alarm'],ax=ax)
    plt.ylabel('Trial Type')
    plt.xlabel('Prediction')
    plt.close()
    return fig

def generate_clustering_plot(trial_templates, trial_stim_amps, distance_metric, centroid_templates, template_stim_amps, incorrect_indices=None):
    """
    Projects trial_templates into 2D using UMAP and plots them, coloring each point based on trial_stim_amp.
    Optionally, centroid_templates can also be plotted with distinct colors and legends.
    Incorrectly classified points can be highlighted if their indices are provided.
    """
    # Perform UMAP dimensionality reduction
    temp_distances = pairwise_distances(X=trial_templates, metric=distance_metric)
    reducer = umap.UMAP(metric='precomputed', n_neighbors=5, min_dist=0, n_jobs=-1)
    trial_templates_2d = reducer.fit_transform(temp_distances)

    # Transform centroid templates into the same 2D space
    if centroid_templates is not None:
        centroid_distances = pairwise_distances(X=centroid_templates, Y=trial_templates, metric=distance_metric)
        centroid_2d = reducer.transform(centroid_distances)

    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        trial_templates_2d[:, 0], trial_templates_2d[:, 1],
        c=trial_stim_amps, cmap="Reds", s=50, edgecolor="black", alpha=0.8
    )

    BORDER_WIDTH = 1
    # Highlight incorrectly classified points
    if incorrect_indices is not None:
        plt.scatter(
            trial_templates_2d[incorrect_indices, 0], trial_templates_2d[incorrect_indices, 1],
            facecolors='none', edgecolors='blue', s=100, label="Incorrectly Classified", linewidth=BORDER_WIDTH
        )

    # Plot centroids with distinct colors and legends
    if centroid_templates is not None:
        centroid_colors = sns.color_palette("Blues", 5)  # Generate distinct colors
        for i, stim_amp in enumerate(template_stim_amps):
            plt.scatter(
                centroid_2d[i, 0], centroid_2d[i, 1],
                c=[centroid_colors[stim_amp]], marker="X", s=200, label=f"Centroid (Stim Amp {stim_amp})", edgecolor="black",linewidths=BORDER_WIDTH
            )

    # Add colorbar for the graded coloring
    cbar = plt.colorbar(scatter)
    cbar.set_label("Stim Amp", fontsize=12)
    cbar.set_ticks(np.unique(trial_stim_amps))  # Set color bar ticks to unique integer values of stim_amps

    # Add plot aesthetics
    plt.title("UMAP Projection of Trial Templates with Centroids", fontsize=16)
    plt.xlabel("UMAP-1", fontsize=12)
    plt.ylabel("UMAP-2", fontsize=12)
    plt.legend(title="Legend", fontsize=10, title_fontsize=12, loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.close()
    return fig  # Return the figure

def plot_confusion_matrix(true_stim_amps, binary_pred):
    cm = np.zeros(shape=(5,2))
    for i in range(cm.shape[0]):
        for j in [0,1]:
            cm[i,j] = ((true_stim_amps == i) & (binary_pred == bool(j))).sum()

    fig, ax = plt.subplots()
    sns.heatmap(normalize_mat_by_row(cm),square=True,cmap='Blues',annot=True,ax=ax)
    plt.xlabel('Stim Prediction')
    plt.ylabel('Stim Amp')
    plt.title('Stimulus Classification Result\nNormalized Per Row')
    ax.set_xticklabels(['No Stim','Stim'])
    ax.set_yticklabels(range(5))
    plt.close()
    return fig

def generate_psth_plot(stim_amps,template_stim_amps,neural_activity,trial_onsets,time_bound,bin_size):
    bin_edges = np.arange(start=time_bound[0],stop=time_bound[1],step=bin_size) 
    fig, axs = plt.subplots(nrows=1, ncols=len(template_stim_amps), figsize=(30, 4),dpi=300,sharey=True)
    for stim_amp_idx,stim_amp in enumerate(template_stim_amps):
        stim_onsets = trial_onsets[stim_amps == stim_amp]
        neuron_psth = get_psth_sum(neural_activity,stim_onsets,bin_size,time_bound,smooth=False)
        peak_index = np.argmax(neuron_psth)
        peak_value = neuron_psth[peak_index]
        peak_time = bin_edges[peak_index]
        ax = axs.flat[stim_amp_idx]
        ax.plot(bin_edges[:-1],neuron_psth,label='PSTH')
        ax.axvline(0,c='r',label='Onset')
        ax.set_xlim(time_bound)
        # mark peak!
        ax.scatter(peak_time,peak_value,marker='^',c='orange',zorder=3,label='Peak')
        ax.set_title(f'stim level: {stim_amp}')
        ax.set_xlabel('time (sec)')
        ax.legend()
    plt.close()
    return fig

def generate_roc_plot(y_true, y_scores,chosen_threshold):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Find the index of the chosen threshold
    threshold_idx = (abs(thresholds - chosen_threshold)).argmin()
    chosen_fpr = fpr[threshold_idx]
    chosen_tpr = tpr[threshold_idx]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random Guess')

    # Mark the chosen threshold point
    plt.scatter([chosen_fpr], [chosen_tpr], color='red', label=f'Threshold = {chosen_threshold:.2f}')
    plt.axvline(x=chosen_fpr, color='red', linestyle='--', lw=1, alpha=0.7)
    plt.axhline(y=chosen_tpr, color='red', linestyle='--', lw=1, alpha=0.7)

    # Annotate the chosen threshold point
    plt.text(chosen_fpr, chosen_tpr, f'({chosen_fpr:.2f}, {chosen_tpr:.2f})', 
             fontsize=10, color='red', ha='right', va='bottom')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.close()
    return fig

def generate_softscore_hist_plot(trial_scores, trial_stim_amps):
    # softscore for each stim amp in time seq
    # hist of softscore per each stim amp
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True, sharey=True)  # Adjust figure size and share axes

    # Use a pleasing color palette
    palette = sns.color_palette("Set2", 5)

    for stim_amp in range(5):
        trial_filt = trial_stim_amps == stim_amp
        y = trial_scores[trial_filt]  # Filtered trial scores

        ax = axes[stim_amp]
        sns.histplot(y, bins=20, kde=True, stat='probability', ax=ax, color=palette[stim_amp], edgecolor="black")  # Plot density

        # Calculate and plot the mean
        mean_value = np.mean(y)
        std_dev = np.std(y)  # Calculate standard deviation (wideness metric)
        ax.axvline(mean_value, color='red', linestyle='--', linewidth=1.5, label=f"Mean: {mean_value:.2f}")
        ax.axvline(mean_value + std_dev, color='blue', linestyle='--', linewidth=1.5, label=f"Std Dev: {std_dev:.2f}")
        ax.axvline(mean_value - std_dev, color='blue', linestyle='--', linewidth=1.5)  # Plot negative std deviation

        # Force the y-axis limits to update after plotting
        ax.set_ylim(0, ax.get_ylim()[1])  # Explicitly set the lower limit to 0

        # Get updated y-axis limits for the current subplot
        y_min, y_max = ax.get_ylim()

        # Add yellow tint between the two STD lines
        ax.fill_betweenx(
            y=[y_min, y_max],  # Fill from the bottom to the top of the y-axis
            x1=mean_value - std_dev,
            x2=mean_value + std_dev,
            color='yellow',
            alpha=0.2  # Adjust transparency
        )

        # Add legends for mean and standard deviation
        ax.legend(loc="upper right")  # Add the mean and std dev as separate legends

        ax.set_title(f'Stim Amp: {stim_amp}')
        ax.set_xlim([0, 1])  # Assuming trial scores are normalized between 0 and 1

    # Add shared x-axis and y-axis labels
    fig.text(0.55, 0.04, 'Trial Scores', ha='center', fontsize=12)  # Adjusted y position for better centering
    fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=12)  # Updated label to 'Density'

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.close()
    return fig  # Return the figure

def generate_softscore_overtime_plot(trial_scores, trial_stim_amps):
    # softscore for each stim amp in time seq
    # hist of softscore per each stim amp
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True, sharey=True)  # Adjust figure size and share axes
    slopes = []

    palette = sns.color_palette("Set2", 5)

    for stim_amp in range(5):
        trial_filt = trial_stim_amps == stim_amp
        x = np.arange(len(trial_scores[trial_filt]))
        y = trial_scores[trial_filt]

        ax = axes[stim_amp]
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={'s': 10, 'color': palette[0]}, 
                    line_kws={'color': 'red', 'linestyle': '--'})
        slope = np.polyfit(x, y, 1)[0]  # Extract slope from regression
        slopes.append(slope)
        ax.legend([f"Slope: {slope:.2f}"], loc="upper right")
        ax.set_title(f'Stim Amp: {stim_amp}')

        ax.set_ylim([0, 1])

    fig.text(0.55, 0.04, 'Trial Occurrence', ha='center', fontsize=12)  # Adjusted y position for better centering
    fig.text(0.04, 0.5, 'Perception', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.close()
    return fig

def generate_temp_distance_plot(stim_templates, sim_metric):
    fig, ax = plt.subplots()
    temp_distances = pairwise_distances(X=stim_templates,metric=sim_metric)
    sns.heatmap(temp_distances/temp_distances.max(),cmap='Blues',square=True,ax=ax)
    plt.title('Normalized Pairwise Distance of Templates')
    plt.xlabel('Stim Level')
    plt.ylabel('Stim Level')
    plt.close()
    return fig