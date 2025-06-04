#%%
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import maximum_filter1d, minimum_filter1d

#%%
filename = 'outputs/logits.npz'
session_data = np.load(filename)
timepoints = session_data['timepoints']
logits = session_data['logits']
stim_presence = session_data['stim_presence']
lick_response = session_data['lick_response']

#%%
# Parameters (adjust based on your experiment)
RESPONSE_WINDOW = 0.5    # seconds after stimulus to detect perception
SAFE_WINDOW = RESPONSE_WINDOW     # seconds before perception to check for stimulus
PERCEPT_DELAY = 1      # seconds before lick to expect perception
LICK_WINDOW = PERCEPT_DELAY        # seconds after perception to expect lick
k = 10.0                 # steepness for sigmoid approximation
lambda_fa = 1.0          # false alarm weight (decoding)
lambda_up = 1.0          # unlicked perception weight (behavior)

# Convert numpy arrays to PyTorch tensors
timepoints = torch.tensor(timepoints, dtype=torch.float32)
logits = torch.tensor(logits, dtype=torch.float32)
stim_presence = torch.tensor(stim_presence, dtype=torch.bool)
lick_response = torch.tensor(lick_response, dtype=torch.bool)

# Precompute time differences between bins
dt = torch.mean(timepoints[1:] - timepoints[:-1]).item()

# Precompute masks using efficient rolling window operations
def create_mask_from_events(event_tensor, window_start, window_end):
    """Create mask where True indicates event in [t+window_start, t+window_end]"""
    mask = torch.zeros_like(event_tensor, dtype=torch.bool)
    event_indices = torch.where(event_tensor)[0]
    
    # Convert time window to sample indices
    start_offset = int(window_start / dt)
    end_offset = int(window_end / dt)
    
    for idx in event_indices:
        start_idx = max(0, idx + start_offset)
        end_idx = min(len(mask), idx + end_offset + 1)
        if start_idx < end_idx:
            mask[start_idx:end_idx] = True
    return mask

# Create masks for efficient loss calculations
safe_stim_mask = create_mask_from_events(stim_presence, -SAFE_WINDOW, 0)
future_lick_mask = create_mask_from_events(lick_response, 0, LICK_WINDOW)

# Get indices of important events
stim_indices = torch.where(stim_presence)[0]
lick_indices = torch.where(lick_response)[0]

# ======================================================================
# 1. DECODING ACCURACY LOSS (Stimulus Alignment) - SOFT VERSION
# ======================================================================
def soft_decoding_loss(T, theta):
    # Apply temperature scaling
    s = F.softmax(logits / T, dim=-1)[...,0]
    percept_indicators = torch.sigmoid(k * (s - theta))
    
    # Miss loss (for stimuli) - soft version
    miss_loss = 0.0
    for idx in stim_indices:
        start_idx = idx
        end_idx = min(len(s), int(idx + RESPONSE_WINDOW/dt) + 1)
        window = percept_indicators[start_idx:end_idx]
        if len(window) > 0:
            # Soft max: differentiable approximation of "any perception in window"
            miss_loss += 1 - torch.max(window)
    
    miss_loss = miss_loss / len(stim_indices) if len(stim_indices) > 0 else 0.0
    
    # FALSE ALARM LOSS - soft version
    # Soft perception events: use percept_indicators directly
    #n_percept = torch.sum(percept_indicators) + 1e-8  # soft count of perception events
    
    # Calculate false alarms: perception without recent stimulus
    false_alarms = percept_indicators * (~safe_stim_mask).float()
    fa_loss = torch.mean(false_alarms)
    
    return miss_loss + lambda_fa * fa_loss

# ======================================================================
# 2. BEHAVIOR MATCHING LOSS (Lick Alignment) - SOFT VERSION
# ======================================================================
def soft_behavior_loss(T, theta):
    # Apply temperature scaling
    s = F.softmax(logits / T, dim=-1)[...,0]
    percept_indicators = torch.sigmoid(k * (s - theta))
    
    # Unexplained lick loss - soft version
    lick_loss = 0.0
    for idx in lick_indices:
        start_idx = max(0, int(idx - PERCEPT_DELAY/dt))
        end_idx = idx + 1
        window = percept_indicators[start_idx:end_idx]
        if len(window) > 0:
            # Soft max: differentiable approximation of "any perception before lick"
            lick_loss += 1 - torch.max(window)
    
    lick_loss = lick_loss / len(lick_indices) if len(lick_indices) > 0 else 0.0
    
    # UNLICKED PERCEPTION LOSS - soft version
    # Soft perception events: use percept_indicators directly
    n_percept = torch.sum(percept_indicators) + 1e-8  # soft count of perception events
    
    # Calculate unlicked perceptions: perception without subsequent lick
    unlicked_percepts = percept_indicators * (~future_lick_mask).float()
    up_loss = torch.sum(unlicked_percepts) / n_percept
    
    return lick_loss + lambda_up * up_loss

# ======================================================================
# OPTIMIZATION FUNCTIONS
# ======================================================================
def optimize_parameters(loss_fn, n_epochs=1000, lr=0.01):
    """Optimize T and theta for given loss function"""
    # Initialize parameters with constraints
    alpha = torch.tensor(0.0, requires_grad=True)  # T = exp(alpha)
    beta = torch.tensor(0.0, requires_grad=True)   # theta = sigmoid(beta)
    
    optimizer = optim.Adam([alpha, beta], lr=lr)
    
    for epoch in range(n_epochs):
        T = torch.exp(alpha)
        theta = torch.sigmoid(beta)
        
        loss = loss_fn(T, theta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}, T={T.item():.4f}, theta={theta.item():.4f}")
    
    return T.item(), theta.item()

# Optimize for decoding accuracy
print("Optimizing for DECODING ACCURACY (soft version)...")
T_dec_soft, theta_dec_soft = optimize_parameters(soft_decoding_loss)
print(f"\nDecoding Optimal (soft): T={T_dec_soft:.4f}, θ={theta_dec_soft:.4f}\n")

# # Optimize for behavior matching
# print("Optimizing for BEHAVIOR MATCHING (soft version)...")
# T_beh_soft, theta_beh_soft = optimize_parameters(soft_behavior_loss)
# print(f"\nBehavior Optimal (soft): T={T_beh_soft:.4f}, θ={theta_beh_soft:.4f}")

# ======================================================================
# POST-OPTIMIZATION ANALYSIS
# ======================================================================
def calculate_perception_events(T, theta):
    """Calculate perception events using optimized parameters"""
    s = F.softmax(logits / T, dim=-1)[...,0]
    return s > theta

# Compare the two optimization results
print("\n" + "="*50)
print("Decoding Optimization Results (soft):")
print(f"Temperature (T): {T_dec_soft:.4f}")
print(f"Threshold (θ): {theta_dec_soft:.4f}")
perception_dec_soft = calculate_perception_events(T_dec_soft, theta_dec_soft)
print(f"Perception events: {perception_dec_soft.sum().item()}/{len(perception_dec_soft)}")

# print("\nBehavior Optimization Results (soft):")
# print(f"Temperature (T): {T_beh_soft:.4f}")
# print(f"Threshold (θ): {theta_beh_soft:.4f}")
# perception_beh_soft = calculate_perception_events(T_beh_soft, theta_beh_soft)
# print(f"Perception events: {perception_beh_soft.sum().item()}/{len(perception_beh_soft)}")

# Additional metrics for evaluation using HARD threshold for final metrics
def calculate_metrics(T, theta, loss_type='decoding'):
    perception_events = calculate_perception_events(T, theta)
    
    if loss_type == 'decoding':
        # Calculate hit rate
        hits = 0
        for idx in stim_indices:
            start_idx = idx
            end_idx = min(len(s), int(idx + RESPONSE_WINDOW/dt) + 1)
            if torch.any(perception_events[start_idx:end_idx]):
                hits += 1
        hit_rate = hits / len(stim_indices) if stim_indices.numel() > 0 else 0
        
        # Calculate false alarm rate
        n_percept = perception_events.sum().item()
        false_alarms = perception_events & (~safe_stim_mask)
        fa_rate = false_alarms.sum().item() / (n_percept + 1e-8) if n_percept > 0 else 0
        
        return {"hit_rate": hit_rate, "fa_rate": fa_rate}
    
    else:  # behavior matching
        # Calculate explained licks
        explained_licks = 0
        for idx in lick_indices:
            start_idx = max(0, int(idx - PERCEPT_DELAY/dt))
            end_idx = idx + 1
            if torch.any(perception_events[start_idx:end_idx]):
                explained_licks += 1
        explain_rate = explained_licks / len(lick_indices) if lick_indices.numel() > 0 else 0
        
        # Calculate unlicked perception rate
        n_percept = perception_events.sum().item()
        unlicked_percepts = perception_events & (~future_lick_mask)
        unlicked_rate = unlicked_percepts.sum().item() / (n_percept + 1e-8) if n_percept > 0 else 0
        
        return {"explain_rate": explain_rate, "unlicked_rate": unlicked_rate}

# Calculate and print metrics for soft versions
dec_metrics_soft = calculate_metrics(T_dec_soft, theta_dec_soft, 'decoding')
# beh_metrics_soft = calculate_metrics(T_beh_soft, theta_beh_soft, 'behavior')

print("\nDecoding Metrics (soft optimized):")
print(f"Hit Rate: {dec_metrics_soft['hit_rate']:.4f}")
print(f"False Alarm Rate: {dec_metrics_soft['fa_rate']:.4f}")

# print("\nBehavior Metrics (soft optimized):")
# print(f"Explained Licks: {beh_metrics_soft['explain_rate']:.4f}")
# print(f"Unlicked Perception Rate: {beh_metrics_soft['unlicked_rate']:.4f}")
# %%