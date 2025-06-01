import numpy as np
import torch
import torch.optim as optim
from scipy.ndimage import maximum_filter1d, minimum_filter1d

filename = ''
session_data = np.load(filename)
timepoints = session_data['timepoints']
logits = session_data['logits']
stim_presence = session_data['stim_presence']
lick_response = session_data['lick_response']


# Parameters (adjust based on your experiment)
RESPONSE_WINDOW = 0.5    # seconds after stimulus to detect perception
SAFE_WINDOW = 1.0        # seconds before perception to check for stimulus
PERCEPT_DELAY = 0.3      # seconds before lick to expect perception
LICK_WINDOW = 1.0        # seconds after perception to expect lick
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
def create_mask_from_events(event_times, window_start, window_end, total_length):
    """Create mask where True indicates event in [t+window_start, t+window_end]"""
    mask = torch.zeros(total_length, dtype=torch.bool)
    event_indices = torch.where(event_times)[0]
    
    for idx in event_indices:
        start_idx = max(0, int(idx + window_start/dt))
        end_idx = min(total_length, int(idx + window_end/dt) + 1)
        if start_idx < end_idx:
            mask[start_idx:end_idx] = True
    return mask

# Create masks for efficient loss calculations
safe_stim_mask = create_mask_from_events(stim_presence, -SAFE_WINDOW, 0, len(timepoints))
future_lick_mask = create_mask_from_events(lick_response, 0, LICK_WINDOW, len(timepoints))

# Get indices of important events
stim_indices = torch.where(stim_presence)[0]
lick_indices = torch.where(lick_response)[0]

# ======================================================================
# 1. DECODING ACCURACY LOSS (Stimulus Alignment)
# ======================================================================
def decoding_loss(T, theta):
    # Apply temperature scaling
    s = torch.sigmoid(logits / T)
    percept_indicators = torch.sigmoid(k * (s - theta))
    
    # Miss loss (for stimuli)
    miss_loss = 0.0
    for idx in stim_indices:
        start_idx = idx
        end_idx = min(len(s), int(idx + RESPONSE_WINDOW/dt) + 1)
        window = percept_indicators[start_idx:end_idx]
        if len(window) > 0:
            miss_loss += 1 - torch.max(window)
    
    miss_loss = miss_loss / len(stim_indices) if len(stim_indices) > 0 else 0.0
    
    # False alarm loss
    fa_loss = torch.mean(percept_indicators * (~safe_stim_mask).float())
    
    return miss_loss + lambda_fa * fa_loss

# ======================================================================
# 2. BEHAVIOR MATCHING LOSS (Lick Alignment)
# ======================================================================
def behavior_loss(T, theta):
    # Apply temperature scaling
    s = torch.sigmoid(logits / T)
    percept_indicators = torch.sigmoid(k * (s - theta))
    
    # Unexplained lick loss
    lick_loss = 0.0
    for idx in lick_indices:
        start_idx = max(0, int(idx - PERCEPT_DELAY/dt))
        end_idx = idx + 1
        window = percept_indicators[start_idx:end_idx]
        if len(window) > 0:
            lick_loss += 1 - torch.max(window)
    
    lick_loss = lick_loss / len(lick_indices) if len(lick_indices) > 0 else 0.0
    
    # Unlicked perception loss
    up_loss = torch.mean(percept_indicators * (~future_lick_mask).float())
    
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
print("Optimizing for DECODING ACCURACY...")
T_dec, theta_dec = optimize_parameters(decoding_loss)
print(f"\nDecoding Optimal: T={T_dec:.4f}, θ={theta_dec:.4f}\n")

# Optimize for behavior matching
print("Optimizing for BEHAVIOR MATCHING...")
T_beh, theta_beh = optimize_parameters(behavior_loss)
print(f"\nBehavior Optimal: T={T_beh:.4f}, θ={theta_beh:.4f}")

# ======================================================================
# POST-OPTIMIZATION ANALYSIS
# ======================================================================
def calculate_perception_events(T, theta):
    """Calculate perception events using optimized parameters"""
    s = torch.sigmoid(logits / T)
    return s > theta

# Compare the two optimization results
print("\n" + "="*50)
print("Decoding Optimization Results:")
print(f"Temperature (T): {T_dec:.4f}")
print(f"Threshold (θ): {theta_dec:.4f}")
perception_dec = calculate_perception_events(T_dec, theta_dec)
print(f"Perception events: {perception_dec.sum().item()}/{len(perception_dec)}")

print("\nBehavior Optimization Results:")
print(f"Temperature (T): {T_beh:.4f}")
print(f"Threshold (θ): {theta_beh:.4f}")
perception_beh = calculate_perception_events(T_beh, theta_beh)
print(f"Perception events: {perception_beh.sum().item()}/{len(perception_beh)}")