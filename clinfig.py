import torch
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from model import RadermeckerTransformer

# ==========================================
# 1. Configuration
# ==========================================
device = torch.device("cpu")
PATIENT_FILE = "data/test/SUB_3.edf" # Point this to a patient with known bursts
STANDARD_10_20 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

# Load Model
model = RadermeckerTransformer(num_channels=19, d_model=64, nhead=4, num_layers=3, num_clinical_classes=3, num_patients=4).to(device)
model.load_state_dict(torch.load("sspe_transformer_weights.pth", map_location=device, weights_only=True))
model.eval()

# ==========================================
# 2. Extract a Single High-Confidence Burst
# ==========================================
print(f"Scanning {PATIENT_FILE} for a representative burst...")
mne.set_log_level('WARNING')
raw = mne.io.read_raw_edf(PATIENT_FILE, preload=True)

# Standardize channels (Same logic as compneuro.py)
rename_mapping = {}
for ch in raw.ch_names:
    clean = ch.upper().replace('EEG', '').replace('-REF', '').replace('-LE', '').strip()
    if clean == 'FP1': clean = 'Fp1'
    elif clean == 'FP2': clean = 'Fp2'
    elif clean == 'T7': clean = 'T3'
    elif clean == 'T8': clean = 'T4'
    elif clean == 'P7': clean = 'T5'
    elif clean == 'P8': clean = 'T6'
    elif len(clean) >= 2 and clean[1].isalpha(): clean = clean[0] + clean[1:].lower()
    rename_mapping[ch] = clean

raw.rename_channels(rename_mapping)
raw.pick_channels(STANDARD_10_20)
raw.reorder_channels(STANDARD_10_20)
raw.notch_filter(freqs=50.0)
raw.filter(l_freq=0.5, h_freq=30.0)
if raw.info['sfreq'] != 256: raw.resample(sfreq=256)

data = raw.get_data() * 1e6
fs = 256
window_samples = int(8 * fs)

target_raw_window = None
target_norm_window = None

# Find the first 95%+ confidence burst to visualize
for start_idx in range(0, data.shape[1] - window_samples, int(0.5 * fs)):
    raw_window = data[:, start_idx : start_idx + window_samples]
    
    # Fast filtering
    if np.max(np.abs(raw_window)) > 2000.0 or np.max(np.abs(raw_window)) < 15.0:
        continue
        
    q75, q25 = np.percentile(raw_window, [75, 25], axis=1, keepdims=True)
    iqr = np.maximum(q75 - q25, 10.0)
    median = np.median(raw_window, axis=1, keepdims=True)
    norm_window = (raw_window - median) / iqr
    
    x_tensor = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0).to(device)
    preds, _, _, _ = model(x_tensor, alpha=0.0)
    conf = torch.softmax(preds, dim=1)[0][0].item() # Assuming Index 0 is SSPE
    
    if conf > 0.95:
        target_raw_window = raw_window
        target_norm_window = norm_window
        print(f"-> Found perfect burst! AI Confidence: {conf*100:.2f}%")
        break

if target_raw_window is None:
    print("Could not find a high-confidence burst in this file. Try a different EDF.")
    exit()

# ==========================================
# 3. Explainable AI: Calculate Gradient Saliency
# ==========================================
print("Calculating Saliency Map...")
x_tensor = torch.tensor(target_norm_window, dtype=torch.float32).unsqueeze(0).to(device)

# -> THE XAI MAGIC: We tell PyTorch to track the gradients of the INPUT image
x_tensor.requires_grad = True 

# Forward Pass
preds, _, _, _ = model(x_tensor, alpha=0.0)

# Backward Pass (Force the gradient specifically for the SSPE class)
preds[0, 0].backward() 

# Calculate the importance of each millisecond by taking the absolute gradient
# We average across all 19 channels to get a 1D time-series heatmap
saliency_map = x_tensor.grad.abs().squeeze().mean(dim=0).cpu().numpy()

# Smooth the heatmap slightly for a better visual glow
window_len = 10
saliency_map = np.convolve(saliency_map, np.ones(window_len)/window_len, mode='same')
saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

# ==========================================
# 4. Generate the Publication Figure
# ==========================================
print("Rendering Figure...")
fig = plt.figure(figsize=(16, 10))
fig.canvas.manager.set_window_title("Explainable AI & Spatial Localization")

# --- SUBPLOT 1: The XAI Saliency Map ---
ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
vis_window = np.clip(target_raw_window, -300, 300)
time_axis = np.linspace(0, 8, vis_window.shape[1])
spacing = 400

# Plot the 19 EEG Channels
for i in range(vis_window.shape[0]):
    ax1.plot(time_axis, vis_window[i] - i * spacing, color='#1f77b4', linewidth=1.0, zorder=2)

ax1.set_yticks([-i * spacing for i in range(vis_window.shape[0])])
ax1.set_yticklabels(STANDARD_10_20)
ax1.set_ylim(-len(STANDARD_10_20)*spacing, spacing)
ax1.set_xlim(0, 8)
ax1.set_title("Transformer Attention Map (Saliency)", fontsize=16, fontweight='bold')
ax1.set_xlabel("Time in Window (Seconds)", fontsize=12)

# Overlay the glowing Heatmap behind the brainwaves
# We create a custom transparent-to-red colormap
colors = [(1, 1, 1, 0), (1, 1, 0, 0.4), (1, 0, 0, 0.8)] # Clear -> Yellow -> Red
custom_cmap = LinearSegmentedColormap.from_list("xai_map", colors)

# Stretch the 1D saliency array to cover the Y-axis height
extent = [0, 8, ax1.get_ylim()[0], ax1.get_ylim()[1]]
ax1.imshow(saliency_map[np.newaxis, :], aspect='auto', cmap=custom_cmap, extent=extent, zorder=1)

# --- SUBPLOT 2: The Topographical Head Map ---
ax2 = plt.subplot2grid((1, 3), (0, 2))

# Extract the maximum voltage spike for each channel to plot on the scalp
peak_voltages = np.max(np.abs(target_raw_window), axis=1)

# Create an MNE Info object to map the standard 10-20 electrode positions
info = mne.create_info(ch_names=STANDARD_10_20, sfreq=256, ch_types='eeg')
info.set_montage('standard_1020')

# Plot the Topomap
im, _ = mne.viz.plot_topomap(
    peak_voltages, 
    info, 
    axes=ax2, 
    show=False, 
    cmap='magma', 
    vlim=(0, np.max(peak_voltages))
)
ax2.set_title("Anatomical Localization\n(Peak Amplitude Distribution)", fontsize=16, fontweight='bold')

# Add a colorbar for the Topomap
cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', fraction=0.05, pad=0.05)
cbar.set_label('Voltage Intensity (µV)')

plt.tight_layout()
plt.savefig("figure_4_xai_topomap.png", dpi=300)
print("Saved publication figure to 'figure_4_xai_topomap.png'!")
plt.show()