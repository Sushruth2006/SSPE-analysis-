import torch
import mne
import numpy as np
from model import RadermeckerTransformer

# ==========================================
# 1. Setup & Load the Trained Brain
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Booting AI on: {device}")

model = RadermeckerTransformer(num_channels=19, d_model=64, nhead=4, num_layers=3, num_clinical_classes=3, num_patients=4).to(device)

try:
    model.load_state_dict(torch.load("sspe_transformer_weights.pth", map_location=device, weights_only=True))
    print("Memory weights successfully loaded.")
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'sspe_transformer_weights.pth'. Did you run train.py first?")

model.eval() 

# ==========================================
# 2. Autonomous Diagnosis Function
# ==========================================
def scan_new_patient(edf_filepath, confidence_threshold=0.85):
    print(f"\nLoading Unseen Patient Data: {edf_filepath}")
    mne.set_log_level('WARNING')
    raw = mne.io.read_raw_edf(edf_filepath, preload=True)
    
    STANDARD_10_20 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
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
    
    # -> FIX 1: Dropped to 30Hz to perfectly match training and kill muscle artifact
    raw.filter(l_freq=0.5, h_freq=30.0)
    
    if raw.info['sfreq'] != 256: raw.resample(sfreq=256)
        
    data = raw.get_data() * 1e6 
    fs = 256
    window_samples = int(8 * fs)    
    step_samples = int(0.25 * fs)   
    
    total_seconds = data.shape[1] / fs
    print(f"Brainwave Preprocessing Complete. Scanning {total_seconds:.1f} seconds of EEG data...\n")
    
    raw_detections = []
    highest_confidence_seen = 0.0
    
    with torch.no_grad(): 
        for start_idx in range(0, data.shape[1] - window_samples, step_samples):
            window = data[:, start_idx : start_idx + window_samples]
            timestamp_start = start_idx / fs
            
            # --- 1. SMART ARTIFACT CATCHER ---
            max_per_channel = np.max(np.abs(window), axis=1)
            broken_electrode_count = np.sum(max_per_channel > 2000.0)
            if broken_electrode_count >= 8: continue 
                
            # --- 1.5 ISOLATE AND MUTE BROKEN WIRES ---
            good_channels_mask = max_per_channel <= 2000.0
            
            # --- 1.6 THE "QUIET BACKGROUND" CATCHER ---
            if np.max(np.abs(window[good_channels_mask])) < 50.0: continue
                
            # --- 2. ROBUST SCALER (IQR Normalization) ---
            # -> FIX 2: Replaced brittle Z-score with Median/IQR robust scaling
            q75, q25 = np.percentile(window, [75, 25], axis=1, keepdims=True)
            iqr = q75 - q25
            median = np.median(window, axis=1, keepdims=True)
            
            normalized_window = (window - median) / (iqr + 1e-6)
            
            # Explicitly mute the bad channels so the Transformer ignores them!
            normalized_window[~good_channels_mask] = 0.0
            
            # --- 3. POLARITY AGNOSTIC SCANNING ---
            x_tensor_normal = torch.tensor(normalized_window, dtype=torch.float32).unsqueeze(0).to(device)
            preds_normal, _, _, _ = model(x_tensor_normal, alpha=0.0)
            conf_normal = torch.softmax(preds_normal, dim=1)[0][0].item()
            
            x_tensor_inv = torch.tensor(normalized_window * -1.0, dtype=torch.float32).unsqueeze(0).to(device)
            preds_inv, _, _, _ = model(x_tensor_inv, alpha=0.0)
            conf_inv = torch.softmax(preds_inv, dim=1)[0][0].item()
            
            sspe_confidence = max(conf_normal, conf_inv)
            
            if 66.0 <= timestamp_start <= 69.0:
                print(f" [Microscope] Window starting at {timestamp_start:.2f}s | Normal Conf: {conf_normal*100:.1f}% | Inverted Conf: {conf_inv*100:.1f}% | Broken Wires: {broken_electrode_count}")
            
            if sspe_confidence > highest_confidence_seen:
                highest_confidence_seen = sspe_confidence
            
            if sspe_confidence >= confidence_threshold:
                raw_detections.append((timestamp_start, sspe_confidence))

    # --- 4. NON-MAXIMUM SUPPRESSION ---
    final_bursts = []
    if len(raw_detections) > 0:
        current_cluster = [raw_detections[0]]
        for i in range(1, len(raw_detections)):
            t, conf = raw_detections[i]
            prev_t, _ = current_cluster[-1]
            if t - prev_t <= 1.5: current_cluster.append((t, conf))
            else:
                best_t, best_conf = max(current_cluster, key=lambda x: x[1])
                final_bursts.append((best_t, best_conf))
                current_cluster = [(t, conf)]
        if current_cluster:
            best_t, best_conf = max(current_cluster, key=lambda x: x[1])
            final_bursts.append((best_t, best_conf))

    print("\n--- CLEAN DETECTIONS ---")
    for window_start, conf in final_bursts:
        true_burst_time = window_start + 2.0 
        print(f"⚠️ PATHOLOGY DETECTED: Radermecker burst at {true_burst_time:.1f}s (Confidence: {conf*100:.1f}%)")
                
    print(f"\nScan Complete. Total unique SSPE epochs detected: {len(final_bursts)}")
    print(f"-> DEBUG: The highest SSPE confidence seen in this file was {highest_confidence_seen*100:.1f}%")

if __name__ == "__main__":
    NEW_FILE = "data/test/SUB_4.EDF" 
    import os
    if os.path.exists(NEW_FILE):
        scan_new_patient(NEW_FILE, confidence_threshold=0.92)
    else:
        print(f"Waiting for test data: Please place your EDF file at '{NEW_FILE}'.")