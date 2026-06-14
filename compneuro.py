import torch
import mne
import numpy as np
import pandas as pd
import glob
import os
import sys
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from model import RadermeckerTransformer

# ==========================================
# 0. Master Configuration 
# ==========================================
ENABLE_INTEL_NPU = False 
BATCH_SIZE = 64 # Determines how many windows the AI processes simultaneously

# ==========================================
# 1. The Human-In-The-Loop GUI Class
# ==========================================
class HumanInTheLoopValidator:
    def __init__(self, patient_name, channel_names, candidates, fs=256):
        self.patient_name = patient_name
        self.channel_names = channel_names
        self.candidates = candidates
        self.approved_candidates = []
        self.current_idx = 0
        self.fs = fs
        
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title(f"Clinical Validation: {patient_name}")
        plt.subplots_adjust(bottom=0.15, right=0.95, left=0.08)
        
        ax_accept_all = plt.axes([0.45, 0.04, 0.15, 0.06]) 
        ax_approve = plt.axes([0.65, 0.04, 0.12, 0.06])
        ax_reject = plt.axes([0.80, 0.04, 0.12, 0.06])
        
        self.btn_accept_all = Button(ax_accept_all, 'Accept All Remaining', color='#c1d4f0', hovercolor='#8fb9df')
        self.btn_accept_all.on_clicked(self.accept_all)
        
        self.btn_approve = Button(ax_approve, 'Approve (Keep)', color='#c1f0c1', hovercolor='#8fdf8f')
        self.btn_approve.on_clicked(self.approve)
        
        self.btn_reject = Button(ax_reject, 'Reject (Trash)', color='#f0c1c1', hovercolor='#df8f8f')
        self.btn_reject.on_clicked(self.reject)
        
        if self.candidates:
            self.draw()
        else:
            plt.close(self.fig)

    def draw(self):
        self.ax.clear()
        cand = self.candidates[self.current_idx]
        
        vis_window = np.clip(cand['raw_window'], -300, 300)
        start_t = cand['timestamp_start']
        end_t = start_t + 8.0
        time_axis = np.linspace(start_t, end_t, vis_window.shape[1])
        
        spacing = 400 
        for i in range(vis_window.shape[0]):
            self.ax.plot(time_axis, vis_window[i] - i * spacing, color='#1f77b4', linewidth=0.8)
            
        self.ax.set_yticks([-i * spacing for i in range(vis_window.shape[0])])
        self.ax.set_yticklabels(self.channel_names)
        self.ax.set_ylim(-len(self.channel_names)*spacing, spacing)
        
        true_time = cand['true_burst_time']
        self.ax.axvspan(true_time - 0.5, true_time + 2.5, color='red', alpha=0.2, label='AI Detected Region')
        self.ax.legend(loc="upper right")
        
        self.ax.set_title(f"Patient: {self.patient_name} | Candidate {self.current_idx + 1} of {len(self.candidates)}\nAI Confidence: {cand['conf']*100:.1f}%", fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Absolute Time in Recording (Seconds)")
        
        self.fig.canvas.draw()
        
    def accept_all(self, event):
        remaining = self.candidates[self.current_idx:]
        self.approved_candidates.extend(remaining)
        print(f" -> Testing Override: Auto-accepted {len(remaining)} remaining bursts.")
        plt.close(self.fig)

    def approve(self, event):
        self.approved_candidates.append(self.candidates[self.current_idx])
        self.next_burst()

    def reject(self, event):
        self.next_burst()

    def next_burst(self):
        self.current_idx += 1
        if self.current_idx < len(self.candidates):
            self.draw()
        else:
            plt.close(self.fig)

# ==========================================
# 2. AI Boot Sequence & Hardware Routing
# ==========================================
device = torch.device("cpu") 
print(f"Booting Computational Engine on: {device}")

try:
    model = RadermeckerTransformer(num_channels=19, d_model=64, nhead=4, num_layers=3, num_clinical_classes=3, num_patients=7).to(device)
    model.load_state_dict(torch.load("sspe_transformer_weights.pth", map_location=device, weights_only=True))
    model.eval()
    print("[SYSTEM] PyTorch Model weights loaded successfully!")
except Exception as e:
    print(f"\n[FATAL CRASH] Failed to load the AI model. Python Error:\n{e}")
    sys.exit(1)

if ENABLE_INTEL_NPU:
    try:
        import openvino.torch
        print("Compiling Transformer architecture for Intel NPU/AI Boost...")
        model = torch.compile(model, backend="openvino")
        print("Hardware Acceleration unlocked.")
    except Exception as e:
        print(f"Hardware compile failed: {e}")

# ==========================================
# 3. Advanced DSP Feature Extractors
# ==========================================
def calculate_relative_delta(raw_window, fs=256):
    freqs, psd = welch(raw_window[9], fs=fs, nperseg=fs)
    delta_mask = (freqs >= 1.0) & (freqs <= 4.0)
    alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
    
    delta_power = np.trapezoid(psd[delta_mask], freqs[delta_mask])
    alpha_power = np.trapezoid(psd[alpha_mask], freqs[alpha_mask])
    return delta_power / (alpha_power + 1e-6)

def calculate_spectral_entropy(raw_window, fs=256):
    freqs, psd = welch(raw_window[9], fs=fs, nperseg=fs)
    psd_norm = psd / np.sum(psd)
    return entropy(psd_norm)

# ==========================================
# 4. The Auto-Extraction Pipeline
# ==========================================
def analyze_cohort(test_folder_path, confidence_threshold=0.50):
    
    raw_files = glob.glob(os.path.join(test_folder_path, "*.edf")) + glob.glob(os.path.join(test_folder_path, "*.EDF"))
    edf_files = list(set(raw_files))
    edf_files.sort()
    
    extracted_data = []
    latent_vectors = []
    STANDARD_10_20 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

    for file_idx, filepath in enumerate(edf_files):
        patient_name = os.path.basename(filepath).split('.')[0]
        print(f"\n--- Scanning Patient: {patient_name} ---")
        
        mne.set_log_level('WARNING')
        raw = mne.io.read_raw_edf(filepath, preload=True)
        
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
        missing = [ch for ch in STANDARD_10_20 if ch not in raw.ch_names]
        if missing:
            print(f"Skipping {patient_name} due to missing standard channels.")
            continue
            
        raw.pick_channels(STANDARD_10_20)
        raw.reorder_channels(STANDARD_10_20)

        raw.notch_filter(freqs=50.0) 
        raw.filter(l_freq=0.5, h_freq=30.0)
        if raw.info['sfreq'] != 256: raw.resample(sfreq=256)
            
        data = raw.get_data() * 1e6 
        fs = 256
        window_samples = int(8 * fs)    
        step_samples = int(0.25 * fs)   
        
        # --- EARLY FUSION: GLOBAL BACKGROUND BASELINE ---
        global_background_baseline = np.percentile(np.abs(data), 25)
        global_background_baseline = np.maximum(global_background_baseline, 1.0)
        # ------------------------------------------------
        
        candidates = []
        tensor_queue = queue.Queue(maxsize=100) 
        
        # --- SPEED UPGRADE 1: EARLY FUSION WORKER ---
        def npu_inference_worker():
            with torch.inference_mode(): 
                while True:
                    batch = tensor_queue.get()
                    if batch is None: 
                        tensor_queue.task_done()
                        break
                    
                    norm_windows = np.stack([item[1] for item in batch])
                    physics_features = np.stack([item[2] for item in batch]) 
                    
                    x_tensor_normal = torch.tensor(norm_windows, dtype=torch.float32).to(device)
                    physics_tensor = torch.tensor(physics_features, dtype=torch.float32).to(device)
                    
                    # -> EARLY FUSION FORWARD PASS
                    preds_normal, _, latent_normal, _ = model(x_tensor_normal, physics_tensor, alpha=0.0)
                    confs_normal = torch.softmax(preds_normal, dim=1)[:, 0].cpu().numpy()
                    
                    x_tensor_inv = torch.tensor(norm_windows * -1.0, dtype=torch.float32).to(device)
                    preds_inv, _, latent_inv, _ = model(x_tensor_inv, physics_tensor, alpha=0.0)
                    confs_inv = torch.softmax(preds_inv, dim=1)[:, 0].cpu().numpy()
                    
                    latent_normal_np = latent_normal.cpu().numpy()
                    latent_inv_np = latent_inv.cpu().numpy()
                    
                    for i, task in enumerate(batch):
                        raw_window, _, phys_arr, timestamp_start, true_burst_time = task
                        c_norm = confs_normal[i]
                        c_inv = confs_inv[i]
                        
                        if max(c_norm, c_inv) >= confidence_threshold:
                            if c_norm > c_inv:
                                chosen_latent = latent_normal_np[i]
                                final_conf = float(c_norm)
                            else:
                                chosen_latent = latent_inv_np[i]
                                final_conf = float(c_inv)
                                
                            candidates.append({
                                "timestamp_start": timestamp_start,
                                "true_burst_time": true_burst_time,
                                "conf": final_conf,
                                "latent": chosen_latent,
                                "raw_window": raw_window.copy(),
                                "physics": phys_arr # Recovered for CSV
                            })
                    
                    tensor_queue.task_done()

        npu_thread = threading.Thread(target=npu_inference_worker)
        npu_thread.daemon = True 
        npu_thread.start()
        
        current_batch = []
        for start_idx in range(0, data.shape[1] - window_samples, step_samples):
            raw_window = data[:, start_idx : start_idx + window_samples]
            timestamp_start = start_idx / fs
            
            voltage_envelope = np.mean(np.abs(raw_window), axis=0)
            peak_sample_idx = np.argmax(voltage_envelope)
            peak_time_in_window = peak_sample_idx / fs
            true_burst_time = timestamp_start + peak_time_in_window
            
            max_per_channel = np.max(np.abs(raw_window), axis=1)
            broken_electrode_count = np.sum(max_per_channel > 2000.0)
            if broken_electrode_count >= 8: continue 
            
            good_channels_mask = max_per_channel <= 2000.0
            if np.max(np.abs(raw_window[good_channels_mask])) < 15.0: continue
                
            q75, q25 = np.percentile(raw_window, [75, 25], axis=1, keepdims=True)
            iqr = np.maximum(q75 - q25, 10.0)
            median = np.median(raw_window, axis=1, keepdims=True)
            normalized_window = (raw_window - median) / iqr
            normalized_window[~good_channels_mask] = 0.0
            
            # --- EARLY FUSION: UPGRADED 4D DYNAMIC MATH ---
            valid_channels_inf = raw_window[good_channels_mask]
            if len(valid_channels_inf) > 0:
                burst_peak_to_peak = np.max(np.ptp(valid_channels_inf, axis=1))
            else:
                burst_peak_to_peak = 0.0
                
            relative_amplitude = burst_peak_to_peak / global_background_baseline
            delta_ratio = calculate_relative_delta(raw_window, fs)
            spec_entropy = calculate_spectral_entropy(raw_window, fs)
            scaled_absolute_v = burst_peak_to_peak / 1000.0
            
            physics_array = np.array([scaled_absolute_v, relative_amplitude, delta_ratio, spec_entropy], dtype=np.float32)
            # ---------------------------------------------
            
            current_batch.append((raw_window, normalized_window, physics_array, timestamp_start, true_burst_time))
            if len(current_batch) >= BATCH_SIZE:
                tensor_queue.put(current_batch)
                current_batch = []
                
        if current_batch:
            tensor_queue.put(current_batch)
            
        tensor_queue.put(None)
        npu_thread.join()

        # --- FIX 4: BIOLOGICALLY-GATED GREEDY NMS ---
        final_candidates = []
        if len(candidates) > 0:
            sorted_cands = sorted(candidates, key=lambda x: x['conf'], reverse=True)
            while len(sorted_cands) > 0:
                best_cand = sorted_cands.pop(0)
                final_candidates.append(best_cand)
                sorted_cands = [c for c in sorted_cands if abs(c['timestamp_start'] - best_cand['timestamp_start']) > 2.5]
            
            final_candidates = sorted(final_candidates, key=lambda x: x['timestamp_start'])

        if final_candidates:
            print(f"\n[HITL] Launching manual validation GUI for {len(final_candidates)} suspected bursts...")
            validator = HumanInTheLoopValidator(patient_name, STANDARD_10_20, final_candidates)
            plt.show(block=True) 
            approved_bursts = validator.approved_candidates
            print(f" -> Doctor Approved: {len(approved_bursts)} | Rejected: {len(final_candidates) - len(approved_bursts)}")
        else:
            approved_bursts = []
            print(" -> No bursts detected by AI.")

        # --- PHASE D: CLINICAL DSP MATH ---
        if len(approved_bursts) > 1:
            approved_bursts = sorted(approved_bursts, key=lambda x: x['timestamp_start'])
            stitched_bursts = [approved_bursts[0]]
            for i in range(1, len(approved_bursts)):
                current = approved_bursts[i]
                last = stitched_bursts[-1]
                if current['timestamp_start'] - last['timestamp_start'] <= 3.0:
                    print(f" -> Stitching overlapping bursts at {last['timestamp_start']:.1f}s and {current['timestamp_start']:.1f}s")
                    if current['conf'] > last['conf']:
                        stitched_bursts[-1] = current
                else:
                    stitched_bursts.append(current)
            approved_bursts = stitched_bursts

        last_burst_timestamp = None
        for cand in approved_bursts:
            true_time = cand['true_burst_time']
            raw_win = cand['raw_window']
            phys_arr = cand['physics'] 
            
            ibi = None
            if last_burst_timestamp is not None:
                ibi = true_time - last_burst_timestamp
            last_burst_timestamp = true_time
            
            # Use raw unscaled voltage for the CSV just for graphing purposes
            peak_to_peak_uV = phys_arr[0] * 1000.0 
            
            latent_vectors.append(cand['latent'])
            extracted_data.append({
                "Patient_ID": patient_name,
                "Timestamp_s": true_time,
                "AI_Confidence": cand['conf'],
                "Peak_to_Peak_uV": peak_to_peak_uV,
                "Scaled_Absolute_V": phys_arr[0],
                "Relative_Amplitude_SNR": phys_arr[1],
                "Inter_Burst_Interval_s": ibi,
                "Relative_Delta_Ratio": phys_arr[2],
                "Spectral_Entropy": phys_arr[3]
            })

    # ==========================================
    # 5. Global Discovery Mapping & HMM Smoothing
    # ==========================================
    if len(extracted_data) < 5:
        print("\nNot enough approved bursts across cohort to perform t-SNE mapping.")
        return

    print("\nMapping AI Hidden Commonalities (Anchored Early Fusion)...")
    
    latent_matrix = np.array(latent_vectors) 
    df_temp = pd.DataFrame(extracted_data)
    physical_features = df_temp[['Scaled_Absolute_V', 'Relative_Delta_Ratio', 'Spectral_Entropy']].values
    
    # --- FIX A: ROBUST PERCENTILE CLIPPING ---
    # Clips mechanical artifacts so standard deviation isn't blown out
    p5 = np.percentile(physical_features, 5, axis=0)
    p95 = np.percentile(physical_features, 95, axis=0)
    physical_clipped = np.clip(physical_features, p5, p95)
    
    phys_mean = physical_clipped.mean(axis=0)
    phys_std = physical_clipped.std(axis=0)
    physical_scaled = (physical_clipped - phys_mean) / (phys_std + 1e-6)
    
    fused_matrix = np.hstack((latent_matrix, physical_scaled * 5.0))
    # -----------------------------------------
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    raw_labels = kmeans.fit_predict(fused_matrix)
    df_temp['Raw_Cluster'] = raw_labels
    
    # Identify the Ordinal Severity (0 = Highest Volts/Stage 1, 2 = Lowest Volts/Stage 3)
    cluster_severity = df_temp.groupby('Raw_Cluster')['Scaled_Absolute_V'].median().sort_values(ascending=False)
    rank_mapping = {cluster_severity.index[0]: 0, cluster_severity.index[1]: 1, cluster_severity.index[2]: 2}
    df_temp['Ordinal_Rank'] = df_temp['Raw_Cluster'].map(rank_mapping)
    
    # --- FIX B: HIDDEN MARKOV MODEL (HMM) TEMPORAL SMOOTHING ---
    try:
        from hmmlearn import hmm
        print("Applying Hidden Markov Model (HMM) to enforce clinical continuity...")
        
        # Define strict biological rules: Mostly stays the same, slight chance to progress, rarely reverses.
        hmm_model = hmm.CategoricalHMM(n_components=3, init_params="")
        hmm_model.startprob_ = np.array([0.6, 0.3, 0.1]) 
        hmm_model.transmat_ = np.array([
            [0.95, 0.05, 0.00], 
            [0.02, 0.93, 0.05], 
            [0.00, 0.02, 0.98]  
        ])
        hmm_model.emissionprob_ = np.array([
            [0.85, 0.15, 0.00], 
            [0.10, 0.80, 0.10], 
            [0.00, 0.15, 0.85]  
        ])
        
        # Apply smoothing chronologically for each patient's individual timeline
        for pid in df_temp['Patient_ID'].unique():
            patient_mask = df_temp['Patient_ID'] == pid
            seq = df_temp.loc[patient_mask, 'Ordinal_Rank'].values.reshape(-1, 1)
            if len(seq) > 2: 
                smoothed_seq = hmm_model.predict(seq)
                df_temp.loc[patient_mask, 'Ordinal_Rank'] = smoothed_seq
                
    except ImportError:
        print("\n[WARNING] 'hmmlearn' not installed. Skipping HMM smoothing. (Run: pip install hmmlearn)")
    except Exception as e:
        print(f"\n[WARNING] HMM Smoothing failed: {e}. Falling back to AI raw clusters.")
    # ----------------------------------------------------------

    # Map the smoothed ranks to strings
    final_mapping_dict = {0: "Stage 1 (Classic)", 1: "Stage 2 (Transitional)", 2: "Stage 3 (Burnout)"}
    cluster_labels = df_temp['Ordinal_Rank'].map(final_mapping_dict).tolist()
    
    perplexity_val = min(30, len(fused_matrix) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42)
    latent_2d = tsne.fit_transform(fused_matrix) 

    # Re-pack the smoothed data into the list to ensure the CSV and plotting are aligned perfectly
    for i, row in enumerate(extracted_data):
        row["Latent_X"] = latent_2d[i, 0]
        row["Latent_Y"] = latent_2d[i, 1]
        row["Morphology_Cluster"] = cluster_labels[i]

    df = pd.DataFrame(extracted_data)

    # --- FIX 6: CLINICAL IBI UPGRADES (SMOOTHING & PERIODICITY) ---
    if len(df) > 5:
        df = df.sort_values(by=['Patient_ID', 'Timestamp_s']).reset_index(drop=True)
        df['Rolling_Median_IBI_s'] = df.groupby('Patient_ID')['Inter_Burst_Interval_s'].transform(
            lambda x: x.rolling(window=5, min_periods=1, center=True).median()
        )
        df['IBI_Periodicity_CoV'] = df.groupby('Patient_ID')['Inter_Burst_Interval_s'].transform(
            lambda x: x.rolling(window=10, min_periods=3).std() / x.rolling(window=10, min_periods=3).mean()
        )
        df['Bursts_Per_Minute'] = 60.0 / (df['Rolling_Median_IBI_s'] + 1e-6)

    df.to_csv("sspe_cohort_analysis.csv", index=False)
    print("\nExtraction Complete! Deep cohort data saved to 'sspe_cohort_analysis.csv'")

# ==========================================
# FINAL EXECUTION BLOCK (DIAGNOSTIC MODE)
# ==========================================
if __name__ == "__main__":
    print("\n[SYSTEM] Main execution block triggered successfully!")
    
    TEST_DIRECTORY = os.path.abspath("data/test/")
    print(f"[SYSTEM] Searching for patients in: {TEST_DIRECTORY}")
    
    if not os.path.exists(TEST_DIRECTORY):
        print(f"\n[FATAL ERROR] The folder does not exist: {TEST_DIRECTORY}")
        sys.exit(1)
        
    valid_files = [f for f in os.listdir(TEST_DIRECTORY) if f.lower().endswith('.edf')]
    
    if len(valid_files) == 0:
        print(f"\n[FATAL ERROR] Found 0 EDF files in the test directory.")
        sys.exit(1)
        
    print(f"[SYSTEM] Found {len(valid_files)} patient files. Launching AI Pipeline...\n")
    analyze_cohort(TEST_DIRECTORY)
            
    print("\n[SYSTEM] Pipeline shut down gracefully.")