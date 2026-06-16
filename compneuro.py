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
from sklearn.mixture import GaussianMixture
from numpy.lib.stride_tricks import sliding_window_view
from model import RadermeckerTransformer

# ==========================================
# 0. Master Configuration 
# ==========================================
ENABLE_INTEL_NPU = False 
BATCH_SIZE = 64 

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
def extract_frequency_features(raw_window, fs=256):
    freqs, psd = welch(raw_window[9], fs=fs, nperseg=fs) 
    
    delta_mask = (freqs >= 1.0) & (freqs <= 4.0)
    alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
    delta_power = np.trapezoid(psd[delta_mask], freqs[delta_mask])
    alpha_power = np.trapezoid(psd[alpha_mask], freqs[alpha_mask])
    delta_ratio = delta_power / (alpha_power + 1e-6)
    
    psd_norm = psd / np.sum(psd)
    spec_entropy = entropy(psd_norm)
    
    return delta_ratio, spec_entropy

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
        
        global_background_baseline = np.percentile(np.abs(data), 25)
        global_background_baseline = np.maximum(global_background_baseline, 1.0)
        
        candidates = []
        tensor_queue = queue.Queue(maxsize=100) 
        
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
                    
                    # 4 Features fed to FiLM: Voltage, SNR, Delta, Entropy
                    preds_normal, _, latent_normal, _ = model(x_tensor_normal, physics_tensor[:, :4], alpha=0.0)
                    confs_normal = torch.softmax(preds_normal, dim=1)[:, 0].cpu().numpy()
                    
                    x_tensor_inv = torch.tensor(norm_windows * -1.0, dtype=torch.float32).to(device)
                    preds_inv, _, latent_inv, _ = model(x_tensor_inv, physics_tensor[:, :4], alpha=0.0)
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
                                "physics": phys_arr 
                            })
                    
                    tensor_queue.task_done()

        npu_thread = threading.Thread(target=npu_inference_worker)
        npu_thread.daemon = True 
        npu_thread.start()
        
        num_windows = (data.shape[1] - window_samples) // step_samples + 1
        if num_windows > 0:
            strided_data = sliding_window_view(data, window_shape=window_samples, axis=1)
            strided_data = strided_data[:, ::step_samples, :]
            strided_data = np.swapaxes(strided_data, 0, 1) 
            
            print(f"   -> Scanning {strided_data.shape[0]} frames...")
            abs_strided = np.abs(strided_data)
            max_per_channel_all = np.max(abs_strided, axis=2) 
            
            broken_counts = np.sum(max_per_channel_all > 2000.0, axis=1)
            valid_broken_mask = broken_counts < 8
            
            good_channels_mask_all = max_per_channel_all <= 2000.0
            max_good_only = np.where(good_channels_mask_all, max_per_channel_all, 0.0)
            window_peaks_all = np.median(max_good_only, axis=1)
            valid_silence_mask = window_peaks_all >= 5.0
            
            survivor_mask = valid_broken_mask & valid_silence_mask
            survivor_indices = np.where(survivor_mask)[0]
            print(f"   -> C-Level Sieve dropped {strided_data.shape[0] - len(survivor_indices)} empty/broken frames.")
            
            current_batch = []
            
            for i in survivor_indices:
                raw_window = strided_data[i]
                timestamp_start = i * step_samples / fs
                
                max_per_channel = max_per_channel_all[i]
                good_channels_mask = good_channels_mask_all[i]
                
                valid_channels_inf = raw_window[good_channels_mask]
                if len(valid_channels_inf) == 0: continue
                
                window_median_abs = np.median(np.abs(valid_channels_inf), axis=0) 
                q75_val = np.percentile(window_median_abs, 75)
                window_peak = np.max(window_median_abs)
                
                if window_peak < (q75_val * 1.2):
                    continue
                
                voltage_envelope = np.mean(abs_strided[i], axis=0)
                peak_sample_idx = np.argmax(voltage_envelope)
                true_burst_time = timestamp_start + (peak_sample_idx / fs)
                    
                q75, q25 = np.percentile(raw_window, [75, 25], axis=1, keepdims=True)
                iqr = np.maximum(q75 - q25, 10.0)
                median = np.median(raw_window, axis=1, keepdims=True)
                normalized_window = (raw_window - median) / iqr
                normalized_window[~good_channels_mask] = 0.0
                
                burst_peak_to_peak = np.max(np.ptp(valid_channels_inf, axis=1))
                relative_amplitude = burst_peak_to_peak / global_background_baseline
                
                delta_ratio, spec_entropy = extract_frequency_features(raw_window, fs)
                
                peak_val = np.max(voltage_envelope)
                half_max_indices = np.where(voltage_envelope > (peak_val * 0.5))[0]
                
                if len(half_max_indices) > 0:
                    burst_width_sec = (half_max_indices[-1] - half_max_indices[0]) / fs
                else:
                    burst_width_sec = 0.0
                    
                derivatives = np.diff(valid_channels_inf, axis=1)
                sharpness = np.max(np.abs(derivatives)) / (peak_val + 1e-6)

                scaled_absolute_v = burst_peak_to_peak / 1000.0
                
                physics_array = np.array([
                    scaled_absolute_v, 
                    relative_amplitude, 
                    delta_ratio, 
                    spec_entropy, 
                    burst_width_sec, 
                    sharpness
                ], dtype=np.float32)
                
                current_batch.append((raw_window, normalized_window, physics_array, timestamp_start, true_burst_time))
                if len(current_batch) >= BATCH_SIZE:
                    tensor_queue.put(current_batch)
                    current_batch = []
                    
            if current_batch:
                tensor_queue.put(current_batch)
                
        tensor_queue.put(None)
        npu_thread.join()

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
                "Spectral_Entropy": phys_arr[3],
                "Burst_Width_sec": phys_arr[4],
                "Sharpness": phys_arr[5]
            })

    # ==========================================
    # 5. Global Discovery Mapping & HMM Smoothing
    # ==========================================
    if len(extracted_data) < 5:
        print("\nNot enough approved bursts across cohort to perform t-SNE mapping.")
        return

    print("\nMapping AI Hidden Commonalities (Fully Restored Pipeline)...")
    
    latent_matrix = np.array(latent_vectors) 
    df_temp = pd.DataFrame(extracted_data)
    
    physical_features = np.array(df_temp[[
        'Scaled_Absolute_V', 'Relative_Amplitude_SNR', 'Relative_Delta_Ratio', 
        'Spectral_Entropy', 'Burst_Width_sec', 'Sharpness'
    ]], dtype=np.float32)
    
    physical_features[:, 0] = np.log10(physical_features[:, 0] + 1e-6)
    physical_features[:, 1] = np.log10(physical_features[:, 1] + 1e-6)
    physical_features[:, 2] = np.log10(physical_features[:, 2] + 1e-6)
    physical_features[:, 4] = np.log10(physical_features[:, 4] + 1e-6)
    physical_features[:, 5] = np.log10(physical_features[:, 5] + 1e-6)
    
    p1 = np.percentile(physical_features, 1, axis=0) 
    p99_5 = np.percentile(physical_features, 99.5, axis=0)
    physical_clipped = np.clip(physical_features, p1, p99_5)
    
    phys_mean = physical_clipped.mean(axis=0)
    phys_std = physical_clipped.std(axis=0)
    physical_scaled = (physical_clipped - phys_mean) / (phys_std + 1e-6)
    
    from sklearn.decomposition import PCA
    
    n_latent_comps = min(6, len(latent_matrix) - 1)
    pca_latent = PCA(n_components=n_latent_comps, random_state=42)
    latent_compressed = pca_latent.fit_transform(latent_matrix)
    
    # AI Latents successfully fused back with physics
    fused_balanced = np.hstack((latent_compressed, physical_scaled * 1.5))
    
    # Anchor via Relative SNR to ignore skull thickness
    snr_idx = n_latent_comps + 1
    sorted_by_snr = np.argsort(fused_balanced[:, snr_idx])
    
    idx_5th = max(5, int(len(sorted_by_snr) * 0.05))
    idx_50th = int(len(sorted_by_snr) * 0.50)
    idx_95th = min(len(sorted_by_snr) - 5, int(len(sorted_by_snr) * 0.95))
    
    seed_classic = fused_balanced[sorted_by_snr[idx_95th - 5 : idx_95th + 5]].mean(axis=0)
    seed_burnout = fused_balanced[sorted_by_snr[idx_5th - 5 : idx_5th + 5]].mean(axis=0)
    seed_transitional = fused_balanced[sorted_by_snr[idx_50th - 5 : idx_50th + 5]].mean(axis=0)
    
    initial_means = np.array([seed_classic, seed_transitional, seed_burnout])
    
    gmm = GaussianMixture(n_components=3, covariance_type='full', means_init=initial_means, random_state=42, n_init=1)
    raw_labels = gmm.fit_predict(fused_balanced)
    df_temp['Raw_Cluster'] = raw_labels
    
    cluster_severity = df_temp.groupby('Raw_Cluster')['Relative_Amplitude_SNR'].median().sort_values(ascending=False)
    rank_mapping = {cluster_severity.index[0]: 0, cluster_severity.index[1]: 1, cluster_severity.index[2]: 2}
    df_temp['Ordinal_Rank'] = df_temp['Raw_Cluster'].map(rank_mapping)

    # -> HMM RESTORED: To prevent blinks from creating "false stage" events mid-recording
    try:
        from hmmlearn import hmm
        print("Applying Hidden Markov Model (HMM) to enforce clinical continuity...")
        
        hmm_model = hmm.CategoricalHMM(n_components=3, init_params="")
        hmm_model.startprob_ = np.array([0.33, 0.33, 0.34]) 
        hmm_model.transmat_ = np.array([
            [0.95, 0.05, 0.00], 
            [0.02, 0.93, 0.05], 
            [0.00, 0.02, 0.98]  
        ])
        hmm_model.emissionprob_ = np.array([
            [0.85, 0.15, 0.00], 
            [0.10, 0.80, 0.10], 
            [0.00, 0.05, 0.95]  
        ])
        
        for pid in df_temp['Patient_ID'].unique():
            patient_mask = df_temp['Patient_ID'] == pid
            seq = df_temp.loc[patient_mask, 'Ordinal_Rank'].values.reshape(-1, 1)
            if len(seq) > 2: 
                smoothed_seq = hmm_model.predict(seq)
                df_temp.loc[patient_mask, 'Ordinal_Rank'] = smoothed_seq
                
    except ImportError:
        print("\n[WARNING] 'hmmlearn' not installed.")
    except Exception as e:
        print(f"\n[WARNING] HMM Smoothing failed: {e}.")

    final_mapping_dict = {0: "Stage 1 (Classic)", 1: "Stage 2 (Transitional)", 2: "Stage 3 (Burnout)"}
    cluster_labels = df_temp['Ordinal_Rank'].map(final_mapping_dict).tolist()
    
    perplexity_val = min(50, max(5, len(fused_balanced) // 3))
    print(f"Running stabilized t-SNE (Perplexity: {perplexity_val})...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity_val, init='pca', learning_rate='auto', random_state=42)
    latent_2d = tsne.fit_transform(fused_balanced) 

    for i, row in enumerate(extracted_data):
        row["Latent_X"] = latent_2d[i, 0]
        row["Latent_Y"] = latent_2d[i, 1]
        row["Morphology_Cluster"] = cluster_labels[i]

    df = pd.DataFrame(extracted_data)

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