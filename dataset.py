import mne
import torch
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from scipy.signal import welch
from scipy.stats import entropy

class RadermeckerDataset(Dataset):
    def __init__(self, edf_files, event_dict, patient_ids, tmin=-2.0, tmax=6.0, target_fs=256, notch_freq=50.0):
        self.data = []
        self.labels = []
        self.patient_labels = []
        
        # --- NEW: Store the global baseline for each window ---
        self.baselines = []
        
        mne.set_log_level('WARNING')
        
        self.STANDARD_10_20 = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2'
        ]
        
        for filepath, p_id in zip(edf_files, patient_ids):
            print(f"\nProcessing {filepath}...")
            
            raw = mne.io.read_raw_edf(filepath, preload=True)
            
            # --- 1. Fix TSV Reading ---
            base_name = os.path.splitext(filepath)[0] 
            tsv_path = base_name + '_events.tsv'
            
            if os.path.exists(tsv_path):
                df = pd.read_csv(tsv_path, sep='\t')
                
                onset_col = [c for c in df.columns if 'onset' in c.lower()][0]
                dur_col = [c for c in df.columns if 'duration' in c.lower()][0]
                label_col = [c for c in df.columns if 'desc' in c.lower() or 'label' in c.lower() or 'trial_type' in c.lower()][0]
                
                annotations = mne.Annotations(onset=df[onset_col], duration=df[dur_col], description=df[label_col])
                raw.set_annotations(annotations)
                print(f" -> Successfully loaded labels using columns: {onset_col}, {dur_col}, {label_col}")
            
            # --- 2. Fix Channel Names ---
            rename_mapping = {}
            for ch in raw.ch_names:
                clean = ch.replace('EEG', '').strip()
                if clean.upper() == 'FP1': clean = 'Fp1'
                elif clean.upper() == 'FP2': clean = 'Fp2'
                elif clean.upper() == 'T7': clean = 'T3'
                elif clean.upper() == 'T8': clean = 'T4'
                elif clean.upper() == 'P7': clean = 'T5'
                elif clean.upper() == 'P8': clean = 'T6'
                elif len(clean) >= 2 and clean[1].isalpha():
                    clean = clean[0] + clean[1:].lower()
                rename_mapping[ch] = clean

            raw.rename_channels(rename_mapping)
            
            missing = [ch for ch in self.STANDARD_10_20 if ch not in raw.ch_names]
            if missing:
                print(f" -> ERROR: Still missing channels: {missing}")
                continue
            
            raw.pick_channels(self.STANDARD_10_20)
            raw.reorder_channels(self.STANDARD_10_20)

            # --- 3. Filter & Epoch ---
            raw.notch_filter(freqs=notch_freq) 
            raw.filter(l_freq=0.5, h_freq=30.0) 
            if raw.info['sfreq'] != target_fs: raw.resample(sfreq=target_fs)
            
            # --- NEW: PRE-CALCULATE HARDWARE BASELINE ---
            # Calculate the 25th percentile of the full recording to get the absolute resting voltage
            raw_data_uV = raw.get_data() * 1e6
            patient_baseline = np.maximum(np.percentile(np.abs(raw_data_uV), 25), 1.0)
            # --------------------------------------------
                
            try:
                events, actually_found_events = mne.events_from_annotations(raw, event_id=event_dict)
                epochs = mne.Epochs(raw, events, event_id=actually_found_events, 
                                    tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True)
                
                extracted_epochs = epochs.get_data(copy=False)
                self.data.append(extracted_epochs)
                self.labels.append(epochs.events[:, -1]) 
                self.patient_labels.append(np.full(len(epochs.events[:, -1]), p_id))
                
                # Attach the patient's global baseline to every single window we just extracted
                self.baselines.extend([patient_baseline] * len(extracted_epochs))
                
            except ValueError as e:
                print(f" -> ERROR: Event extraction failed: {e}")
                continue
                
        # --- 4. Finalize Dataset ---
        if len(self.data) > 0:
            self.data = np.concatenate(self.data, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
            self.patient_labels = np.concatenate(self.patient_labels, axis=0)
            self.baselines = np.array(self.baselines)
            print(f"\n[Dataset Ready] {len(self.labels)} windows extracted.")
        else:
            print("\n[Dataset Empty] No valid files or labels found.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw_x = self.data[idx] * 1e6 
        fs = 256
        
        # Move channel masking UP so we can use it for physics math
        max_per_channel = np.max(np.abs(raw_x), axis=1)
        good_channels_mask = max_per_channel <= 2000.0

        # ==========================================
        # EARLY FUSION: ON-THE-FLY PHYSICS MATH
        # ==========================================
        # 1. Global Peak-to-Peak (Check the whole skull, not just Cz)
        valid_channels = raw_x[good_channels_mask]
        if len(valid_channels) > 0:
            burst_peak_to_peak = np.max(np.ptp(valid_channels, axis=1))
        else:
            burst_peak_to_peak = 0.0
            
        # 2. Hardware-Agnostic Severity (SNR)
        relative_amplitude = burst_peak_to_peak / self.baselines[idx]
        
        # 3. Slow-Wave Ratio (Delta/Alpha) - Still using Cz (idx 9) for frequency baseline
        freqs, psd = welch(raw_x[9], fs=fs, nperseg=fs)
        delta_mask = (freqs >= 1.0) & (freqs <= 4.0)
        alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
        delta_power = np.trapezoid(psd[delta_mask], freqs[delta_mask])
        alpha_power = np.trapezoid(psd[alpha_mask], freqs[alpha_mask])
        delta_ratio = delta_power / (alpha_power + 1e-6)
        
        # 4. Electrical Chaos (Spectral Entropy)
        psd_norm = psd / np.sum(psd)
        spec_entropy = entropy(psd_norm)
        
        # -> NEW FIX: Scale the absolute voltage so it doesn't crash the neural network gradients
        scaled_absolute_v = burst_peak_to_peak / 1000.0 
        
        # -> CRITICAL FIX: Ensure this array contains exactly 4 variables!
        physics_array = np.array([scaled_absolute_v, relative_amplitude, delta_ratio, spec_entropy], dtype=np.float32)
        
        # ==========================================
        # SHAPE NORMALIZATION & DROPOUT
        # ==========================================
        q75, q25 = np.percentile(raw_x, [75, 25], axis=1, keepdims=True)
        iqr = q75 - q25
        median = np.median(raw_x, axis=1, keepdims=True)
        
        normalized_x = (raw_x - median) / (iqr + 1e-6)
        normalized_x[~good_channels_mask] = 0.0
        
        if np.random.rand() < 0.5:
            num_drop = np.random.randint(1, 4) 
            drop_indices = np.random.choice(19, num_drop, replace=False)
            normalized_x[drop_indices] = 0.0
            
        # ==========================================
        # TENSOR CONVERSION
        # ==========================================
        x = torch.tensor(normalized_x, dtype=torch.float32)
        physics_tensor = torch.tensor(physics_array, dtype=torch.float32)
        y_clinical = torch.tensor(self.labels[idx], dtype=torch.long)
        y_patient = torch.tensor(self.patient_labels[idx], dtype=torch.long)
        
        # Now yielding 4 items to match the new train_loader unpacking
        return x, physics_tensor, y_clinical, y_patient