import os
import glob
import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

class PPGSeq2SeqDataset(Dataset):
    """
    Continuous Sequence-to-Sequence Dataset for 1-D U-Net.
    Outputs [Batch, 1, 1000] inputs and [Batch, 1, 1000] stepped target AUC labels.
    """
    def __init__(self, data_dir, window_size=1000, step_size=500, transform=None):
        self.window_size = window_size
        self.step_size = step_size
        self.windows = []
        self.targets = []
        
        self._load_and_preprocess_data(data_dir)

    def _load_and_preprocess_data(self, data_dir):
        files = glob.glob(os.path.join(data_dir, '**', '*.hea'), recursive=True)
        if not files:
            print(f"Warning: No .hea files found in {data_dir}.")
            return

        all_signals = []
        for file in files:
            try:
                record_name = file[:-4]
                record = wfdb.rdrecord(record_name)
                
                if 'pleth_1' in record.sig_name:
                    idx = record.sig_name.index('pleth_1')
                    signal = record.p_signal[:, idx]
                elif 'pleth' in record.sig_name:
                    idx = record.sig_name.index('pleth')
                    signal = record.p_signal[:, idx]
                else:
                    signal = record.p_signal[:, 0]

                signal = signal[~np.isnan(signal)]
                all_signals.append(signal)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        # Extract sliding windows and calculate continuous stepped targets
        for signal in all_signals:
            if len(signal) < self.window_size:
                continue
            
            for start_idx in range(0, len(signal) - self.window_size + 1, self.step_size):
                window = signal[start_idx : start_idx + self.window_size]
                
                # Dynamic Target Generation
                # 1. Find the troughs (valleys) using find_peaks on -window
                # distance=200 means we assume at least 200 samples (~0.4s at 500Hz) between heartbeats
                inverted_window = -window
                troughs, _ = find_peaks(inverted_window, distance=150) # 150 = 0.3s (safe for HR up to 200bpm)
                
                target = np.zeros(self.window_size)
                
                if len(troughs) >= 2:
                    # 2. Iterate through pairs of consecutive troughs
                    for i in range(len(troughs) - 1):
                        start_t = troughs[i]
                        end_t = troughs[i+1]
                        
                        segment = window[start_t:end_t]
                        
                        # 3. Remove baseline wander for the segment
                        clean_segment = segment - np.min(segment)
                        
                        # 4. Calculate AUC
                        auc = trapezoid(clean_segment)
                        
                        # 5. Fill the target array
                        target[start_t:end_t] = auc
                        
                    # 6. Edge Handling
                    # Forward-fill before the first trough using the first computed AUC
                    first_trough = troughs[0]
                    target[:first_trough] = target[first_trough:troughs[1]][0] if len(troughs) > 1 else 0
                    
                    # Backward-fill after the last trough using the last computed AUC
                    last_trough = troughs[-1]
                    target[last_trough:] = target[troughs[-2]:last_trough][0] if len(troughs) > 1 else 0
                else:
                    # If not enough troughs, fallback to whole window AUC
                    clean_segment = window - np.min(window)
                    target[:] = trapezoid(clean_segment)
                
                self.windows.append(window)
                self.targets.append(target)
                
        if len(self.windows) > 0:
            self._normalize()

    def _normalize(self):
        self.windows = np.array(self.windows)
        self.targets = np.array(self.targets)

        # Normalize windows (Z-score)
        self.window_scaler = StandardScaler()
        self.windows = self.window_scaler.fit_transform(self.windows)

        # Normalize targets (Min-Max globally so regression deals with 0-1 values ideally)
        self.target_scaler = MinMaxScaler()
        # Reshape to 1D for global scaling across the entire dataset, then back to (N, window_size)
        targets_flat = self.targets.reshape(-1, 1)
        targets_flat_scaled = self.target_scaler.fit_transform(targets_flat)
        self.targets = targets_flat_scaled.reshape(self.targets.shape)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Format: [Channels, Length] -> [1, 1000]
        x = torch.tensor(self.windows[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
        return x, y
