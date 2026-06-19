import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt

def plot_30s_sample(record_path, out_file="models/30s_pleth_sample.png"):
    # Load the WFDB record
    record = wfdb.rdrecord(record_path)
    
    # Identify the correct pleth channel
    if 'pleth_1' in record.sig_name:
        idx = record.sig_name.index('pleth_1')
    elif 'pleth' in record.sig_name:
        idx = record.sig_name.index('pleth')
    else:
        idx = 0
        
    signal = record.p_signal[:, idx]
    fs = record.fs  # Sampling frequency
    
    # Calculate number of samples for 30 seconds
    duration_sec = 30
    num_samples = int(duration_sec * fs)
    
    # Ensure the signal has enough samples
    if len(signal) > num_samples:
        signal_slice = signal[:num_samples]
    else:
        signal_slice = signal
        
    # Create time axis in seconds
    time_axis = np.arange(len(signal_slice)) / fs
    
    # Plotting
    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, signal_slice, color='red', linewidth=1.5)
    plt.title(f"30-Second PPG (Pleth) Signal Sample (fs={fs}Hz)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    plt.savefig(out_file)
    print(f"Saved 30-second plot to {out_file}")

if __name__ == "__main__":
    # Choose a sample record from the ptt-ppg folder
    data_dir = "ptt-ppg" if os.path.exists("ptt-ppg") else "../ptt-ppg"
    sample_record = os.path.join(data_dir, "s10_run")
    
    models_dir = "models" if os.path.exists("models") else "../models"
    out_path = os.path.join(models_dir, "30s_pleth_sample.png")
    
    plot_30s_sample(sample_record, out_file=out_path)
