import os
import torch
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from model_unet import UNet1D

def plot_30s_unet_predictions():
    device = torch.device("cpu")
    models_dir = "models" if os.path.exists("models") else "../models"
    model_path = os.path.join(models_dir, "best_unet1d.pth")
    
    tflite_path = os.path.join(models_dir, "best_unet1d.tflite")
    use_tflite = os.path.exists(tflite_path)
    
    if use_tflite:
        import tensorflow as tf
        print(f"Loading TFLite model from {tflite_path}")
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    else:
        # Load U-Net
        model = UNet1D(in_channels=1, out_channels=1)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"Loaded trained PyTorch U-Net model from {model_path}")
        else:
            print("U-Net model not found. Run train_unet.py first.")
            return

    # Load Signal
    data_dir = "ptt-ppg" if os.path.exists("ptt-ppg") else "../ptt-ppg"
    sample_record = os.path.join(data_dir, "s10_run")
    record = wfdb.rdrecord(sample_record)
    
    idx = 0
    if 'pleth_1' in record.sig_name:
        idx = record.sig_name.index('pleth_1')
    elif 'pleth' in record.sig_name:
        idx = record.sig_name.index('pleth')
        
    signal = record.p_signal[:, idx]
    fs = record.fs
    
    num_samples = int(30 * fs)
    signal_slice = signal[:num_samples]
    
    # U-Net sequence length = 1000
    window_size = 1000
    num_windows = len(signal_slice) // window_size
    
    time_axis = np.arange(num_windows * window_size) / fs
    signal_slice = signal_slice[:num_windows * window_size]
    
    predictions = np.zeros(num_windows * window_size)
    
    # Slide model across the 30s
    if not use_tflite:
        with torch.no_grad():
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                window = signal_slice[start:end]
                
                mean = np.mean(window)
                std = np.std(window) + 1e-8
                window_norm = (window - mean) / std
                
                x = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                pred = model(x).squeeze().cpu().numpy()
                predictions[start:end] = pred
    else:
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = signal_slice[start:end]
            
            mean = np.mean(window)
            std = np.std(window) + 1e-8
            window_norm = (window - mean) / std
            
            x_np = window_norm.astype(np.float32)[np.newaxis, :, np.newaxis]
            interpreter.set_tensor(input_details[0]['index'], x_np)
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]['index']).squeeze()
            predictions[start:end] = pred
            
    # Aesthetic Theming
    fig, ax1 = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#121212')
    ax1.set_facecolor('#121212') 
    
    # Find true physiological peaks/troughs
    from scipy.signal import find_peaks
    troughs, _ = find_peaks(-signal_slice, distance=150)
    
    # Aesthetic Theming
    fig, ax1 = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#121212')
    ax1.set_facecolor('#121212') 
    
    # --- 1. Color Gradient for AUC ---
    norm = Normalize(vmin=np.min(predictions), vmax=np.max(predictions))
    cmap = plt.cm.plasma
    
    # Plot the raw signal
    ax1.plot(time_axis, signal_slice, color='white', linewidth=1.5, alpha=1.0, zorder=3)
    
    # Interpolate a baseline representing the bottom of the curve
    # physically connecting the troughs, just like the integral math does
    baseline = np.interp(time_axis, time_axis[troughs], signal_slice[troughs])
    
    # Plot the relative line under the curve
    ax1.plot(time_axis, baseline, color='#aaaaaa', linewidth=1.2, linestyle='--', alpha=0.8, zorder=2, label="Relative Baseline (Zero-Flux)")
    
    # Create the heatmap under each individual peak
    if len(troughs) >= 2:
        for i in range(len(troughs) - 1):
            start_t = troughs[i]
            end_t = troughs[i+1]
            
            # Sample the U-Net's predicted AUC value for this specific heartbeat
            mid_t = start_t + (end_t - start_t) // 2
            pred_auc = predictions[mid_t]
            color = cmap(norm(pred_auc))
            
            # Shade ONLY between the baseline and the signal for this specific peak
            ax1.fill_between(time_axis[start_t:end_t+1], 
                             baseline[start_t:end_t+1], 
                             signal_slice[start_t:end_t+1], 
                             color=color, alpha=0.85, edgecolor='none', zorder=1)
    
    ax1.set_title("30s Heartbeat Sequence | Physiological Heatmap Overlay", color='white', fontsize=16, pad=15)
    ax1.set_xlabel("Time (seconds)", color='white', fontsize=12)
    ax1.set_ylabel("Raw PPG Amplitude", color='white', fontsize=12)
    ax1.grid(True, alpha=0.15, color='white')
    ax1.tick_params(colors='white')
    ax1.legend(loc="upper right", facecolor='#121212', edgecolor='white', labelcolor='white')
    
    # Colorbar mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, orientation='vertical', pad=0.08)
    cbar.set_label("U-Net Predicted Volume (AUC Magnitude)", color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    out_path = os.path.join(models_dir, "30s_unet_overlaid.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Saved refined gradient plot to {out_path}")

if __name__ == "__main__":
    plot_30s_unet_predictions()
