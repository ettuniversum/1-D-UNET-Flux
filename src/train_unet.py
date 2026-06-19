import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data_processing_unet import PPGSeq2SeqDataset
from model_unet import UNet1D

def plot_visual_debugger(model, dataset, device, out_path="models/unet_debug_plot.png"):
    model.eval()
    
    # Pick a random sample
    idx = np.random.randint(0, len(dataset))
    x, y_true = dataset[idx]
    
    x_batch = x.unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred = model(x_batch).squeeze().cpu().numpy()
        
    x_np = x.squeeze().numpy()
    y_true_np = y_true.squeeze().numpy()
    
    plt.figure(figsize=(12, 6))
    
    # Plot normalized input PPG
    plt.plot(x_np, color='gray', alpha=0.5, label='Input PPG (Z-scored)')
    
    # Plot Ground Truth
    plt.step(range(len(y_true_np)), y_true_np, where='post', color='blue', linewidth=2, label='Ground Truth AUC')
    
    # Plot Predicted
    plt.step(range(len(y_pred)), y_pred, where='post', color='red', linewidth=2, linestyle='--', label='Predicted AUC')
    
    plt.title("U-Net Visual Debugger: Continuous Seq2Seq Regression")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Visual debugger plot saved to {out_path}")


def train_unet(model, train_loader, val_loader, epochs=10, learning_rate=1e-3, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    models_dir = "models" if os.path.exists("models") else "../models"
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_unet1d.pth")
    
    best_val_loss = float('inf')
    
    print("Starting U-Net training loop...")
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best U-Net model checkpoint to {best_model_path}")
            
    print(f"Training complete. Best model saved as '{best_model_path}'.")
    return best_model_path

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = "ptt-ppg" if os.path.exists("ptt-ppg") else "../ptt-ppg"
    print("Loading U-Net Dataset...")
    
    torch.manual_seed(42)
    dataset = PPGSeq2SeqDataset(data_dir=data_dir, window_size=1000, step_size=500)
    
    if len(dataset) == 0:
        print("Dataset empty. Exiting.")
        exit(0)
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = UNet1D(in_channels=1, out_channels=1).to(device)
    
    # Train
    best_path = train_unet(model, train_loader, val_loader, epochs=10, learning_rate=1e-3, device=device)
    
    # Visual Debugger
    model.load_state_dict(torch.load(best_path, map_location=device))
    models_dir = "models" if os.path.exists("models") else "../models"
    debug_path = os.path.join(models_dir, "unet_debug_plot.png")
    plot_visual_debugger(model, val_dataset, device, out_path=debug_path)
