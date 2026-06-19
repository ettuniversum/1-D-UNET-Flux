import os
import torch
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

from model_unet import UNet1D

def export_tflite():
    models_dir = "models" if os.path.exists("models") else "../models"
    pth_path = os.path.join(models_dir, "best_unet1d.pth")
    onnx_path = os.path.join(models_dir, "best_unet1d.onnx")
    tf_path = os.path.join(models_dir, "best_unet1d_tf")
    tflite_path = os.path.join(models_dir, "best_unet1d.tflite")
    
    device = torch.device('cpu')
    
    print("1. Loading PyTorch model...")
    model = UNet1D(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    
    print("2. Exporting to ONNX...")
    dummy_input = torch.randn(1, 1, 1000)
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'], 
                      opset_version=13)
                      
    print("3. Converting ONNX to TensorFlow SavedModel...")
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)
    
    print("4. Converting TensorFlow SavedModel to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    # Optional: Quantization could be added here
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Successfully exported TFLite model to {tflite_path}")

if __name__ == "__main__":
    export_tflite()
