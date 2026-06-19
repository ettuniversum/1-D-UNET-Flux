import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
import tensorflow as tf
import onnx2tf
import shutil

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
                      opset_version=13, dynamo=False)
                      
    print("3. Converting ONNX to TensorFlow SavedModel / TFLite...")
    # Run onnx2tf conversion
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_path,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    
    # Locate the generated float32 TFLite model and copy it to the desired path
    generated_tflite = os.path.join(tf_path, "best_unet1d_float32.tflite")
    if os.path.exists(generated_tflite):
        shutil.copy(generated_tflite, tflite_path)
        print(f"Copied TFLite model to {tflite_path}")
    else:
        print(f"Error: Generated TFLite file not found at {generated_tflite}")
        return
        
    print("4. Verifying model outputs...")
    # Generate test input: [Batch, Channel, Length] = [1, 1, 1000]
    test_input_pt = torch.randn(1, 1, 1000)
    test_input_np = test_input_pt.numpy()
    
    # Run PyTorch prediction
    with torch.no_grad():
        out_pt = model(test_input_pt).numpy()
        
    # Run ONNX Runtime prediction
    ort_session = ort.InferenceSession(onnx_path)
    out_onnx = ort_session.run(None, {'input': test_input_np})[0]
    
    # Run TFLite prediction
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"TFLite Input Details: Shape={input_details[0]['shape']}, Type={input_details[0]['dtype']}")
    print(f"TFLite Output Details: Shape={output_details[0]['shape']}, Type={output_details[0]['dtype']}")
    
    # Transpose input to NWC / NLC layout if required
    # (since onnx2tf converts NCW -> NWC)
    tflite_in_shape = input_details[0]['shape']
    if np.array_equal(tflite_in_shape, [1, 1000, 1]):
        test_input_tflite = np.transpose(test_input_np, (0, 2, 1))
    else:
        test_input_tflite = test_input_np
        
    interpreter.set_tensor(input_details[0]['index'], test_input_tflite)
    interpreter.invoke()
    out_tflite = interpreter.get_tensor(output_details[0]['index'])
    
    # Transpose TFLite output back to NCL layout if it was converted to NLC
    if out_tflite.shape == (1, 1000, 1):
        out_tflite_ncl = np.transpose(out_tflite, (0, 2, 1))
    else:
        out_tflite_ncl = out_tflite
        
    # Compare outputs
    diff_onnx = np.max(np.abs(out_pt - out_onnx))
    diff_tflite = np.max(np.abs(out_pt - out_tflite_ncl))
    
    print(f"Max absolute difference PyTorch vs ONNX: {diff_onnx:.6e}")
    print(f"Max absolute difference PyTorch vs TFLite: {diff_tflite:.6e}")
    
    if diff_tflite < 1e-3:
        print("Verification SUCCESSFUL! Predictions match closely.")
    else:
        print("Warning: Verification completed with significant difference.")

if __name__ == "__main__":
    export_tflite()
