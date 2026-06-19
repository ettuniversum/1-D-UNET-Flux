import os
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf

from model_unet import UNet1D

def export_tflite():
    models_dir = "models" if os.path.exists("models") else "../models"
    pth_path = os.path.join(models_dir, "best_unet1d.pth")
    tflite_path = os.path.join(models_dir, "best_unet1d.tflite")
    
    device = torch.device('cpu')
    
    print("1. Loading PyTorch model...")
    pt_model = UNet1D(in_channels=1, out_channels=1)
    if os.path.exists(pth_path):
        pt_model.load_state_dict(torch.load(pth_path, map_location=device))
        pt_model.eval()
    else:
        print("PyTorch model not found!")
        return

    print("2. Building equivalent Keras Model...")
    def double_conv(x, filters):
        x = tf.keras.layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters, 3, padding='same', activation='relu')(x)
        return x

    inputs = tf.keras.Input(shape=(1000, 1))

    # Encoder
    c1 = double_conv(inputs, 32)
    p1 = tf.keras.layers.MaxPooling1D(2, strides=2, padding='same')(c1)

    c2 = double_conv(p1, 64)
    p2 = tf.keras.layers.MaxPooling1D(2, strides=2, padding='same')(c2)

    c3 = double_conv(p2, 125)

    # Decoder
    u1 = tf.keras.layers.Conv1DTranspose(64, 2, strides=2, padding='same')(c3)
    u1 = tf.keras.layers.Concatenate()([u1, c2])
    c4 = double_conv(u1, 64)

    u2 = tf.keras.layers.Conv1DTranspose(32, 2, strides=2, padding='same')(c4)
    u2 = tf.keras.layers.Concatenate()([u2, c1])
    c5 = double_conv(u2, 32)

    outputs = tf.keras.layers.Conv1D(1, 1)(c5)

    keras_model = tf.keras.Model(inputs, outputs)

    print("3. Transferring weights from PyTorch to Keras...")
    pt_convs = []
    for m in pt_model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
            pt_convs.append(m)
            
    tf_convs = []
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D) or isinstance(layer, tf.keras.layers.Conv1DTranspose):
            tf_convs.append(layer)
            
    assert len(pt_convs) == len(tf_convs), f"Conv layers mismatch! PT: {len(pt_convs)}, TF: {len(tf_convs)}"
    
    for pt_layer, tf_layer in zip(pt_convs, tf_convs):
        pt_w = pt_layer.weight.detach().numpy()
        pt_b = pt_layer.bias.detach().numpy() if pt_layer.bias is not None else None
        
        if isinstance(pt_layer, nn.Conv1d):
            # PyTorch: (out_channels, in_channels, kernel_size)
            # Keras: (kernel_size, in_channels, out_channels)
            tf_w = np.transpose(pt_w, (2, 1, 0))
        elif isinstance(pt_layer, nn.ConvTranspose1d):
            target_shape = tf_layer.get_weights()[0].shape
            if target_shape == (pt_w.shape[2], pt_w.shape[1], pt_w.shape[0]):
                tf_w = np.transpose(pt_w, (2, 1, 0))
            elif target_shape == (pt_w.shape[2], pt_w.shape[0], pt_w.shape[1]):
                tf_w = np.transpose(pt_w, (2, 0, 1))
            else:
                raise ValueError(f"Shape mismatch: PT {pt_w.shape} vs TF {target_shape}")
        
        if pt_b is not None:
            tf_layer.set_weights([tf_w, pt_b])
        else:
            tf_layer.set_weights([tf_w])

    print("4. Validating predictions match...")
    dummy_input_np = np.random.randn(1, 1000, 1).astype(np.float32)
    pt_dummy = torch.tensor(dummy_input_np.transpose(0, 2, 1))
    pt_out = pt_model(pt_dummy).detach().numpy().transpose(0, 2, 1)
    tf_out = keras_model(dummy_input_np).numpy()
    
    diff = np.max(np.abs(pt_out - tf_out))
    print(f"Max absolute difference between PyTorch and Keras: {diff:.6f}")
    assert diff < 1e-4, "Predictions do not match!"

    print("5. Converting Keras model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Successfully exported TFLite model to {tflite_path}")

if __name__ == "__main__":
    export_tflite()
