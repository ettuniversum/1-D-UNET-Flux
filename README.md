# 1-D-CNN-Flux

## PTT-PPG Dataset
This repository contains the code and resources for working with the PTT-PPG dataset. The `ptt-ppg` folder (ignored in version control) contains the raw dataset files.

## Project Goal
The primary objective of this project is to create a 1-Dimensional Convolutional Neural Network (1-D CNN) to predict or calculate the Area Under the Curve (AUC) of the Photoplethysmogram (PPG) signal. Measuring the AUC of the PPG signal allows us to estimate and monitor blood volume changes.

## Raw PPG Signal
The raw physiological dataset consists of continuous PPG windows, sliced into exact 1000-sample sequences (e.g. roughly 10 seconds at 100Hz).
![Raw 30s PPG Signal](models/30s_pleth_sample.png)

## U-Net Architecture & Predictions
We successfully transitioned from a standard ResNet architecture to a continuous sequence-to-sequence **1-D U-Net**. This architecture accepts the raw 1000-sample window and outputs a continuous 1000-sample array, forming a stepped, relative baseline under the peaks that dynamically visualizes individual heartbeat AUC (blood volume). 
![U-Net Overlay](models/30s_unet_overlaid.png)

## Model Evaluation
Our validation process directly evaluates how well the generated sequence aligns with calculated physiological targets. Below is the loss plot tracking predicted vs actual performance throughout the validation set.
![Scatter Actual vs Predicted](models/scatter_actual_vs_predicted.png)
