# Brain Tumor Detection using ResNeXt & Transfer Learning - 94.2% Accuracy

## Problem
MRI brain tumor dataset has 4 classes (Glioma, Meningioma, Pituitary, No Tumor). Training from scratch gives only 81% accuracy due to less medical data.

## My Solution
Used ResNeXt101 pretrained on ImageNet. Applied Transfer Learning - freezed base layers, added custom dense layers. Used data augmentation (rotation, zoom, flip) to increase dataset.

## Results
- Baseline Custom CNN: 81.4% accuracy
- With ResNeXt + Transfer Learning: 94.2% accuracy
- F1-Score: 0.93
- Tech: Python, TensorFlow, Keras, ResNeXt, ImageNet

## How to Run
