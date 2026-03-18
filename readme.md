
# Brain Tumor MRI Classification using ResNet18

A deep learning-based brain tumor MRI classification project built with PyTorch and deployed with Gradio for real-time prediction.

## Overview

This project classifies brain MRI images into four classes:
- glioma
- meningioma
- pituitary
- notumor

The model is based on transfer learning using a pretrained ResNet18 architecture.  
Users can upload an MRI image through a Gradio interface and receive:
- predicted class
- confidence score
- short explanation
- disclaimer

## Tech Stack

- Deep Learning Model: ResNet18
- Technique: Transfer Learning
- Framework: PyTorch
- Frontend / Demo: Gradio
- Image Preprocessing:
  - resizing to 224x224
  - tensor conversion
  - normalization

## Project Structure

```text
Project/
├── app.py
├── training.py
├── class_names.txt
├── requirements.txt
├── README.md
├── models/
│   └── best_brain_tumor_model.pth
└── archive/
    ├── Training/
    └── Testing/
````

## How It Works

1. MRI images are preprocessed using resizing, normalization, and tensor conversion.
2. A pretrained ResNet18 model is fine-tuned for 4-class classification.
3. The best model is saved after training.
4. A Gradio app loads the trained model and predicts the uploaded MRI image.

## Classes

* Glioma
* Meningioma
* Pituitary
* No Tumor

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python training.py
```

Run the Gradio app:

```bash
python app.py
```

## Sample Output

The app returns:

* Prediction label
* Confidence percentage
* Short medical-style explanation
* Disclaimer that it is not a medical diagnosis

## Notes

* Dataset used: Brain Tumor MRI Dataset
* Model type: ResNet18 transfer learning baseline
* Future extension: hybrid quantum-classical feature refinement

## Author

Yash
LinkedIn: [https://www.linkedin.com/in/yash-developer/](https://www.linkedin.com/in/yash-developer/)
