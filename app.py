"""Brain Tumor Gradio App"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# =========================
# PATHS
# =========================
MODEL_PATH = r"E:\Desktop\CLASS\VI\DL\Project\models\best_brain_tumor_model.pth"
CLASS_FILE = r"E:\Desktop\CLASS\VI\DL\Project\class_names.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# =========================
# LOAD CLASSES
# =========================
with open(CLASS_FILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# =========================
# MODEL
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# PREDICT
# =========================
def predict(image):
    image = image.convert("RGB")
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]

    probs = probs.cpu().numpy()
    idx = probs.argmax()
    confidence = probs[idx] * 100
    label = class_names[idx]

    descriptions = {
        "glioma": "Tumor in brain glial cells.",
        "meningioma": "Tumor in brain covering layers.",
        "pituitary": "Tumor in hormone gland.",
        "notumor": "No tumor detected."
    }

    result = f"""
 Prediction: {label.upper()}

 Confidence: {confidence:.2f}%

 Explanation:
{descriptions[label]}

 Disclaimer:
Not a medical diagnosis.
"""
    return result

# =========================
# UI
# =========================
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="Brain Tumor Classifier",
    description="Upload MRI Image"
)

if __name__ == "__main__":
    app.launch()