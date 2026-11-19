
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Keep this order identical to the training notebook.
DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def load_model(checkpoint_path="model/densenet121_chexpert.pth"):
    """Load the trained DenseNet-121 checkpoint."""
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(
        model.classifier.in_features,
        len(DISEASES)
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    # Handle either a raw state_dict or a wrapped checkpoint.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model


def predict_xray(image_path, model):
    """Return experimental probability scores for five findings."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)[0].cpu().tolist()

    return {
        "predictions": {
            disease: round(float(probability), 4)
            for disease, probability in zip(DISEASES, probabilities)
        },
        "model_note": (
            "Experimental educational model. These scores are not "
            "diagnoses and must not guide clinical decisions."
        ),
    }
