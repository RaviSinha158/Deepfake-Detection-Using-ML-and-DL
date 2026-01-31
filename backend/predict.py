import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

# Enable HEIC support (safe, does NOT affect RGB images)
from pillow_heif import register_heif_opener
register_heif_opener()

from training.model import DeepfakeModel

# ================= DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODEL ==================
model = DeepfakeModel().to(device)
model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=device))
model.eval()

# ================= TRANSFORM (UNCHANGED) ==
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= FACE CHECK (GATE ONLY) =
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def has_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )
    return len(faces) > 0

# ================= VIDEO SUPPORT ==========
def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    frame_path = video_path + "_frame.jpg"
    cv2.imwrite(frame_path, frame)
    return frame_path

# ================= PREDICT =================
def predict_image(image_path):
    """
    Returns:
    label,
    confidence,
    preview_filename
    """

    preview_path = image_path

    # 1️⃣ VIDEO → FRAME
    if image_path.lower().endswith((".mp4", ".mov", ".avi")):
        frame_path = extract_first_frame(image_path)
        if frame_path is None:
            return "ERROR", 0.0, None
        image_path = frame_path
        preview_path = frame_path

    # 2️⃣ FACE CHECK (NO CROPPING)
    if not has_face(image_path):
        return "NO FACE DETECTED", 0.0, os.path.basename(preview_path)

    # 3️⃣ LOAD IMAGE (FULL IMAGE AS TRAINED)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # 4️⃣ MODEL INFERENCE
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    # ⚠️ SAME LABEL LOGIC YOU SAID WORKED
    label = "FAKE" if pred.item() == 1 else "REAL"

    return label, float(confidence.item()), os.path.basename(preview_path)
