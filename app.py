import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

from training.model import DeepfakeModel

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load Model
# -----------------------------
model = DeepfakeModel().to(device)
model.load_state_dict(
    torch.load("models/deepfake_model.pth", map_location=device)
)
model.eval()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict(image):
    if image is None:
        return "No image uploaded", ""

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = "FAKE" if pred.item() == 1 else "REAL"
    confidence_percent = confidence.item() * 100

    return label, f"{confidence_percent:.2f}%"

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    # ---------- HEADER (HTML, new-tab links) ----------
    gr.HTML(
        """
        <h1>🧠 Deepfake Detection Using Machine Learning and Deep Learning</h1>

        <p><b>Developed by:</b> Ravi Sinha</p>

        <p>
        🔗 <b>GitHub:</b>
        <a href="https://github.com/RaviSinha158" target="_blank">
            https://github.com/RaviSinha158
        </a><br>
        🔗 <b>LinkedIn:</b>
        <a href="https://www.linkedin.com/in/ravisinhaio/" target="_blank">
            https://www.linkedin.com/in/ravisinhaio/
        </a>
        </p>

        <p>
        Upload an image to check whether it is <b>REAL</b> or <b>FAKE</b> using a trained deep learning model.
        </p>
        """
    )

    # ---------- INPUT ----------
    with gr.Row():
        image_input = gr.Image(
            type="pil",
            label="Upload Image"
        )

    # ---------- OUTPUT ----------
    with gr.Row():
        prediction = gr.Textbox(
            label="Prediction",
            interactive=False
        )
        confidence = gr.Textbox(
            label="Confidence",
            interactive=False
        )

    # ---------- BUTTON ----------
    analyze_btn = gr.Button("Analyze")

    analyze_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[prediction, confidence]
    )

    # ---------- FOOTER ----------
    gr.HTML(
        """
        <hr>
        <p style="text-align:center;">
        👨‍💻 Developed by <b>Ravi Sinha</b><br>
        🌐 Public Demo hosted on <b>Hugging Face Spaces</b>
        </p>
        """
    )

# -----------------------------
# Launch App
# -----------------------------
demo.launch()
