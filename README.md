## 👨‍💻 Developer

**Ravi Sinha**  
BCA Student, Manipal University Jaipur  

🔗 **GitHub:** https://github.com/RaviSinha158  
🔗 **LinkedIn:** https://www.linkedin.com/in/ravisinhaio/


## 🌐 Public Demo

A public, lightweight demo of the trained deepfake detection model is deployed
using Hugging Face Spaces for easy access and testing.

🔗 **Live Demo:**  
https://huggingface.co/spaces/BlestR/Deepfake_Detection_Using_ML_and_DL_Techniques

The main Flask-based interface is intended for local academic demonstration,
screenshots, and evaluation.




# Deepfake Detection Using Machine Learning and Deep Learning

A web-based Deepfake Detection System developed using Machine Learning and
Deep Learning techniques. The system is capable of identifying whether an
uploaded image or video contains REAL or FAKE (deepfake / AI-generated) content.
It also provides confidence scores and a user-friendly web interface for
demonstration and academic use.

---

## 🚀 Project Highlights

- Trained on a large-scale deepfake dataset containing **140,000+ images**
- Achieved **~91% validation accuracy** on unseen data
- Supports both **image and video-based deepfake detection**
- Web-based interactive interface using Flask
- GPU acceleration supported via PyTorch (CUDA)

---

## ✨ Features

- Deepfake detection using a trained CNN-based deep learning model
- Supports image formats:
  - JPG, PNG, WEBP
  - HEIC / HEIF (iPhone images)
- Supports video formats:
  - MP4 (first-frame based analysis)
- Displays prediction result (REAL / FAKE)
- Confidence score visualization
- Simple and easy-to-deploy architecture

---

## 🧠 Model & Training Details

- Architecture: Convolutional Neural Network (CNN)
- Framework: PyTorch
- Training Dataset Size:
  - 100,000 training images (Real + Fake)
  - 20,000 validation images
  - Additional test samples for evaluation
- Achieved Performance:
  - Validation Accuracy: **~91.29%**
- Training performed using GPU acceleration for faster convergence

The model was trained using supervised learning on publicly available deepfake
datasets containing real and AI-generated facial images.

---

## 🧪 Video Deepfake Detection

Video deepfake detection is implemented by extracting a representative frame
from the uploaded video. This frame is then passed to the trained image-based
deepfake detection model for classification.

This approach provides efficient inference while maintaining good accuracy
for academic and demonstration purposes.

---

## 🛠 Technology Stack

- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Image & Video Processing**: OpenCV, Pillow
- **Web Framework**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Model Deployment**: Local Flask server

---

## 📂 Project Structure

deepfake-original/
├── backend/
│ └── predict.py
├── frontend_flask/
│ ├── app.py
│ ├── templates/
│ │ └── index.html
│ └── static/
│ └── uploads/
├── training/
│ ├── model.py
│ ├── train.py
│ └── dataset.py
├── models/
│ └── deepfake_model.pth
├── data/
│ ├── train/
│ ├── val/
│ └── test/
├── LICENSE
├── README.md
└── requirements.txt


---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection

2. Create a virtual environment
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

▶️ Run the Web Application
python frontend_flask/app.py


Open your browser and visit:

http://127.0.0.1:5000


📌 Usage

Upload an image or video file

The system analyzes the input

Displays:

Prediction: REAL / FAKE

Confidence score

Preview of uploaded content

⚠️ Limitations

Predictions may vary for low-quality or highly compressed images

Model performance may degrade on unseen deepfake generation techniques

Video detection is frame-based and does not analyze temporal consistency

📄 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this software with attribution.

🙌 Acknowledgements

PyTorch Community

Open-source deepfake datasets used for academic research

OpenCV and Flask contributors

⭐ If you find this project useful, feel free to star the repository!
