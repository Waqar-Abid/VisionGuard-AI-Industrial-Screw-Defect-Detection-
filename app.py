
import os
from io import BytesIO

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename

# 🔧 Matplotlib in web/server mode (NO GUI/Tkinter)
import matplotlib
matplotlib.use("Agg")             # <-- important: set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import base64
import io

# -------------------------
# Config (match your notebooks)
# -------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 6
weights_path = "best_model.pth"   # put your trained weights here

# Classes (your order from ImageFolder)
class_names = ["good", "manipulated_front", "scratch_head",
               "scratch_neck", "thread_side", "thread_top"]

# 04 (image) transforms — your eval pipeline
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
test_tfms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 05 (video) transforms — EXACTLY like your realtime notebook
video_tfms = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# -------------------------
# Model (ResNet50 head swap, load your weights)
# -------------------------
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
state = torch.load(weights_path, map_location=device)
model.load_state_dict(state)
model = model.to(device)
model.eval()
print("✅ Model loaded and ready!")

# -------------------------
# Inference helpers
# -------------------------
def predict_pil_image(pil_img):
    """Single image prediction (04 style)."""
    x = test_tfms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        _, pred = torch.max(logits, 1)
    return class_names[pred.item()]

def generate_video_stream(video_path):
    """
    Stream processed frames (05 style): resize->PIL->ToTensor->Normalize(0.5).
    Draw label with cv2.putText. Yields MJPEG bytes.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    frame_skip = 3
    frame_id = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1
        if frame_id % frame_skip != 0:
            continue

        # ---- PREPROCESS (match notebook) ----
        img = cv2.resize(frame, (224, 224))
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        x = video_tfms(img_pil).unsqueeze(0).to(device)

        # ---- PREDICT ----
        with torch.no_grad():
            logits = model(x)
            _, pred = torch.max(logits, 1)
        label = class_names[pred.item()]

        # ---- DRAW ----
        cv2.putText(frame, f"Pred: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ---- ENCODE & YIELD ----
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# ---- Image upload -> show prediction + image (web-safe, repeatable) ----
@app.route("/predict-image", methods=["POST"])
def predict_image_route():
    if "image" not in request.files:
        return redirect(url_for("index"))
    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    # Use your notebook logic (no Tkinter, web-only)
    pil_img = Image.open(file.stream).convert("RGB")

    # predict (same as notebook)
    img_t = test_tfms(pil_img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img_t)
        _, pred = torch.max(outputs, 1)
    prediction = class_names[pred.item()]

    # Render result image using matplotlib Agg, return as base64
    fig, ax = plt.subplots()
    ax.imshow(pil_img)
    ax.axis("off")
    ax.set_title(f"Prediction: {prediction}", fontsize=14)
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg", bbox_inches="tight")
    plt.close(fig)
    img_bytes = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render_template("index.html",
                           image_label=prediction,
                           image_bytes=img_bytes)

# ---- Video upload -> play processed stream (UNCHANGED) ----
@app.route("/upload-video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return redirect(url_for("index"))
    file = request.files["video"]
    if file.filename == "":
        return redirect(url_for("index"))

    fname = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, fname)
    file.save(path)
    return redirect(url_for("play_video", filename=fname))

@app.route("/play/<filename>")
def play_video(filename):
    return render_template("play_video.html", filename=filename)

@app.route("/video-feed/<filename>")
def video_feed(filename):
    video_path = os.path.join(UPLOAD_DIR, filename)
    return Response(generate_video_stream(video_path),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
