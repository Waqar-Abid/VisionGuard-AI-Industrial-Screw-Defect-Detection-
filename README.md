# 🏭 VisionGuard AI — Industrial Screw Defect Detection


##  Project Overview
**VisionGuard** is an AI-powered defect detection system designed for **industrial screws**.  
It leverages a **ResNet50 deep learning model** trained on the **MVTec Screw dataset** to automatically classify screws into six categories:

- good (non-defective)
- manipulated_front
- scratch_head
- scratch_neck
- thread_side
- thread_top  

The system supports:
- **Dataset exploration & visualization**
- **Model training & evaluation**
- **Real-time video & webcam detection**
- **Web-based interface for image & video uploads**

This helps manufacturers streamline quality control with fast, reliable defect detection.

---
## Live Demo
![Live Demo](image-1.png)

---
##  Purpose
The purpose of this repository is to demonstrate how **deep learning can automate quality inspection** in manufacturing pipelines by identifying defects in screws with high accuracy.

---

##  Features
- **Data Exploration**: Class distribution visualization, sample grids, and heatmaps.
- **Data Preparation**: Train/validation/test splits, augmentations, and loaders.
- **Model Training**: Transfer learning with ResNet50 and fine-tuning for screw defects.
- **Evaluation**: Confusion matrix, classification reports, and test accuracy.
- **Real-Time Detection**: Live prediction on video streams or webcams.
- **Web Interface**: Upload images/videos for automated classification.

---

##  Architecture
![](image.png)

**Pipeline:**
1. **Dataset (MVTec Screw)** → split into `train / val / test` (6 classes).
2. **Data Preparation** → PyTorch `ImageFolder`, augmentations, loaders.
3. **Model Training** → ResNet50 pretrained on ImageNet → fine-tuned for defect detection.
4. **Model Evaluation** → Accuracy, Precision, Recall, F1, Confusion Matrix.
5. **Deployment**:
   - **Web App** (Flask + HTML template for image/video upload).
   

---

##  Prerequisites
- Python 3.10+
- PyTorch & TorchVision
- OpenCV
- Matplotlib, Seaborn
- Flask (for web app)

---

##  Setup Instructions
### 1. Clone & setup:

```
git clone https://github.com/<your-username>
VisionGuard.git
cd VisionGuard

```
### 2. Create a virtual environment:
```

python -m venv visionguard
source visionguard/bin/activate   # Linux/Mac
visionguard\Scripts\activate     # Windows

```
### 3. Install dependencies:

```
pip install -r requirements.txt
```
---
## Project Structure
```
VisionGuard/
├── 01-data-exploration.ipynb     # Dataset analysis & visualization
├── 02-data-preparation.ipynb     # Dataloaders, augmentations
├── 03-model-training.ipynb       # Training ResNet50
├── 04-model-evaluation.ipynb     # Evaluation & metrics
├── 05-realtime-detection.ipynb   # Live webcam/video defect detection
├── templates/
│   ├── index.html                # Upload interface (image/video)
│   └── play.html                 # Live video streaming page
├── static/                       # CSS/JS or saved images
├── best_model.pth                # Trained model checkpoint
├── model.py                      # Exported trained model
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── architecture.png              
```
---
## Configuration
### 1. Run Jupyter Notebooks
For training and evaluation:
```
jupyter notebook
```
### 2. Run Web App
Start the Flask server:

```
python app.py
```
Then open:

```
http://127.0.0.1:5000
```
---
## Results
Validation Accuracy: ~89%

Test Accuracy: ~84%


---
## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.