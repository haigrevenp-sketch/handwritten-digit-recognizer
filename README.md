# 🤖 Handwritten Digit Recognizer using CNN

> A production-grade AI web application built with Flask + TensorFlow/Keras that recognises handwritten digits (0–9) from a drawing canvas or uploaded image, complete with Grad-CAM explainability, PDF reports, SQLite history, and a stunning glassmorphism UI.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **Draw to Predict** | 280×280 interactive canvas with brush-size control and undo |
| **Upload to Predict** | PNG / JPG drag-and-drop with image preview |
| **Real-time Prediction** | Live digit suggestion while drawing |
| **Top-3 Results** | Predicted digit + confidence + animated bar chart |
| **Grad-CAM** | Heatmap showing which pixels influenced the prediction |
| **Confidence Chart** | Bar chart of all 10 class probabilities |
| **PDF Report** | Downloadable per-prediction report via ReportLab |
| **SQLite History** | All predictions persisted; browsable in-app table |
| **Performance Dashboard** | Accuracy/loss curves, confusion matrix, sample predictions |
| **Dark / Light Mode** | Toggle; preference saved to `localStorage` |
| **Fully Responsive** | Mobile-first Bootstrap 5 layout |

---

## 🗂 Project Structure

```
digit_recognizer/
├── app.py                    # Flask backend + REST API
├── train_model.py            # CNN training script (run once)
├── requirements.txt
├── README.md
├── templates/
│   ├── index.html            # Home / predictor page
│   └── about.html            # About / architecture page
├── static/
│   ├── css/style.css         # Full theme (dark-first glassmorphism)
│   ├── js/script.js          # Canvas, upload, charts, history, etc.
│   └── images/               # Auto-generated training charts
├── model/
│   ├── digit_model.h5        # Saved Keras model (after training)
│   └── stats.json            # Training metrics JSON
└── database/
    └── predictions.db        # SQLite prediction history
```

---

## 🚀 Quick Start

### 1. Clone / Download

```bash
git clone https://github.com/your-username/digit-recognizer.git
cd digit-recognizer
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the CNN (generates model + charts)

```bash
python train_model.py
```

This will:
- Download MNIST automatically via Keras
- Train the CNN for up to 15 epochs (EarlyStopping enabled)
- Save `model/digit_model.h5` and `model/stats.json`
- Generate `static/images/training_curves.png`, `confusion_matrix.png`, `sample_predictions.png`

Expected output: **~99.3% test accuracy** in ~5 minutes on a CPU.

### 5. Start the Flask Server

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Home page |
| GET  | `/about` | About page |
| POST | `/api/predict/draw` | Predict from base-64 canvas PNG |
| POST | `/api/predict/upload` | Predict from multipart image upload |
| POST | `/api/predict/realtime` | Lightweight real-time prediction (no DB) |
| GET  | `/api/history` | JSON list of recent predictions |
| GET  | `/api/stats` | Model training statistics |
| GET  | `/api/report/<id>` | Download PDF report for prediction ID |

### Example: cURL predict

```bash
# Draw endpoint — send base-64 image
curl -X POST http://localhost:5000/api/predict/draw \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,<base64string>"}'
```

Response:
```json
{
  "predicted": 7,
  "confidence": 98.6,
  "top3": [
    {"digit": 7, "probability": 98.6},
    {"digit": 1, "probability": 0.9},
    {"digit": 9, "probability": 0.5}
  ],
  "gradcam": "<base64 png>",
  "chart":   "<base64 png>",
  "thumbnail": "<base64 png>"
}
```

---

## 🧠 CNN Architecture

```
INPUT       28 × 28 × 1
────────────────────────────────────────
CONV 3×3    32 filters  ReLU + BatchNorm
CONV 3×3    32 filters  ReLU + BatchNorm
MaxPool 2×2 + Dropout(0.25)
────────────────────────────────────────
CONV 3×3    64 filters  ReLU + BatchNorm
CONV 3×3    64 filters  ReLU + BatchNorm
MaxPool 2×2 + Dropout(0.25)
────────────────────────────────────────
CONV 3×3   128 filters  ReLU + BatchNorm
MaxPool 2×2 + Dropout(0.25)
────────────────────────────────────────
Flatten → Dense(256) → Dense(128) → Dense(10, softmax)
```

---

## ☁️ Deployment

### Heroku

```bash
# Add Procfile
echo "web: gunicorn app:app" > Procfile

heroku create your-app-name
git push heroku main
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t digit-recognizer .
docker run -p 5000:5000 digit-recognizer
```

### Render / Railway

Set the start command to:
```
gunicorn app:app
```

---

## 🛠 Customisation

| What | Where |
|------|-------|
| Theme colours | `static/css/style.css` → `:root` variables |
| Contact info | `templates/index.html` → `#contact` section |
| Model architecture | `train_model.py` → `Sequential([...])` |
| Training epochs | `train_model.py` → `EPOCHS = 15` |
| History limit | `app.py` → `get_prediction_history(limit=20)` |

---

## 📄 License

MIT — free to use, modify, and distribute for personal and commercial projects.

---

## 🙏 Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) — Yann LeCun et al.
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Bootstrap 5](https://getbootstrap.com/)
- [Chart.js](https://www.chartjs.org/)
