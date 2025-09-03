
# 🦁 Camera Trap Platform

A **deep-learning based system** for assisting in the annotation of time-lapse photography datasets in animal ecology.
This platform combines **object detection**, **species classification**, and a **web-based interface** to streamline the workflow of ecologists working with large-scale camera trap data.

---

## ✨ Features

- 🚀 **MegaDetector V6** for animal detection and bounding box extraction.
- 🐾 **Species classification** using multiple deep learning backbones (ResNet50, ConvNeXt-Tiny, ViT-B/16, MobileNetV3-Large).
- 🖥️ **Interactive web interface** for uploading images, visualizing detections, and providing user feedback.
- 📊 **Progressive training & cross-validation pipelines** to ensure robust model evaluation.
- 🔄 **Feedback integration** for iterative model improvement.

---

## ⚙️ Requirements

- Python 3.10+
- Node.js & npm
- CUDA 12.5 (recommended if running on GPU)

---

## 🚀 Quickstart

### 1. Backend (Django API + Models)


```bash
cd wildife_classifier
python manage.py runserver
```


### 2. Frontend (React + Vite)

```bash
cd camera-trap-platform
npm run dev
```
