# 🧠 KANVisionLSTM_FFT: Deepfake and Content Classification using Xception + LSTM + FFT

This project presents a deep learning pipeline to **detect real vs fake images** and **classify them** into one of three categories: `human_faces`, `animals`, or `vehicles`. The architecture combines CNN or ViT backbones (like Xception), an LSTM layer for sequential reasoning, and an FFT channel for frequency-domain feature enhancement.

---

## 🔍 Overview

- **Dataset**: `ArtiFact_240K` — contains images divided into `real` and `fake` for three classes.
- **Model**: `KANVisionLSTM_FFT`
  - Combines **RGB + FFT** inputs via 1x1 convolution.
  - Supports CNN and ViT backbones (e.g., Xception).
  - Applies **LSTM** and **attention mechanism** over features.
  - Uses two heads for:
    - Real/Fake detection (binary classification)
    - Object class prediction (multi-class classification)
- **Outputs**:
  - `Real/Fake` prediction (0 = fake, 1 = real)
  - `Class` prediction (`human_faces`, `animals`, `vehicles`)

---

## 🧱 Folder Structure

```
project/
│
├── ArtiFact_240K/
│   ├── train/
│   ├── validation/
│   └── test/
├── model_script.ipynb
├── test.csv
└── Model_Weights_Training_Log.log
```

---

## ⚙️ Dependencies

Install all required dependencies:

```bash
pip install torch torchvision timm scikit-learn matplotlib tqdm pandas pillow
```

---

## 🚀 How to Run

### 1. Prepare Dataset

Ensure your dataset is structured like this:

```
ArtiFact_240K/
├── train/
│   ├── real/
│   │   ├── human_faces/
│   │   ├── animals/
│   │   └── vehicles/
│   └── fake/
│       └── same as above...
├── validation/
└── test/
```

### 2. Train the Model

Open the file `model_script.ipynb` in jupyter notebook and run all the cells.

This will:
- Load and preprocess the dataset
- Train the model for 3 epochs
- Log training details
- Evaluate on the validation set

---

## 📂 Output Files

- **`Model_Weights_Training_Log.log`**  
  Contains saved model weights and training logs.

- **`test.csv`**  
  File containing predictions in the required format:  

  | image       | label | class        |
  |-------------|--------|--------------|
  | img1.jpg    | 1      | human_faces  |
  | img2.jpg    | 0      | animals      |
  | ...         | ...    | ...          |

---

## 📈 Metrics

After training, the script displays:
- ✅ Real/Fake Accuracy
- 🏷️ Class Accuracy

These evaluate binary classification and multi-class classification performance respectively.

