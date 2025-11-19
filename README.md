# 🧬 Skin Cancer Classification Using Deep CNN

This project uses a **Deep Convolutional Neural Network (CNN)** to classify images of **two different types of skin cancer**. The model is trained using TensorFlow/Keras and includes several enhancements to improve performance and reduce overfitting.

---

## 📌 **Project Features**
- Deep CNN architecture for binary image classification  
- Advanced **data augmentation** for improving generalization  
- **Batch Normalization** for stable training  
- **Dropout layers** to reduce overfitting  
- **Callbacks**:
  - EarlyStopping  
  - ReduceLROnPlateau  
- Training/validation curves for accuracy and loss  
- Test evaluation with final accuracy score  
- Model saving using `.h5` format  

---

## 📂 **Project Structure**

```
📁 skin-cancer-classification/
│
├── 📁 train/              # Training images (two classes)
├── 📁 test/               # Test images (two classes)
├── project.py             # Main training script (CNN model)
├── model.h5               # Saved trained model
├── README.md              # Project documentation
└── requirements.txt       # List of dependencies
```

---

## 🚀 **How the Model Works**

1. Images are preprocessed using:
   - Rescaling  
   - Brightness adjustments  
   - Random zoom  
   - Horizontal flipping  

2. The CNN consists of:
   - 3 Convolution + MaxPooling blocks  
   - BatchNormalization  
   - Dropout layers  
   - Dense layers with ReLU  
   - Sigmoid output for binary classification  

3. Model is trained using:
   - Binary Crossentropy Loss  
   - Adam optimizer  

---

## 🧠 **Model Accuracy**
The model achieved:

- **Training Accuracy:** ~80%+  
- **Validation Accuracy:** (Varies based on data)  
- **Test Accuracy:** **82%+**  

---

## 💾 **Saving & Loading the Model**

### Save:
```python
model.save("skin_cancer_cnn_model.h5")
```

### Load:
```python
from tensorflow.keras.models import load_model
model = load_model("skin_cancer_cnn_model.h5")
```

---

## 📊 **Training Curves**
Accuracy and loss curves are plotted for both training and validation sets to visualize performance over epochs.

---

## 🛠 **Installation**

```bash
pip install tensorflow keras matplotlib pillow
```

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## ▶️ **Run the Project**
```bash
python project.py
```

---

## 🤝 **Contributions**
Pull requests are welcome!  
If you have suggestions for improvements, feel free to open an issue.

---

## 📜 **License**
This project is licensed under the MIT License.

---

## ⭐ **Support**
Don't forget to ⭐ the repo if you find it useful!

