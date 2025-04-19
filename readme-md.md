# Klasifikasi Gambar Sikat Gigi dengan Deep Learning

Proyek ini adalah implementasi model klasifikasi gambar yang dapat mengidentifikasi jenis sikat gigi dari sebuah gambar. Model ini dibangun menggunakan Convolutional Neural Network (CNN) dengan TensorFlow dan Keras.

## Dataset

Dataset terdiri dari gambar sikat gigi yang diorganisir dalam struktur folder berikut:

```
Dataset_sikat_gigi/
├── train/
│   ├── [kelas_1]/
│   ├── [kelas_2]/
│   └── ...
└── test/
    ├── [kelas_1]/
    ├── [kelas_2]/
    └── ...
```

Link dataset: [Google Drive](https://drive.google.com/drive/folders/1lMrivQ7F7kDzjc0einQeckRRQ2JuW6L0?usp=drive_link)

## Fitur Utama

- Klasifikasi multi-kelas untuk gambar sikat gigi
- Data augmentation untuk meningkatkan performa model
- Implementasi arsitektur CNN dengan multiple convolutional blocks
- Visualisasi confusion matrix dan metrik evaluasi
- Hyperparameter tuning untuk optimalisasi model
- Fine-tuning model untuk mencapai akurasi minimal 80%

## Arsitektur Model

Model CNN yang digunakan terdiri dari:
- 3 blok konvolusi dengan batch normalization dan dropout
- Fully connected layers dengan dropout untuk mencegah overfitting
- Output layer dengan aktivasi softmax untuk klasifikasi multi-kelas

## Hyperparameter

Parameter utama yang digunakan:
- Image size: 224 x 224 pixels
- Batch size: 16
- Learning rate: 0.001 (dengan ReduceLROnPlateau)
- Dropout rate: 0.5
- Epochs: 30 (dengan Early Stopping)

## Hasil

Hasil evaluasi model:
- Akurasi: >80% pada test set
- Confusion matrix menunjukkan performa model untuk masing-masing kelas
- Classification report memberikan detail precision, recall, dan f1-score

## Cara Menggunakan Model

### Persyaratan

Instal semua dependensi yang dibutuhkan:

```bash
pip install -r requirements.txt
```

### Melatih Model

Untuk melatih model dari awal, jalankan notebook `sikat_gigi_classification.ipynb` di Google Colab.

### Prediksi Gambar Baru

```python
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model('sikat_gigi_model.h5')

# Prediksi gambar
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    
    class_names = ['kelas_1', 'kelas_2', ...]  # Ganti dengan nama kelas yang sesuai
    predicted_class_name = class_names[predicted_class]
    
    return predicted_class_name, confidence

# Contoh penggunaan
img_path = 'test_image.jpg'
predicted_class, confidence = predict_image(img_path)
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence*100:.2f}%")
```

## Struktur Repository

```
.
├── sikat_gigi_classification.ipynb  # Notebook Google Colab
├── requirements.txt                 # Dependensi yang dibutuhkan
├── README.md                        # Dokumentasi proyek
└── model/
    └── sikat_gigi_model.h5          # Model terlatih (jika sudah di-push)
```

## Pengembangan Lebih Lanjut

Beberapa ide untuk pengembangan lebih lanjut:
1. Implementasi transfer learning dengan model pre-trained (ResNet, EfficientNet)
2. Penambahan data untuk kelas yang memiliki sampel sedikit
3. Implementasi teknik augmentasi data yang lebih kompleks
4. Deployment model sebagai web API
