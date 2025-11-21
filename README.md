# VisionTransformer-Comparison

**Nama:** Elsa Elisa Yohana Sianturi  
**NIM:** 122140135  
**Kelas:** Deep Learning RA

## Deskripsi Proyek

Proyek ini merupakan eksplorasi dan perbandingan tiga arsitektur Vision Transformer (ViT) yang berbeda untuk klasifikasi gambar makanan:

- **CaiT** (Class-Attention in Image Transformers)
- **DeiT** (Data-efficient Image Transformers)
- **Swin Transformer** (Shifted Window Transformer)

Setiap model dilatih menggunakan transfer learning dengan fine-tuning pada dataset makanan kustom.

## Struktur File

```
.
├── CaiT.ipynb          # Notebook untuk model CaiT
├── Deit.ipynb          # Notebook untuk model DeiT
├── swin.ipynb          # Notebook untuk model Swin Transformer
└── README.md           # Dokumentasi proyek
```

## Spesifikasi Dataset

Dataset yang digunakan harus memiliki struktur sebagai berikut:

- **CSV File**: `label.csv` berisi kolom `filename` dan `label`
- **Image Directory**: Folder berisi gambar-gambar makanan
- **Split**: 80% training, 20% validation (stratified)

### Format Dataset

```
/content/drive/MyDrive/Tugas2/
├── label.csv
└── dataset/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Instalasi dan Setup

### 1. Install Dependencies

Setiap notebook sudah menyertakan instalasi library yang diperlukan. Jalankan cell pertama:

```python
!pip install timm seaborn tqdm
```

### 2. Mount Google Drive (Untuk Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Konfigurasi Path Dataset

Sesuaikan path dataset di setiap notebook:

```python
csv_path = "/content/drive/MyDrive/Tugas2/label.csv"
img_dir  = "/content/drive/MyDrive/Tugas2/dataset/"
```

## Cara Menjalankan Code

### Menggunakan Google Colab 

1. **Upload notebook ke Google Colab**

   - Buka [Google Colab](https://colab.research.google.com/)
   - Upload file `.ipynb` yang ingin dijalankan (CaiT.ipynb, Deit.ipynb, atau swin.ipynb)

2. **Jalankan secara berurutan**

   - Klik `Runtime` → `Run all` atau
   - Jalankan cell per cell dengan menekan `Shift + Enter`

3. **Urutan Eksekusi Cell:**
   - Install Library
   - Import Library
   - Load Dataset & Encode Label
   - Class Dataset dan Augmentasi
   - Konfigurasi Training
   - Load Model
   - Fungsi Training dan Early Stopping
   - Training Model
   - Plot Kurva
   - Evaluasi
   - Pengukuran Waktu Inferensi


## Detail Konfigurasi Model

### CaiT (cait_s36_384)

- **Input Size**: 384 × 384 pixels
- **Pretrained**: ImageNet-1K
- **Fine-tuning**: Head layer only
- **Batch Size**: 16
- **Learning Rate**: 1e-4

### DeiT (deit_base_patch16_224)

- **Input Size**: 224 × 224 pixels
- **Pretrained**: ImageNet-1K
- **Fine-tuning**: Head layer only
- **Batch Size**: 16
- **Learning Rate**: 1e-4

### Swin Transformer (swin_tiny_patch4_window7_224)

- **Input Size**: 224 × 224 pixels
- **Pretrained**: ImageNet-1K
- **Fine-tuning**: Head layer only
- **Batch Size**: 16
- **Learning Rate**: 1e-4

## Hyperparameter Training

```python
config = {
    "epochs": 12,
    "lr": 1e-4,
    "batch_size": 16,
    "patience": 3,  # Early stopping
    "device": "cuda" or "cpu"
}
```



## Output dan Hasil

Setiap notebook akan menghasilkan:

1. **Model Weights**: Disimpan di `/content/drive/MyDrive/Tugas2/result/`

   - `CaiT_best.pth`
   - `deit_best.pth`
   - `swin_best.pth`

2. **Visualisasi**:

   - Distribusi dataset
   - Training vs Validation Loss curve
   - Training vs Validation Accuracy curve
   - Confusion Matrix
   - Classification Report

3. **Metrics**:
   - Training Loss & Accuracy
   - Validation Loss & Accuracy
   - Inference Time (ms per image)
   - Throughput (images/second)
   - Per-class Precision, Recall, F1-score



## Referensi

- [timm Library](https://github.com/rwightman/pytorch-image-models)
- [CaiT Paper](https://arxiv.org/abs/2103.17239)
- [DeiT Paper](https://arxiv.org/abs/2012.12877)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)

## Lisensi

Proyek ini dibuat untuk keperluan akademik.

---


