# 🏠 Housing Price Prediction Model

<p align="center">
  <a href="#english">English</a> | <a href="#türkçe">Türkçe</a>
</p>

---

## <a id="english"></a>🇬🇧 English

A machine learning project built with Python to predict housing prices using regression algorithms.

### 📊 Project Overview
This script performs data preprocessing, feature engineering, and rigorous hyperparameter optimization to compare the performance of two different machine learning models:
1. **Decision Tree Regressor** (Optimized for max depth, min samples split, and min samples leaf)
2. **K-Nearest Neighbors (KNN) Regressor** (Optimized for the best K value using scaled features)

### 🛠 Workflow & Pipeline
* **Data Loading & Inspection:** Reading the dataset using Pandas.
* **Preprocessing:** Handling categorical variables with One-Hot Encoding (`pd.get_dummies`) and splitting data.
* **Scaling:** Applying `StandardScaler` for distance-based algorithms like KNN.
* **Model Training & Tuning:** Automated loops to find the best hyperparameters based on Root Mean Squared Error (RMSE).
* **Evaluation:** Calculating RMSE and MSE metrics for both models.

### 📈 Model Evaluation & Results
After running hyperparameter optimization loops, the models achieved the following optimal performance metrics:

* **Decision Tree Regressor Optimization:**
  * Best Max Depth: `6` (RMSE = 1,629,731.81)
  * Best Min Samples Split: `3` (RMSE = 1,607,407.09)
  * Best Min Samples Leaf: `9` (RMSE = 1,564,664.18)
  * **Final Decision Tree Performance:** RMSE = `1,564,664.18` | MSE = `2,448,174,006,705.60`

* **KNN Regressor Optimization:**
  * Best K-Value: `5`
  * **Final KNN Performance:** RMSE = `1,398,656.73` | MSE = `1,956,240,657,808.66`

### 🚀 How to Run
1. Clone the repository or download the files.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Run the script:
   ```bash
   python app.py
   ```

### 📁 Dataset
* The dataset used in this project is sourced from Kaggle: [Housing Price Prediction by Harsh Kumar Datalab](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction?resource=download).

---

## <a id="türkçe"></a>🇹🇷 Türkçe

Python ile regresyon algoritmaları kullanarak konut fiyatı tahmini yapmak üzere geliştirilmiş bir makine öğrenmesi projesi.

### 📊 Proje Özeti
Bu betik; veri ön işleme, özellik mühendisliği ve sıkı bir hiperparametre optimizasyonu gerçekleştirerek iki farklı makine öğrenmesi modelinin performansını karşılaştırır:
1. **Karar Ağacı Regresörü (Decision Tree Regressor)** (Maksimum derinlik, minimum örnek bölünmesi ve yaprak örnek sayısı için optimize edilmiştir)
2. **En Yakın Komşu Regresörü (KNN Regressor)** (Ölçeklendirilmiş özellikler kullanılarak en iyi K değeri için optimize edilmiştir)

### 🛠 İş Akışı ve Boru Hattı (Workflow & Pipeline)
* **Veri Yükleme ve İnceleme:** Pandas kullanarak veri setinin okunması.
* **Ön İşleme:** Kategorik değişkenlerin One-Hot Encoding (`pd.get_dummies`) ile işlenmesi ve veri setinin bölünmesi.
* **Ölçeklendirme:** KNN gibi mesafe tabanlı algoritmalar için `StandardScaler` uygulanması.
* **Model Eğitimi ve Optimizasyon:** Kök Ortalama Kare Hata (RMSE) metriğine dayalı en iyi hiperparametreleri bulmak için otomatik döngüler.
* **Değerlendirme:** Her iki model için de RMSE ve MSE metriklerinin hesaplanması.

### 📈 Model Değerlendirme ve Sonuçlar
Hiperparametre optimizasyon döngüleri çalıştırıldıktan sonra modeller aşağıdaki optimal performans metriklerine ulaşmıştır:

* **Karar Ağacı Regresörü Optimizasyonu:**
  * En İyi Maksimum Derinlik (Depth): `6` (RMSE = 1,629,731.81)
  * En İyi Bölünme Örnek Sayısı (Split): `3` (RMSE = 1,607,407.09)
  * En İyi Yaprak Örnek Sayısı (Leaf): `9` (RMSE = 1,564,664.18)
  * **Final Karar Ağacı Performansı:** RMSE = `1,564,664.18` | MSE = `2,448,174,006,705.60`

* **KNN Regresör Optimizasyonu:**
  * En İyi K Değeri: `5`
  * **Final KNN Performansı:** RMSE = `1,398,656.73` | MSE = `1,956,240,657,808.66`

### 🚀 Nasıl Çalıştırılır?
1. Repoyu klonlayın veya dosyaları indirin.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Betiği çalıştırın:
   ```bash
   python app.py
   ```

### 📁 Veri Seti
* Bu projede kullanılan veri seti Kaggle'dan alınmıştır: [Housing Price Prediction by Harsh Kumar Datalab](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction?resource=download).