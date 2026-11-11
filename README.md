# 🚀 Machine Learning Practices & Projects

<p align="center">
  <a href="#english">English</a> | <a href="#türkçe">Türkçe</a>
</p>

---

## <a id="english"></a>🇬🇧 English

This repository contains machine learning practices, algorithms, and end-to-end mini projects implemented in Python, focused on building robust and data-driven solutions.

### 📂 Repository Structure

- **`Housing-Price-Prediction/`**: Predicting housing prices using Decision Tree and KNN regression models with hyperparameter tuning.
- **`car-price-prediction/`**: End-to-end regression pipeline comparing Multiple Linear and Polynomial Regression; features brand extraction on the `name` column to prevent overfitting and `MemoryError` optimization.
- **`titanic-survival-prediction/`**: End-to-end classification pipeline for survival prediction using the Titanic dataset; implements missing value imputation via training median, feature scaling, and iterative hyperparameter tuning across KNN, Decision Tree, Random Forest, SVM, and Naive Bayes models.
- **`customer-segmentation/`**: End-to-end unsupervised learning pipeline for customer segmentation using the Mall Customers dataset; explores and compares K-Means, Hierarchical (Agglomerative), and DBSCAN clustering algorithms based on annual income and spending score.
- **`q-learning-cliff-walking/`**: Tabular Q-learning reinforcement learning agent built with Gymnasium (`CliffWalking-v1`); trains an agent through trial-and-error to learn an optimal collision-free policy, achieving zero penalties and optimal step convergence during evaluation.
- **`pca-lda-tsne-comparison/`**: End-to-end dimensionality reduction pipeline comparing PCA, LDA, and T-SNE on the Car Details dataset; uses quantile-based target discretization (`pd.qcut`) to split selling prices into 3 equal price tiers for supervised projections, complete with comparative visual assets.

### 🛠 Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn 
- **Reinforcement Learning:** Gymnasium
- **Data Visualization:** Matplotlib
- **Version Control:** Git, GitHub, GitHub Desktop

---
*Connecting theory with practice to build smart, data-driven solutions.*

---

## <a id="türkçe"></a>🇹🇷 Türkçe

Bu repo, sağlam ve veri odaklı çözümler üretmeye odaklanmış, Python ile geliştirilmiş makine öğrenmesi pratikleri, algoritmaları ve uçtan uca mini projeler içermektedir.

### 📂 Depo Yapısı

- **`Housing-Price-Prediction/`**: Hiperparametre optimizasyonu ile Karar Ağacı (Decision Tree) ve KNN regresyon modellerini kullanarak konut fiyatı tahmini.
- **`car-price-prediction/`**: Çoklu Doğrusal ve Polinomal Regresyonu karşılaştıran uçtan uca regresyon boru hattı; aşırı öğrenmeyi (overfitting) önlemek ve `MemoryError` optimizasyonu sağlamak için `name` sütunundan marka çıkarımı (brand extraction) içerir.
- **`titanic-survival-prediction/`**: Titanic veri setini kullanarak hayatta kalma tahmini yapan uçtan uca sınıflandırma boru hattı; eğitim verisinin medyanı ile eksik veri doldurma, özellik ölçeklendirme ve KNN, Karar Ağacı, Random Forest, SVM ve Naive Bayes modelleri arasında yinelemeli hiperparametre optimizasyonu uygular.
- **`customer-segmentation/`**: Mall Customers veri setini kullanarak müşteri segmentasyonu yapan uçtan uca gözetimsiz öğrenme boru hattı; yıllık gelir ve harcama skoruna dayanarak K-Means, Hiyerarşik (Agglomerative) ve DBSCAN kümeleme algoritmalarını inceler ve karşılaştırır.
- **`q-learning-cliff-walking/`**: Gymnasium (`CliffWalking-v1`) ortamı kullanılarak geliştirilmiş tablo tabanlı Q-learning pekiştirmeli öğrenme ajanı; deneme-yanılma yoluyla optimal ve güvenli politikayı öğrenerek değerlendirme aşamasında sıfır ceza ve en kısa adım sayısıyla hedefe ulaşır.
- **`pca-lda-tsne-comparison/`**: Car Details veri seti üzerinde PCA, LDA ve T-SNE'yi karşılaştıran uçtan uca boyut indirgeme boru hattı; satış fiyatlarını denetimli projeksiyonlar için 3 eşit fiyat katmanına bölen nicem tabanlı hedef mühendisliği (`pd.qcut`) uygular ve grafiklerle görselleştirir.

### 🛠 Teknoloji Yığını ve Kütüphaneler
- **Dil:** Python
- **Veri Manipülasyonu ve Analizi:** Pandas, NumPy
- **Makine Öğrenmesi:** Scikit-Learn 
- **Pekiştirmeli Öğrenme:** Gymnasium
- **Veri Görselleştirme:** Matplotlib
- **Versiyon Kontrolü:** Git, GitHub, GitHub Desktop

---
*Akıllı ve veri odaklı çözümler üretmek için teoriyi pratikle buluşturuyoruz.*