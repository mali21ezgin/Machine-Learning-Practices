# Turkish Spam Classification (NLP & Supervised Learning Pipeline)

<p align="center">
  <a href="#english">English</a> | <a href="#türkçe">Türkçe</a>
</p>

---

## <a id="english"></a>🇬🇧 English

### Project Overview
This project implements an end-to-end **Natural Language Processing (NLP)** and text classification pipeline using **Python**, **Scikit-Learn**, and **Matplotlib**. It processes the [Turkish Spam Dataset](https://www.kaggle.com/datasets/cuneytdemir/turkish-spam-dataset) via `kagglehub`, cleans the text data, transforms raw messages into numerical feature vectors using **TF-IDF**, and compares multiple machine learning algorithms with hyperparameter tuning and k-fold cross-validation.

- **Dataset:** Dynamically fetched via `kagglehub` (`cuneytdemir/turkish-spam-dataset`).
- **Text Vectorization:** Converted raw text messages into numerical feature representations using `TfidfVectorizer`.

### Key Engineering Decisions & Approach
1. **Data Preprocessing & Cleaning:**
   * Handled potential formatting issues and bad lines using custom `pandas_kwargs` (`on_bad_lines="skip"`, `encoding="utf-8"`).
   * Checked and dropped missing values to ensure data integrity.
2. **Model Training, Optimization & Evaluation:**
   * **Decision Tree Classifier:** Optimized via `GridSearchCV` and 5-fold cross-validation (`KFold`) over splitting criteria, max depth, and sample split parameters.
     
     <p align="center">
       <img src="assets/decision_tree.png" alt="Decision Tree Confusion Matrix" width="500"/>
     </p>

   * **Random Forest Classifier:** Trained with hyperparameter tuning on estimator counts, depth, and leaf constraints to boost robustness.
     
     <p align="center">
       <img src="assets/rf.png" alt="Random Forest Confusion Matrix" width="500"/>
     </p>

   * **Multinomial Naive Bayes:** Implemented as a strong probabilistic baseline for text classification.
     
     <p align="center">
       <img src="assets/navie_bayes.png" alt="Naive Bayes Confusion Matrix" width="500"/>
     </p>

   * **Logistic Regression:** Optimized with the `saga` solver across regularization penalties (`l1`, `l2`) and inverse regularization strengths (`C`).
     
     <p align="center">
       <img src="assets/logistic_reg.png" alt="Logistic Regression Confusion Matrix" width="500"/>
     </p>

3. **Performance Metrics:** Evaluated models using detailed classification reports (`precision`, `recall`, `f1-score`) alongside custom-styled `ConfusionMatrixDisplay` plots.

### 🛠 Tech Stack & Libraries
- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **NLP & Text Vectorization:** Scikit-Learn (`TfidfVectorizer`)
- **Machine Learning & Evaluation:** Scikit-Learn (`DecisionTreeClassifier`, `RandomForestClassifier`, `MultinomialNB`, `LogisticRegression`, `GridSearchCV`, `KFold`, `classification_report`)
- **Data Visualization:** Matplotlib
- **Dataset Source:** Kaggle Hub (`kagglehub`)

---
*Transforming text data into actionable insights through robust NLP pipelines.*

---

## <a id="türkçe"></a>🇹🇷 Türkçe

### Proje Özeti
Bu proje, **Python**, **Scikit-Learn** ve **Matplotlib** kullanarak uçtan uca bir **Doğal Dil İşleme (NLP)** ve metin sınıflandırma boru hattı uygular. `kagglehub` üzerinden [Turkish Spam Dataset](https://www.kaggle.com/datasets/cuneytdemir/turkish-spam-dataset) veri setini dinamik olarak yükler; metinleri temizler, **TF-IDF** ile sayısal vektörlere dönüştürür ve hiperparametre optimizasyonu ile çapraz doğrulama eşliğinde birden fazla makine öğrenmesi algoritmasını karşılaştırır.

- **Veri Seti:** `kagglehub` (`cuneytdemir/turkish-spam-dataset`) kullanılarak dinamik olarak indirilmiştir.
- **Metin Vektörleştirme:** Ham metin mesajları `TfidfVectorizer` kullanılarak sayısal özellik matrislerine dönüştürülmüştür.

### Mühendislik Kararları ve Yaklaşım
1. **Veri Ön İşleme ve Temizleme:**
   * `pandas_kwargs` ayarları (`on_bad_lines="skip"`, `encoding="utf-8"`) kullanılarak olası biçimlendirme hataları ve bozuk satırlar güvenli bir şekilde ele alınmıştır.
   * Eksik veriler kontrol edilip temizlenerek veri bütünlüğü sağlanmıştır.
2. **Model Eğitimi, Optimizasyon ve Değerlendirme:**
   * **Karar Ağacı (Decision Tree):** Bölütleme kriteri, maksimum derinlik ve örneklem bölme parametreleri üzerinde `GridSearchCV` ve 5 katlı çapraz doğrulama (`KFold`) ile optimize edilmiştir.
     
     <p align="center">
       <img src="assets/decision_tree.png" alt="Karar Ağacı Karmaşıklık Matrisi" width="500"/>
     </p>

   * **Rastgele Orman (Random Forest):** Güçlü genelleme için ağaç sayısı, derinlik ve yaprak kısıtları optimize edilerek eğitilmiştir.
     
     <p align="center">
       <img src="assets/rf.png" alt="Random Forest Karmaşıklık Matrisi" width="500"/>
     </p>

   * **Multinomial Naive Bayes:** Metin sınıflandırma için güçlü bir olasılıksal temel model olarak kurulmuştur.
     
     <p align="center">
       <img src="assets/navie_bayes.png" alt="Naive Bayes Karmaşıklık Matrisi" width="500"/>
     </p>

   * **Lojistik Regresyon (Logistic Regression):** `saga` çözücüsü eşliğinde düzenlileştirme cezaları (`l1`, `l2`) ve ters düzenlileştirme gücü (`C`) parametreleriyle optimize edilmiştir.
     
     <p align="center">
       <img src="assets/logistic_reg.png" alt="Logistic Regression Karmaşıklık Matrisi" width="500"/>
     </p>

3. **Performans Metrikleri:** Modeller, detaylı sınıflandırma raporları (`precision`, `recall`, `f1-score`) ve özelleştirilmiş `ConfusionMatrixDisplay` grafikleriyle değerlendirilmiştir.

### 🛠 Teknoloji Yığını ve Kütüphaneler
- **Dil:** Python
- **Veri İşleme:** Pandas, NumPy
- **NLP ve Vektörleştirme:** Scikit-Learn (`TfidfVectorizer`)
- **Makine Öğrenmesi ve Değerlendirme:** Scikit-Learn (`DecisionTreeClassifier`, `RandomForestClassifier`, `MultinomialNB`, `LogisticRegression`, `GridSearchCV`, `KFold`, `classification_report`)
- **Veri Görselleştirme:** Matplotlib
- **Veri Kaynağı:** Kaggle Hub (`kagglehub`)

---
*Sağlam NLP boru hatlarıyla metin verilerini eyleme dönüştürülebilir içgörülere çevirmek.*