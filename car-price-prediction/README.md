# Car Price Prediction with Multiple Linear & Polynomial Regression

[English](#english) | [Türkçe](#türkçe)

## 🇬🇧 English

### Project Overview
This project implements an end-to-end Machine Learning pipeline using **Python** and **Scikit-Learn** to predict used car selling prices (`selling_price`). It compares the performance of **Multiple Linear Regression** and **Polynomial Regression**.

* **Dataset:** [Car Details from Car Dekho Dataset on Kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/car-details-dataset)

### Key Engineering Decisions & Challenges Handled

1. **Handling High Cardinality and Preventing Overfitting (`name` Feature Engineering):**
   * The raw `name` column contained hundreds of distinct, highly specific car models. Directly applying One-Hot Encoding (`get_dummies`) would generate over 1,500 columns, causing the model to **memorize (overfit)** individual car names rather than learning general price trends.
   * **Solution:** We extracted only the primary brand (the first word of the name string) using string manipulation. This reduced the feature space to manageable, meaningful car brands (e.g., Maruti, Hyundai, Honda) instead of hundreds of unique variants.

2. **Solving the `MemoryError` (Combinatorial Explosion):**
   * Initially, we attempted a Polynomial Regression loop with higher degrees (`range(2, 5)`). However, combining the massive number of raw name dummies with polynomial feature transformations caused a combinatorial explosion, demanding over 7 GiB of RAM and crashing the system with a `MemoryError`.
   * **Solution:** By simplifying the `name` feature into brands and narrowing the polynomial degree loop to `range(2, 3)` (Degree 2), we successfully stabilized memory usage and ran the models smoothly.

3. **Multicollinearity Prevention:** 
   * Applied `drop_first=True` during One-Hot Encoding to avoid the dummy variable trap.

### Experimental Results

* **Missing Values Check:** `False` (No missing values found in the dataset)
* **Multiple Linear Regression:**
  * **RMSE:** `390,871.48`
  * **MSE:** `152,780,514,424.72`
* **Polynomial Regression (Degree 2):**
  * **RMSE:** `480,538.71`
  * **Best Degree:** `2`

> **Conclusion:** For this specific feature configuration, **Multiple Linear Regression** achieved a significantly lower error rate than Polynomial Regression, proving that a standard linear hyperplane generalized better to unseen test data.

---

## 🇹🇷 Türkçe

### Proje Özeti
Bu proje, ikinci el araba satış fiyatlarını (`selling_price`) tahmin etmek amacıyla **Python** ve **Scikit-Learn** kullanılarak geliştirilmiş uçtan uca bir Makine Öğrenmesi pipeline'ıdır. **Çoklu Lineer Regresyon** ve **Polinomal Regresyon** modellerini karşılaştırır.

* **Veri Seti:** [Kaggle Car Details from Car Dekho Veri Seti](https://www.kaggle.com/datasets/akshaydattatraykhare/car-details-dataset)

### Mühendislik Kararları ve Karşılaşılan Zorluklar

1. **Yüksek Kardinalite ve Ezberleme (Overfitting) Önleme (`name` Özellik Mühendisliği):**
   * Ham `name` sütununda yüzlerce farklı ve spesifik araba modeli bulunuyordu. Doğrudan One-Hot Encoding (`get_dummies`) yapmak 1.500'den fazla sütun üretilmesine ve modelin genel eğilimleri öğrenmek yerine araba isimlerini **ezberlemesine (overfitting)** yol açıyordu.
   * **Çözüm:** Metin manipülasyonu yapılarak ismin sadece ilk kelimesi (ana marka: Maruti, Hyundai, Honda vb.) alındı. Böylece model gereksiz detaylardan arındırıldı.

2. **`MemoryError` (Bellek Patlaması) Çözümü:**
   * Başlangıçta Polinom Regresyon döngüsü `range(2, 5)` olarak kurulmuştu. Ancak ham model isimlerinin yarattığı devasa sütun sayısı, polinom dönüşümüyle birleştiğinde kombinasyonel patlamaya yol açmış ve 7 GiB'dan fazla RAM isteyerek bilgisayarı **`MemoryError`** ile çökertmişti.
   * **Çözüm:** `name` sütununun markaya indirgenmesi ve polinom derece aralığının `range(2, 3)` (2. derece) olarak optimize edilmesiyle bellek sorunu tamamen çözüldü.

3. **Çoklu Doğrusallık (Multicollinearity) Önlemi:**
   * One-Hot Encoding işleminde `drop_first=True` kullanılarak kukla değişken tuzağından kaçınıldı.

### Deneysel Sonuçlar

* **Eksik Değer Kontrolü:** `False` (Veri setinde eksik veri bulunmamaktadır)
* **Çoklu Lineer Regresyon Sonucu:**
  * **RMSE:** `390.871,48`
  * **MSE:** `152.780.514.424,72`
* **Polinomal Regresyon Sonucu (2. Derece):**
  * **RMSE:** `480.538,71`
  * **En İyi Derece:** `2`

> **Sonuç:** Bu veri seti ve özellik dağılımında **Çoklu Lineer Regresyon**, Polinomal Regresyona kıyasla çok daha düşük bir hata payı (RMSE) vererek test verilerine daha başarılı bir şekilde uyum sağlamıştır.