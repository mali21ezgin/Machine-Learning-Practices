# Q-Learning Cliff Walking Agent

<p align="center">
  <a href="#english">English</a> | <a href="#türkçe">Türkçe</a>
</p>

---

## <a id="english"></a>🇬🇧 English

### Project Overview
This project implements a tabular **Q-Learning** reinforcement learning agent using **Python** and **Gymnasium** (`CliffWalking-v1`). The agent learns through trial and error to navigate a gridworld from a start state to a goal state while avoiding a hazardous cliff zone.

- **Environment:** Gymnasium `CliffWalking-v1` (Gridworld environment with a cliff penalty).
- **Algorithm:** Tabular Q-Learning (Off-policy TD control).

### Key Engineering Decisions & Parameters
- **Exploration vs. Exploitation:** Implemented an $\epsilon$-greedy policy ($\epsilon = 0.1$) to balance random exploration and optimal action exploitation.
- **Learning Parameters:** 
  - Learning Rate ($\alpha$): `0.1`
  - Discount Factor ($\gamma$): `0.6`
  - Training Episodes: `10,000`
- **Evaluation:** Tested over `100` episodes using the learned greedy policy ($\epsilon = 0$).

### Experimental Results
- **Average Steps per Episode:** `13.0` (Optimal path length)
- **Total Penalties (Cliff Falls):** `0`

> **Conclusion:** After 10,000 training episodes, the Q-learning agent successfully converged to the optimal policy, achieving zero penalties and navigating the grid in the minimum required steps.

### 🛠 Tech Stack & Libraries
- **Language:** Python
- **Reinforcement Learning:** Gymnasium
- **Data & Computation:** NumPy, TQDM

---
*Learning optimal behaviors through trial, error, and reward optimization.*

---

## <a id="türkçe"></a>🇹🇷 Türkçe

### Proje Özeti
Bu proje, **Python** ve **Gymnasium** (`CliffWalking-v1`) kullanarak tablo tabanlı bir **Q-Learning** pekiştirmeli öğrenme ajanı uygular. Ajan, tehlikeli uçurum bölgesinden kaçınarak başlangıç durumundan hedef duruma en güvenli şekilde ulaşmayı deneme-yanılma yoluyla öğrenir.

- **Ortam:** Gymnasium `CliffWalking-v1` (Uçurum cezası içeren ızgara dünya ortamı).
- **Algoritma:** Tablo Tabanlı Q-Learning (Off-policy TD kontrolü).

### Mühendislik Kararları ve Parametreler
- **Keşif ve Sömürü (Exploration vs. Exploitation):** Rastgele keşif ile optimal eylem sömürüsünü dengelemek için $\epsilon$-greedy politikası ($\epsilon = 0.1$) kullanıldı.
- **Öğrenme Parametreleri:**
  - Öğrenme Oranı ($\alpha$): `0.1`
  - İndirim Oranı ($\gamma$): `0.6`
  - Eğitim Bölümleri (Episodes): `10,000`
- **Değerlendirme:** Öğrenilen açgözlü (greedy) politika ($\epsilon = 0$) kullanılarak `100` bölüm boyunca test edildi.

### Deneysel Sonuçlar
- **Bölüm Başına Ortalama Adım:** `13.0` (Optimum yol uzunluğu)
- **Toplam Ceza (Uçuruma Düşme):** `0`

> **Sonuç:** 10.000 eğitim bölümünün ardından Q-learning ajanı optimal politikaya başarıyla yakınsamış, sıfır cezayla ve en kısa adımlarla ızgarayı tamamlamıştır.

### 🛠 Teknoloji Yığını ve Kütüphaneler
- **Dil:** Python
- **Pekiştirmeli Öğrenme:** Gymnasium
- **Hesaplama:** NumPy, TQDM

---
*Deneme, hata ve ödül optimizasyonu yoluyla optimal davranışları öğrenmek.*