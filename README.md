# CreditScore

Credit Risk Toolkit – Logistic Regression (Elastic Net), Univariate AUC, Stability & Permutation Importance

Zestaw narzędzi do budowy i oceny modeli PD w ryzyku kredytowym, oparty o scikit-learn. Repozytorium zawiera:

Moduł pobierający dane German Credit Data

Modul preprocessujący te dane

Jednocechowy screening (AUC/ROC) – szybka ocena „mocy” pojedynczych zmiennych,

Regresję logistyczną z Elastic Net + OHE – model bazowy PD,

Stability selection – odporna selekcja cech (częstotliwość wyboru przy L1/EN),

Permutation importance (z grupowaniem OHE) – wpływ cech na AUC modelu.