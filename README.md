# All_NBA_Team


## Opis projektu

### Cel
Celem projektu jest stworzenie modelu predykcyjnego, który prognozuje, którzy zawodnicy NBA zostaną wybrani do zespołów **All-NBA** oraz **All-Rookie** na podstawie ich statystyk z bieżącego sezonu. Model opiera się na analizie danych z oficjalnej bazy NBA i wykorzystuje algorytmy klasyfikacyjne do przewidywania wyników.

### Dane
Projekt korzysta z danych dostępnych w pakiecie [nba_api](https://github.com/swar/nba_api), który pozwala na pobieranie aktualnych statystyk zawodników bezpośrednio ze źródeł NBA. Dane obejmują m.in.:
- Punkty na mecz (PTS)
- Zbiórki na mecz (REB)
- Asysty na mecz (AST)
- Przechwyty (STL)
- Bloki (BLK)
- Skuteczność rzutów z gry (FG%)
- Skuteczność rzutów za 3 punkty (3P%)
- Minuty na mecz (MPG)
- Wartość wskaźnika **Player Efficiency Rating (PER)**

Statystyki są normalizowane i przetwarzane przed wykorzystaniem w modelu.

### Metodologia
Model klasyfikacyjny został zbudowany przy użyciu dwóch algorytmów:
1. **Random Forest Classifier** – wykorzystywany do prognozowania wyborów do All-NBA.
2. **Decision Tree Classifier** – stosowany w klasyfikacji zawodników do All-Rookie.

Proces predykcji składa się z następujących etapów:
- Pobranie i wstępne przetworzenie danych,
- Normalizacja oraz inżynieria cech,
- Trenowanie modeli na historycznych danych z ostatnich sezonów,
- Prognozowanie na podstawie bieżącego sezonu.

Modele zostały wytrenowane na danych z lat 1998–2023, a ich skuteczność była oceniana na podstawie wskaźników **Accuracy**, **F1-score** oraz **ROC-AUC**.

### Wykorzystane technologie
- **Język**: Python 3.8+
- **Biblioteki**:
  - `pandas`, `numpy` – do obróbki danych
  - `scikit-learn` – do budowy modeli predykcyjnych
  - `nba_api` – do pobierania danych o zawodnikach NBA
  - `matplotlib`, `seaborn` – do wizualizacji wyników

### Struktura projektu
```
All_NBA_Team/                 
│── models/                  # Folder z zapisanymi modelami
│   │── clas_model.pkl    # Model do predykcji All-NBA
│   │── clas_r_model.pkl # Model do predykcji All-Rookie            
│── nba_predictions.py       # Główny plik skryptu do predykcji
│── requirements.txt         # Lista wymaganych bibliotek
│── README.md                # Dokumentacja projektu
```



## Autor
Antonina Frąckowiak.  
