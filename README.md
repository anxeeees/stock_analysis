# Semestrální práce - Analýza trhu s akciemi pomocí Pythonu
# Ester Stankovská pro předmět 4EK317, LS - 2024

## Úvod

Tato semestrální práce se zaměřuje na analýzu trhu s akciemi za pomocí jazyka Python. Cílem práce je poskytnout ucelený přehled o různých aspektech investování do akcií, včetně technických analýz, predikce cen, analýzy dividend a dalších relevantních ukazatelů.

## Obsah

1. **Instalace a Spuštění**
2. **Analýza Dat**
    - Načítání a příprava dat
    - Vizualizace dat pomocí grafů a tabulek
3. **Predikce Cen Akcií**
    - Využití LSTM modelu pro predikci cen
    - Vizualizace predikcí a porovnání s reálnými daty
4. **Technické Indikátory**
    - Klouzavý průměr (MA)
    - Relativní síla indexu (RSI)
    - Moving Average Convergence Divergence (MACD)
    - Bollingerovy pásky
    - Stochastický oscilátor
5. **Analýza Dividend**
    - Výpočet dividendního výnosu a poměru vyplacených dividend
6. **Monte Carlo Simulace**
    - Simulace budoucích cen akcií pomocí Monte Carlo metody
7. **Aktuality a Novinky**
    - Získání a zobrazení aktuálních zpráv a novinek spojených s vybranými akciemi
8. **Závěr**

## Instalace a Spuštění

Pro spuštění této aplikace je třeba mít nainstalovaný Python a následující knihovny:
- numpy
- pandas
- matplotlib
- streamlit
- keras
- scikit-learn
- requests
- yfinance
- seaborn
- nltk

Aplikaci lze spustit pomocí následujícího příkazu:

```bash
streamlit run web_app.py
