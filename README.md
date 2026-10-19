# Verifica sperimentale della frequenza di Schroeder

Codice e misure per tesina di Acustica: misura della risposta all'impulso di una stanza reale non trattata e verifica della transizione fra **regime modale** e **regime statistico** secondo il criterio di Schroeder (1962).

## Struttura

```
src/
  rew_io.py                 lettura dei .mdat e degli export di REW
  integration.py            estremi di integrazione (Lundeby) e integrale di Schroeder
notebooks/
  calibrazione.ipynb        loopback scheda audio + .cal dell'ECM8000
  analisi.ipynb             IR, T20 per bande, risposta in frequenza media
data/
  room/                     IR misurate: Danger_Pos{1..5}.mdat (3 sweep ciascuno)
    psy_smooth/             stesse misure con smoothing psicoacustico di REW
  calibration/
    mdat/                   misure native di calibrazione (loopback, HS5, ECM8000)
    txt/                    export testuali delle stesse (freq dB phase)
    soundcard_minifuse4.cal compensazione loopback della scheda audio
    ecm8000_mic.cal         calibrazione microfono, output di calibrazione.ipynb
  reference/                datasheet HS5 digitalizzato + progetto WebPlotDigitizer
  results/                  tabelle prodotte da analisi.ipynb
```

## Uso

```bash
python3 -m venv .acoustics && source .acoustics/bin/activate
pip install -r requirements.txt
jupyter lab
```

I notebook si eseguono dalla cartella `notebooks/` e risolvono i percorsi
relativamente alla radice del repo. Le figure vengono salvate in `figures/`,
oppure nella cartella indicata dalla variabile d'ambiente `FIG_DIR`.

## Formato `.mdat`

I file di Room EQ Wizard sono serializzazioni Java non documentate e i parser
generici (`javaobj-py3`) falliscono sui blocchi `writeObject` custom. Il modulo
`src/rew_io.py` legge lo stream a mano: i campi primitivi di
`roomeqwizard.MeasData` seguono il class descriptor nell'ordine di
dichiarazione, e le IR sono gli array `float[]` di 131072 campioni (2.73 s @
48 kHz), riconoscibili per lunghezza fra gli altri array del file (risposta in
frequenza precalcolata, curve di calibrazione, icona di anteprima). REW allinea
internamente le IR sul picco, al campione 48000: per questo la media coerente
dei tre sweep è lecita senza risincronizzazione.

## Riferimenti

- M. R. Schroeder, *Frequency-Correlation Functions of Frequency Responses in
  Rooms*, J. Acoust. Soc. Am. **34**, 1819 (1962).
- H. Kuttruff, *Room Acoustics*.
- ISO 3382-1 (misura dei parametri acustici degli ambienti).

## Licenza

Codice e dati sono distribuiti con licenza MIT, vedi `LICENSE`.
