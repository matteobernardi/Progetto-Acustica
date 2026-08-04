# Verifica sperimentale della frequenza di Schroeder

Codice e misure per una tesina di Acustica (LM Fisica): misura della risposta
all'impulso di una stanza reale non trattata e verifica della transizione fra
**regime modale** e **regime statistico** prevista da Schroeder (1962),

$$ f_S = 2000\sqrt{T_{60}/V} \quad [\mathrm{Hz}] $$

condizione equivalente a richiedere che almeno tre modi normali cadano, in
media, nella banda a mezza potenza di una singola risonanza.

**Idea fisica.** La densità modale di una cavità cresce come $f^2$ (stessa
legge della densità di stati di una buca 3D), mentre la larghezza di riga di
ogni risonanza resta fissata dallo smorzamento, $\Delta f_{3\mathrm{dB}} =
2.2/T_{60}$. Sopra una certa soglia le code delle risonanze si sovrappongono e
la funzione di trasferimento cessa di essere risolvibile modo per modo: diventa
un processo gaussiano complesso, con conseguenze statistiche verificabili senza
parametri liberi (modulo di Rayleigh, spaziatura media dei massimi $4/T_{60}$,
autocorrelazione lorentziana con $\tau = T_{60}/13.8$).

Setup: sala 5.70 × 4.80 × 2.48 m ($V = 67.85$ m³), sorgente Yamaha HS5,
microfono Behringer ECM8000 calibrato, interfaccia Arturia MiniFuse 4, sweep
ESS 20–22000 Hz a 48 kHz, 5 posizioni microfoniche × 3 ripetizioni.

## Struttura

```
src/rew_io.py     lettura dei .mdat di REW (serializzazione Java) e degli
                  export testuali; percorsi del repo
notebooks/
  01_calibrazione_microfono.ipynb   loopback scheda audio + .cal dell'ECM8000
                                    per confronto con datasheet HS5
  02_analisi_schroeder.ipynb        IR, media coerente, T20 per bande,
                                    f_S auto-consistente, statistica della FR
data/
  room/                    IR in sala: Danger_Pos{1..5}.mdat (3 sweep ciascuno)
    psy_smooth/            stesse misure con smoothing psicoacustico di REW
  calibration/
    mdat/                  misure native di calibrazione (loopback, HS5, ECM8000)
    txt/                   export testuali delle stesse (freq dB phase)
    soundcard_minifuse4.cal  compensazione loopback della scheda audio
    ecm8000_mic.cal          calibrazione microfono, output del notebook 01
  reference/               datasheet HS5 digitalizzato + progetto WebPlotDigitizer
figures/                   output dei notebook (non versionato)
```

## Uso

```bash
python3 -m venv .acoustics && source .acoustics/bin/activate
pip install -r requirements.txt
jupyter lab
```

I notebook si eseguono dalla cartella `notebooks/` e risolvono i percorsi
relativamente alla radice del repo, quindi funzionano su qualunque macchina
senza modifiche.

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

## Documenti

La parte scritta (tesina, teoria, protocolli di misura, log di sessione) non
sta in questo repo: vive su Google Drive in
`Università/Master/Illuminotecnica:Acustica/Progetto/`.

## Riferimenti

- M. R. Schroeder, *Frequency-Correlation Functions of Frequency Responses in
  Rooms*, J. Acoust. Soc. Am. **34**, 1819 (1962).
- H. Kuttruff, *Room Acoustics*.
- ISO 3382-1 (misura dei parametri acustici degli ambienti).
