"""Estremi di integrazione e integrale di Schroeder per una banda d'ottava.

Modulo parallelo a `rew_io`: quello legge i file, questo prepara il
decadimento energetico su cui poi si misura il tempo di riverberazione.
Il calcolo di T20/T30 resta fuori, nel notebook.

Uso tipico::

    from integration import BandDecay

    banda = BandDecay.from_ir(ir, fc=500, fs=48000)
    banda.find_limits()             # costruzione di Lundeby
    curva = banda.schroeder()       # curva di decadimento in dB

Il punto delicato è `find_limits`. L'estremo superiore di integrazione non
si puo' fissare a priori: integrare fino in fondo somma anche la coda di
rumore, che essendo stazionaria contribuisce sempre di piu' andando
indietro nel tempo, appiattisce la curva su un plateau e allunga il tempo
di riverberazione. Lundeby et al. (1995) trovano il punto in cui il
decadimento incontra il rumore iterando fra due stime che dipendono l'una
dall'altra: la pendenza del decadimento e il livello del rumore.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass
class BandDecay:
    """Energia di una risposta all'impulso filtrata in una banda d'ottava.

    Attributi impostati alla creazione:
        fc      centro della banda [Hz]
        fs      frequenza di campionamento [Hz]
        energy  energia istantanea h^2, a partire dal picco

    Attributi riempiti da `find_limits()`:
        t_cross  estremo superiore di integrazione [s]
        noise    livello del rumore di fondo [lineare, non dB]
        slope    pendenza del decadimento tardo [dB/s]
        n_iter   iterazioni usate
        ok       True se l'algoritmo e' arrivato a convergenza
    """

    fc: float
    fs: float
    energy: np.ndarray

    t_cross: float = np.nan
    noise: float = np.nan
    slope: float = np.nan
    n_iter: int = 0
    ok: bool = False

    # ------------------------------------------------------------------
    # costruzione
    # ------------------------------------------------------------------
    @classmethod
    def from_ir(cls, ir, fc, fs, order=3):
        """Filtra la IR nella banda d'ottava centrata su fc.

        Il filtro e' applicato avanti e indietro (`sosfiltfilt`) per avere
        fase nulla: un filtro causale ritarderebbe le basse frequenze e
        falserebbe l'inizio del decadimento. Il segnale viene poi tagliato
        al picco, cioe' all'arrivo del suono diretto.
        """
        sos = butter(order, [fc/np.sqrt(2), fc*np.sqrt(2)],
                     'bandpass', fs=fs, output='sos')
        x = sosfiltfilt(sos, ir)
        x = x[int(np.abs(x).argmax()):]
        return cls(fc=fc, fs=fs, energy=x**2)

    @property
    def t(self):
        """Asse dei tempi dell'energia, con origine sul suono diretto [s]."""
        return np.arange(len(self.energy)) / self.fs

    # ------------------------------------------------------------------
    # utilita' interne
    # ------------------------------------------------------------------
    def _block_mean(self, n_blocco):
        """Media l'energia su blocchi contigui: da campioni a intervalli.

        Serve a lisciare la curva prima delle regressioni: l'energia
        istantanea di una singola IR fluttua troppo per fittarci una retta.
        Restituisce (energia media per blocco, asse dei tempi dei blocchi).
        """
        n = len(self.energy) // n_blocco
        eb = self.energy[:n*n_blocco].reshape(n, n_blocco).mean(axis=1)
        t = (np.arange(n) + 0.5) * n_blocco / self.fs
        return eb, t

    @staticmethod
    def _db(x, riferimento):
        """Livello in dB rispetto a un riferimento, senza log(0)."""
        return 10*np.log10(np.maximum(x, 1e-300) / riferimento)

    # ------------------------------------------------------------------
    # estremi di integrazione
    # ------------------------------------------------------------------
    def find_limits(self, blocco_ms=20, max_iter=8):
        """Costruzione di Lundeby: trova dove il decadimento incontra il rumore.

        I numeri fra parentesi nei commenti sono le tappe dell'articolo
        originale. Modifica l'oggetto e lo restituisce, cosi' si puo'
        concatenare con `.schroeder()`.
        """
        e, fs = self.energy, self.fs

        # --- stima iniziale, con blocchi di lunghezza fissa ---
        noise = e[int(0.9*len(e)):].mean()                  # (2) ultimo 10%
        n_blocco = int(blocco_ms * 1e-3 * fs)               # (1)

        eb, t = self._block_mean(n_blocco)
        y = self._db(eb, eb.max())
        y_noise = self._db(noise, eb.max())

        fine = np.argmax(y < y_noise + 10)                  # (3) primo punto a +10 dB
        if fine < 2:                                        # rumore gia' vicino al picco
            return self
        slope, q = np.polyfit(t[:fine], y[:fine], 1)
        if slope >= 0:                                      # banda senza decadimento
            return self
        t_cross = (y_noise - q) / slope                     # (4)

        # --- raffinamento iterativo ---
        for i in range(1, max_iter + 1):
            # (5) blocchi ridimensionati sulla pendenza: 5 intervalli ogni 10 dB
            n_blocco = max(1, int((10/abs(slope)) / 5 * fs))
            eb, t = self._block_mean(n_blocco)               # (6)

            # (7) il rumore si stima da 5 dB di decadimento dopo il crosspoint
            i_noise = int((t_cross + 5/abs(slope)) * fs)
            noise = e[min(i_noise, int(0.9*len(e))):].mean()

            y = self._db(eb, eb.max())
            y_noise = self._db(noise, eb.max())

            # (8) regressione tarda su 20 dB, da 5 dB sopra il rumore.
            # Il vincolo t < t_cross e' essenziale: senza, la regressione
            # include blocchi gia' appiattiti sul rumore, la pendenza
            # risulta piu' dolce, il crosspoint si sposta avanti e il
            # procedimento diverge.
            sel = (y > y_noise + 5) & (y < y_noise + 25) & (t < t_cross)
            if sel.sum() < 3:
                break
            slope, q = np.polyfit(t[sel], y[sel], 1)
            if slope >= 0:
                break

            t_cross_new = (y_noise - q) / slope              # (9)
            converge = abs(t_cross_new - t_cross) < n_blocco/fs
            t_cross = t_cross_new
            self.n_iter = i
            if converge:
                self.ok = True
                break

        self.t_cross = t_cross
        self.noise = noise
        self.slope = slope
        # l'estremo deve cadere dentro il segnale disponibile
        self.ok = self.ok and 0 < t_cross < len(e)/fs
        return self

    # ------------------------------------------------------------------
    # integrale
    # ------------------------------------------------------------------
    def truncated_energy(self):
        """Energia troncata al crosspoint e depurata del rumore di fondo."""
        if np.isnan(self.t_cross):
            raise RuntimeError('chiama prima find_limits()')
        i_cross = int(self.t_cross * self.fs)
        return np.clip(self.energy[:i_cross] - self.noise, 0, None)

    def schroeder(self, compensate=True):
        """Curva di decadimento di Schroeder, in dB e normalizzata a 0 dB.

        Integrale all'indietro dell'energia troncata:
        `cumsum` del vettore rovesciato, poi rirovesciato.

        Con `compensate=True` si aggiunge il termine di Chu, cioe' l'energia
        della coda esponenziale che il troncamento ha buttato via:
        C = e(t_cross) * tau, con tau = 10 / (|slope| * ln10). Per T20 e'
        trascurabile; conta per EDT e per i parametri di chiarezza, che
        dipendono dall'energia totale.
        """
        e = self.truncated_energy()

        C = 0.0
        if compensate and self.slope < 0 and len(e) > 20:
            tau = 10.0 / (abs(self.slope) * np.log(10))
            coda = e[-int(0.05*len(e)):].mean()
            C = coda * tau * self.fs        # *fs perche' cumsum somma, non integra

        energia = np.cumsum(e[::-1])[::-1] + C
        return 10*np.log10(energia / energia[0] + 1e-30)

    def schroeder_time(self, curva):
        """Asse dei tempi corrispondente a una curva restituita da schroeder()."""
        return np.arange(len(curva)) / self.fs
