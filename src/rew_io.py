"""Lettura dei file prodotti da Room EQ Wizard (REW).

Due formati, due strategie:

* **`.mdat`** — è una *serializzazione Java* (header ``REW Measurement Data
  File V2``), non un formato documentato. I parser generici (`javaobj-py3`)
  falliscono con ``Invalid reference handle`` perché REW usa blocchi
  ``writeObject`` custom. Qui si legge lo stream a mano sfruttando due fatti
  del protocollo di serializzazione Java: (a) i valori dei campi primitivi
  seguono immediatamente il *class descriptor* nell'ordine di dichiarazione,
  (b) gli array primitivi sono introdotti da token riconoscibili
  (``ur\\x00\\x02[F`` per il descrittore completo, ``uq`` + handle per i
  riferimenti successivi allo stesso descrittore).

* **`.txt` / `.cal`** — export testuali a tre colonne ``freq dB phase`` con
  header variabile di righe che iniziano per ``*``.

Uso tipico::

    import sys; sys.path.append('../src')
    from rew_io import ROOM_DIR, read_mdat, load_rew_export

    m = read_mdat(ROOM_DIR / 'Danger_Pos1.mdat')
    m.fields['sampleRate'], len(m.irs)
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Percorsi del repo — risolti rispetto alla posizione di questo file, così i
# notebook funzionano da qualunque cwd e su qualunque macchina che cloni.
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data'
ROOM_DIR = DATA_DIR / 'room'                 # IR in sala (.mdat)
SMOOTH_DIR = ROOM_DIR / 'psy_smooth'         # versioni smoothed da REW
CAL_DIR = DATA_DIR / 'calibration'           # .cal della catena di misura
CAL_MDAT_DIR = CAL_DIR / 'mdat'              # misure di calibrazione native
CAL_TXT_DIR = CAL_DIR / 'txt'                # export testuali delle stesse
REF_DIR = DATA_DIR / 'reference'             # datasheet e curve di confronto
FIG_DIR = Path('/Users/matteob/Library/CloudStorage/GoogleDrive-matteobernardi001@gmail.com/'
               'Il mio Drive/Files/Università/Master/Illuminotecnica:Acustica/Progetto/'
               'tesina/figure')

# Token del protocollo Java Object Serialization usati sotto.
_TC_ENDBLOCKDATA_NULL = b'\x78\x70'
_MEASDATA_ANCHOR = b'sr\x00\x15roomeqwizard.MeasData'
_ARRAY_FULL_DESC = b'ur\x00\x02[F'           # newArray + classDesc float[]
_ARRAY_REF_DESC = b'\x75\x71\x00\x7e'        # newArray + reference a classDesc

_PRIM_SIZES = {'B': 1, 'C': 2, 'D': 8, 'F': 4, 'I': 4, 'J': 8, 'S': 2, 'Z': 1}
_PRIM_FMTS = {'B': '>b', 'C': '>H', 'D': '>d', 'F': '>f',
              'I': '>i', 'J': '>q', 'S': '>h', 'Z': '>?'}

# Lunghezza standard delle IR salvate da REW: 131072 campioni = 2.73 s @ 48 kHz.
IR_LEN_DEFAULT = 131072


def _read_utf(data: bytes, p: int) -> tuple[str, int]:
    """Legge una stringa Java modified-UTF8 (uint16 di lunghezza + bytes)."""
    n = struct.unpack('>H', data[p:p + 2])[0]
    return data[p + 2:p + 2 + n].decode('utf-8', 'replace'), p + 2 + n


def parse_measdata_fields(data: bytes) -> dict[str, object]:
    """Campi primitivi della **prima** istanza di ``roomeqwizard.MeasData``.

    Restituisce un dict con, fra gli altri, ``sampleRate``, ``startFreq``,
    ``endFreq``, ``numSweeps``, ``dataLength`` e le dimensioni della stanza
    impostate in REW (``roomLength``/``roomWidth``/``roomHeight``, che valgono
    i default se non sono state configurate nel programma).
    """
    anchor = data.find(_MEASDATA_ANCHOR)
    if anchor < 0:
        raise ValueError('class descriptor MeasData non trovato')

    p = anchor + 2
    _, p = _read_utf(data, p)          # nome classe
    p += 8 + 1                         # serialVersionUID (8 B) + flags (1 B)
    nfields = struct.unpack('>H', data[p:p + 2])[0]
    p += 2

    fields: list[tuple[str, str]] = []
    for _ in range(nfields):
        tc = chr(data[p]); p += 1
        fname, p = _read_utf(data, p)
        if tc in 'L[':                 # campo oggetto: segue il nome del tipo
            tag = data[p]; p += 1
            if tag == 0x74:            # TC_STRING: tipo scritto per esteso
                _, p = _read_utf(data, p)
            elif tag == 0x71:          # TC_REFERENCE a una stringa già vista
                p += 4
            else:
                raise ValueError(f'tag tipo inatteso {tag:#x} @ {p}')
        fields.append((fname, tc))

    if data[p:p + 2] != _TC_ENDBLOCKDATA_NULL:
        raise ValueError('fine del class descriptor inattesa')
    p += 2

    vals: dict[str, object] = {}
    for fname, tc in fields:
        if tc in 'L[':                 # i campi oggetto non sono inline
            continue
        vals[fname] = struct.unpack(_PRIM_FMTS[tc], data[p:p + _PRIM_SIZES[tc]])[0]
        p += _PRIM_SIZES[tc]
    return vals


def extract_irs(data: bytes, ir_len: int = IR_LEN_DEFAULT) -> list[np.ndarray]:
    """Tutti gli array ``float[]`` di lunghezza ``ir_len``: una IR per sweep.

    Il filtro sulla lunghezza è ciò che distingue le IR dagli altri array
    presenti nel file (risposta in frequenza su bin lineari con
    ``dataLength`` ≈ 60021 campioni, curve di calibrazione, icona di
    anteprima). Gli array sono restituiti in ordine di offset nello stream,
    che coincide con l'ordine cronologico degli sweep.
    """
    out: list[tuple[int, np.ndarray]] = []

    m = re.search(re.escape(_ARRAY_FULL_DESC), data)
    if m:
        i = m.start()
        ln = struct.unpack('>i', data[i + 19:i + 23])[0]
        if ln == ir_len:
            out.append((i, np.frombuffer(data[i + 23:i + 23 + 4 * ln], dtype='>f4')))

    for mm in re.finditer(_ARRAY_REF_DESC, data):
        j = mm.start()
        ln = struct.unpack('>i', data[j + 6:j + 10])[0]
        if ln == ir_len and j + 10 + 4 * ln <= len(data):
            arr = np.frombuffer(data[j + 10:j + 10 + 4 * ln], dtype='>f4')
            # scarta falsi positivi: un array di IR è finito e di ampiezza ~1
            if np.isfinite(arr).all() and np.abs(arr).max() < 1e6:
                out.append((j, arr))

    return [a.astype(np.float64) for _, a in sorted(out)]


@dataclass
class Measurement:
    """Contenuto utile di un `.mdat`: metadati + IR dei singoli sweep."""

    path: Path
    fields: dict[str, object] = field(repr=False, default_factory=dict)
    irs: list[np.ndarray] = field(repr=False, default_factory=list)

    @property
    def fs(self) -> float:
        return float(self.fields['sampleRate'])

    @property
    def mean_ir(self) -> np.ndarray:
        """Media coerente degli sweep.

        Valida solo se le IR sono allineate: il segnale deterministico si somma
        in ampiezza, il rumore incoerente in potenza, con guadagno di SNR pari
        a 10·log10(N). REW allinea internamente le IR sul picco.
        """
        return np.mean(self.irs, axis=0)

    @property
    def room_dims(self) -> tuple[float, float, float]:
        f = self.fields
        return (float(f['roomLength']), float(f['roomWidth']), float(f['roomHeight']))


def read_mdat(path: str | Path, ir_len: int = IR_LEN_DEFAULT) -> Measurement:
    """Legge un `.mdat` REW e restituisce metadati e IRs."""
    path = Path(path)
    data = path.read_bytes()
    return Measurement(path=path,
                       fields=parse_measdata_fields(data),
                       irs=extract_irs(data, ir_len=ir_len))


def load_rew_export(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Carica un export testuale REW e restituisce ``(freq, dB)``.

    Il numero di righe di header varia con la versione di REW e col tipo di
    misura, quindi lo si determina cercando la prima riga il cui primo campo
    è convertibile a float invece di fissare ``skiprows``.
    """
    path = Path(path)
    skip = 0
    with open(path) as fp:
        for line in fp:
            try:
                float(line.split()[0])
                break
            except (ValueError, IndexError):
                skip += 1
    data = np.loadtxt(path, skiprows=skip)
    return data[:, 0], data[:, 1]


def write_cal_file(path: str | Path, freq: np.ndarray, db: np.ndarray,
                   phase: np.ndarray | None = None, note: str = '') -> Path:
    """Scrive un file di calibrazione nel formato REW.

    Con ``phase=None`` (default) scrive due colonne ``freq dB``: REW ricostruisce
    la fase a minima fase dal modulo, che e' l'ipotesi corretta per un
    trasduttore. Scrivere una colonna di fase identicamente nulla insieme a un
    modulo non piatto descrive invece un filtro non causale.

    Caricabile in REW da *Preferences -> Cal files*. Le righe di commento
    devono iniziare con ``*``.
    """
    path = Path(path)
    cols = [freq, db] if phase is None else [freq, db, phase]
    header = note if note else 'Calibration file'
    np.savetxt(path, np.column_stack(cols),
               fmt='%.2f', header=header, comments='* ')
    return path


def frac_octave_smooth(freq: np.ndarray, db: np.ndarray,
                       fraction: float = 3.0) -> np.ndarray:
    """Lisciatura a 1/``fraction`` di ottava su una griglia log-spaziata.

    Media mobile pesata Hann in log(f). Sui dati REW (passo costante in ppo)
    la larghezza in punti e' fissa, quindi la finestra e' esattamente una
    frazione di ottava a ogni frequenza. Serve a togliere il ripple da
    interferenza (riflessioni) lasciando l'andamento lento, che e' l'unico
    che possa plausibilmente essere risposta del trasduttore.
    """
    ppo = 1.0 / np.median(np.diff(np.log2(freq))) # punti per ottava
    n = max(3, int(round(ppo / fraction)))
    if n % 2 == 0:
        n += 1
    w = np.hanning(n + 2)[1:-1]
    w /= w.sum()
    pad = n // 2
    padded = np.concatenate([np.full(pad, db[0]), db, np.full(pad, db[-1])])
    return np.convolve(padded, w, mode='valid')


def log_taper(freq: np.ndarray, f_lo: float, f_hi: float,
              octaves: float = 1.0) -> np.ndarray:
    """Finestra 0->1->0 con raccordi a coseno rialzato in log(f).

    Sale da 0 a 1 fra ``f_lo`` e ``f_lo * 2**octaves``, scende fra
    ``f_hi / 2**octaves`` e ``f_hi``. Sostituisce il troncamento netto: una
    discontinuita' nel modulo corrisponde a una risposta impulsiva che decade
    come 1/t (fenomeno di Gibbs) e sporca la coda della IR.
    """
    x = np.log2(freq)
    w = np.zeros_like(freq)
    up = (x - np.log2(f_lo)) / octaves
    dn = (np.log2(f_hi) - x) / octaves
    ramp = lambda t: 0.5 * (1.0 - np.cos(np.pi * np.clip(t, 0.0, 1.0)))
    w = ramp(up) * ramp(dn)
    return w
