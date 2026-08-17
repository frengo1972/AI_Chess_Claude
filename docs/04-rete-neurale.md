# La rete neurale

File: `backend/app/engine/network.py`

## Architettura

Una singola torre residuale convoluzionale con due teste, come in AlphaZero:

```
input  (14·T + 7, 8, 8)
   │
   ├─ Conv 3×3 → F filtri, BatchNorm, ReLU
   │
   ├─ ResidualBlock × B        ┐
   │    Conv 3×3 → BN → ReLU   │  ogni blocco:
   │    Conv 3×3 → BN          │    out = ReLU(x + f(x))
   │    (+ Squeeze-Excitation) │
   │                           ┘
   ├──────────────┬────────────────────────┐
   │              │                        │
 TESTA POLICY   TESTA VALUE
 Conv 3×3 → F   Conv 1×1 → 32
 BN, ReLU       BN, ReLU
 Conv 1×1 → 73  Linear → 128 → ReLU
 flatten        Linear → 1 → tanh
   │              │
 4672 logit     valore ∈ [-1, 1]
```

## Perché convoluzioni

La scacchiera è una griglia 8×8 e gli schemi scacchistici sono **locali e traslabili**:
un pedone isolato, una forchetta di cavallo, una torre su colonna aperta hanno la stessa
forma in qualunque punto della scacchiera. Una CNN condivide i pesi su tutte le case, e
impara quindi il concetto una volta sola. Ogni blocco residuale allarga di 2 case il
campo recettivo: con 6 blocchi la rete "vede" contemporaneamente circa metà scacchiera,
con 20 blocchi la vede tutta più volte.

## Le due teste

**Policy** — dice *dove guardare*. Non deve essere perfetta: deve concentrare la
probabilità sulle mosse che vale la pena esplorare, così l'MCTS non spreca simulazioni.
Il suo bersaglio in addestramento è la distribuzione delle visite dell'MCTS della
partita precedente: la ricerca migliora la policy, la policy rende la ricerca più
efficiente, e il ciclo si autoalimenta.

**Value** — dice *quanto vale la posizione*, in `[-1, +1]`, dal punto di vista di chi
muove. `tanh` garantisce il range. Il bersaglio è il risultato finale della partita in
cui quella posizione è comparsa (`+1` vittoria, `0` patta, `-1` sconfitta). All'inizio è
un segnale rumorosissimo — una posizione buona può comparire in una partita persa — ma
mediato su centinaia di migliaia di posizioni converge.

## Squeeze-Excitation (opzionale)

Con `network.se_ratio > 0`, ogni blocco residuale riceve un modulo SE: fa un pooling
globale, ne ricava una coppia (scala, offset) per canale e ri-modula l'uscita. È il
trucco introdotto da Leela Chess Zero per dare a una rete convoluzionale locale un canale
di informazione globale ("sono in un finale", "il mio re è esposto") a costo quasi nullo.
Disattivato di default nei preset piccoli.

## Dimensioni dei preset

| Preset | T | Blocchi | Filtri | Parametri | Peso |
|---|---|---|---|---|---|
| `tiny` | 2 | 3 | 64 | ~0,9 M | ~3,5 MB |
| `small` | 4 | 6 | 96 | ~4,0 M | ~15 MB |
| `medium` | 8 | 10 | 128 | ~12 M | ~46 MB |
| `large` | 8 | 20 | 256 | ~92 M | ~350 MB |

(I valori esatti sono mostrati dalla pagina Training, calcolati a runtime da
`ChessNet.describe()`.)

La regola pratica: **il numero di parametri non è il KPI da massimizzare**. Una rete
grande addestrata poco gioca peggio di una rete piccola addestrata molto, perché il
collo di bottiglia è il numero di partite di self-play, non la capacità. Su questa
macchina il preset `small` è il compromesso giusto.

## Inizializzazione e checkpoint

I pesi convoluzionali usano l'inizializzazione di Kaiming (adatta a ReLU), i layer lineari
Xavier. Un checkpoint (`.pt`) contiene lo `state_dict`, la `NetworkConfig` con cui è stato
costruito e i metadati (iterazione, partite giocate, Elo stimato): questo permette di
ricaricare un modello senza sapere a priori la sua forma, ed è ciò che rende possibile
avere modelli di preset diversi elencati insieme nel menu di gioco.

Il salvataggio è atomico (scrittura su `.tmp` e `replace`), così l'API non può mai leggere
un checkpoint scritto a metà mentre il trainer sta salvando.

## Inferenza: cache e batch

`backend/app/engine/evaluator.py` mette davanti alla rete due ottimizzazioni:

* **batching** — l'MCTS raccoglie fino a `search.max_batch_size` foglie prima di fare un
  singolo forward pass. Su GPU la differenza è enorme: valutare 16 posizioni insieme costa
  quasi quanto valutarne una.
* **cache** — le posizioni già valutate vengono memorizzate usando come chiave i **byte
  esatti del tensore di input**. Usare la FEN come chiave sarebbe più veloce ma sbagliato,
  perché due posizioni con la stessa FEN possono avere storia diversa e quindi input diverso.
