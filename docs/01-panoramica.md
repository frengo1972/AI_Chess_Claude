# Panoramica del progetto

Questo progetto è un esperimento completo di *reinforcement learning*: una rete neurale
impara a giocare a scacchi **giocando solo contro sé stessa**, senza database di partite
umane, senza libri di apertura e senza mai vedere l'output di un motore tradizionale.
Quando l'apprendimento è finito, la rete può affrontare un essere umano oppure Stockfish
a forza limitata, così da misurare quanto è diventata forte.

## Le tre parti

```
┌──────────────────────────────────────────────────────────────────────┐
│  APPRENDIMENTO (processo separato, CPU + GPU)                        │
│                                                                      │
│   self-play ──► replay buffer ──► training ──► arena ──► promozione  │
│      ▲                                                        │      │
│      └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────┘
                                  │  checkpoint .pt + KPI su SQLite
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  BACKEND FastAPI                                                     │
│                                                                      │
│   /api/game        →  motore neurale  (rete + MCTS)                  │
│   /api/analysis    →  Stockfish       (SOLO per l'umano)             │
│   /api/training    →  controllo e KPI del training                   │
│   /api/docs        →  questa documentazione                          │
└──────────────────────────────────────────────────────────────────────┘
                                  │  HTTP / WebSocket
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FRONTEND Vite + React                                               │
│                                                                      │
│   /            scacchiera a tutto schermo, pezzi trascinabili        │
│   /training    dashboard KPI in tempo reale                          │
│   /docs        documentazione                                        │
└──────────────────────────────────────────────────────────────────────┘
```

## Come sceglie una mossa il motore neurale

Il punto chiave dell'implementazione è che **le regole non vengono imparate**: sono
imposte dalla struttura del codice. Ogni mossa nasce da due fasi nettamente separate.

```
posizione
   │
   ├─► FASE 1  generatore di mosse legali          (python-chess)
   │           produce l'insieme M delle mosse legali, e solo quelle
   │
   └─► FASE 2  rete neurale + ricerca              (PyTorch + MCTS)
               assegna una probabilità a ciascuna mossa di M
               e un valore alla posizione, poi la ricerca sceglie
```

La rete produce sempre 4672 numeri (tutte le mosse geometricamente concepibili sulla
scacchiera). Prima di usarli, la maschera delle mosse legali azzera tutto ciò che non
appartiene a `M`. La rete quindi **non può proporre una mossa illegale**, nemmeno appena
inizializzata con pesi casuali.

Vedi [le regole e la generazione delle mosse](02-regole-e-mosse-legali.md) e
[la codifica](03-codifica.md).

## Le due modalità di gioco della rete

Lo stesso codice copre due stili, controllati da un solo parametro,
`search.simulations`:

| `simulations` | Comportamento | Costo per mossa | Forza |
|---|---|---|---|
| `1` | **Policy pura**: la rete valuta la posizione una volta e gioca la mossa legale con probabilità più alta. Nessuna ricerca. | 1 forward pass | bassa, ma i dati si generano ~50-100× più in fretta |
| `N > 1` | **AlphaZero**: albero PUCT con `N` valutazioni; la mossa scelta è quella più visitata. | `N` forward pass | molto più alta |

Questo permette di confrontare direttamente i due approcci nella stessa dashboard.
Vedi [la ricerca MCTS](05-ricerca-mcts.md).

## L'aiuto è solo per l'umano

Nel frontend, mentre giochi, vedi la barra di valutazione e le mosse migliori calcolate
da **Stockfish**. Serve a permetterti di giocare a un buon livello contro la rete. La
rete neurale non ha alcun accesso a quelle informazioni: non è una regola di buona
educazione, è una proprietà verificata da un test automatico.
Vedi [l'isolamento](08-isolamento-stockfish.md).

## Aspettative realistiche

Addestrare AlphaZero "vero" ha richiesto migliaia di TPU. Su un portatile con una GPU da
8 GB, l'obiettivo ragionevole con il preset `small` è:

* dopo ~1 ora: la rete conosce il valore dei pezzi e non regala materiale in modo grossolano;
* dopo ~6-12 ore: gioca aperture sensate, riconosce matti semplici, stimabile intorno
  a 800-1400 Elo contro Stockfish limitato;
* oltre: serve scalare rete, simulazioni e partite per iterazione (preset `medium`/`large`).

I preset sono in `backend/app/config.py` e sono esposti nella pagina di training.
