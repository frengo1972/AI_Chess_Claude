# AI Chess — una rete neurale che impara gli scacchi da sola

Esperimento completo di *reinforcement learning*: una rete neurale impara a giocare a
scacchi **giocando contro sé stessa**, senza partite umane, senza libri di apertura e
senza mai vedere l'output di un motore tradizionale. Finito l'addestramento, può
affrontare un essere umano o Stockfish a forza limitata per misurare quanto è diventata
forte.

```
 self-play ──► replay buffer ──► training ──► arena ──► promozione ──┐
     ▲                                                               │
     └───────────────────────────────────────────────────────────────┘
```

## Cosa c'è dentro

| | |
|---|---|
| **Motore neurale** | ResNet policy/value in stile AlphaZero + ricerca PUCT-MCTS. `simulations = 1` degenera in "policy pura", `simulations = N` è AlphaZero |
| **Regole** | delegate a `python-chess`: la rete non può nemmeno proporre una mossa illegale |
| **Apprendimento** | self-play multiprocesso su CPU, backprop su GPU, gating in arena, KPI su SQLite |
| **Interfaccia** | Vite + React, scacchiera a tutto schermo con drag & drop, statistiche sui bordi |
| **Self-play in diretta** | le partite di addestramento su più scacchiere, rallentabili per seguirle a occhio |
| **Assistenza** | valutazione e mosse migliori di Stockfish — **solo per l'umano**, verificato da test |
| **Documentazione** | 12 pagine consultabili dall'app stessa (`/docs`) |

## Avvio rapido

```powershell
# tutto in un colpo (Windows)
.\scripts\setup.ps1
.\scripts\dev.ps1
```

Manualmente:

```bash
cd backend
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu130   # o senza index per CPU

cd ..
python scripts/download_assets.py        # Stockfish + grafica dei pezzi

cd frontend && npm install
```

Poi, in due terminali:

```bash
cd backend && python -m uvicorn app.main:app --port 8077
cd frontend && npm run dev
```

Interfaccia su <http://localhost:5173>, API su <http://127.0.0.1:8077>
(OpenAPI su `/docs`).

## Addestrare

Dalla pagina **Training**, oppure:

```bash
cd backend
python -m app.engine.train --preset small --name esperimento1 --iterations 60
```

Preset disponibili: `tiny`, `small`, `medium`, `large`, `policy-only`. I checkpoint
finiscono in `backend/checkpoints/<run-id>/`, i KPI in `backend/data/training.db`.

Un run si può fermare e riprendere anche giorni dopo, accumulando forza nel tempo:
il pulsante **Riprendi** della pagina Training, oppure

```bash
python -m app.engine.train --run-id small-ab12cd34 --resume --iterations 40
```

Riprendono insieme ai pesi anche l'ottimizzatore, il learning rate, il replay buffer e
l'Elo accumulato — [perché conta](docs/06-apprendimento.md).

## Il vincolo centrale

La barra di valutazione e i suggerimenti servono all'**essere umano**, per poter giocare
alla pari con la rete. La rete non ha accesso a quelle informazioni, e non è una
convenzione: `backend/tests/test_engine_isolation.py` verifica staticamente e a runtime
che nessun modulo del motore possa raggiungere Stockfish, e l'API rifiuta con 403 le
richieste di assistenza nelle partite senza giocatore umano.

Dettagli in [`docs/08-isolamento-stockfish.md`](docs/08-isolamento-stockfish.md).

## Documentazione

| | |
|---|---|
| [01 Panoramica](docs/01-panoramica.md) | architettura e aspettative realistiche |
| [02 Regole e mosse legali](docs/02-regole-e-mosse-legali.md) | come le regole vincolano la rete |
| [03 Codifica](docs/03-codifica.md) | piani di input, spazio d'azione 4672 |
| [04 Rete neurale](docs/04-rete-neurale.md) | architettura e dimensionamento |
| [05 Ricerca MCTS](docs/05-ricerca-mcts.md) | PUCT, virtual loss, temperatura |
| [06 Apprendimento](docs/06-apprendimento.md) | il ciclo di RL |
| [07 KPI](docs/07-kpi.md) | come leggere la dashboard |
| [08 Isolamento](docs/08-isolamento-stockfish.md) | perché la rete non vede Stockfish |
| [09 API](docs/09-api.md) | riferimento HTTP |
| [10 Avvio](docs/10-avvio.md) | installazione, uso, problemi frequenti |
| [11 Riferimenti](docs/11-riferimenti.md) | fonti e scelte progettuali |
| [12 Osservare il self-play](docs/12-osservare-il-self-play.md) | guardare l'addestramento mentre avviene |

## Test

```bash
cd backend
python -m pytest -q
```

## Licenze di terze parti

* **Stockfish** — GPL-3.0, scaricato in `engines/` (non ridistribuito).
* **Pezzi cburnett** — CC BY-SA 3.0 (Colin M.L. Burnett, via Lichess). La grafica di
  chess.com è proprietaria e **non** viene usata.
