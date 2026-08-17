# Installazione e uso

## Requisiti

* Python 3.12 (3.11-3.13 vanno bene; PyTorch non ha ancora ruote stabili per ogni
  release più recente)
* Node.js 20+
* Facoltativa ma consigliata: una GPU NVIDIA con driver recenti

## Primo avvio

```bash
# 1. ambiente Python
cd backend
py -3.12 -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt

# 2. PyTorch con CUDA (salta l'index se vuoi la versione CPU)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 3. asset di terze parti: Stockfish + grafica dei pezzi
cd ..
python scripts/download_assets.py

# 4. frontend
cd frontend
npm install
```

Su Windows è disponibile anche `scripts/setup.ps1`, che esegue tutti i passaggi.

## Avvio quotidiano

Due terminali:

```bash
# terminale 1 — API
cd backend
.venv\Scripts\activate
python -m uvicorn app.main:app --port 8077

# terminale 2 — interfaccia
cd frontend
npm run dev
```

Poi apri `http://localhost:5173`.

Su Windows, `scripts/dev.ps1` avvia entrambi (`-Port` per cambiare porta).

La porta di default dell'API è **8077**: la 8000 è quasi sempre già presa. Se la cambi,
aggiorna il proxy in `frontend/vite.config.ts`.

## Le tre pagine

**Gioca** (`/`) — la scacchiera occupa tutta l'altezza dello schermo, i pezzi si spostano
trascinandoli (o con due clic). Sui bordi:

* a sinistra la barra di valutazione, l'analisi di Stockfish con le mosse migliori e le
  frecce sulla scacchiera, il bilancio materiale;
* a destra l'elenco delle mosse, i dati della rete avversaria (valore, simulazioni, nodi,
  variante principale), i comandi.

L'interruttore **Assistenza** disattiva completamente l'aiuto di Stockfish, per giocare
alla pari con la rete.

**Training** (`/training`) — avvio e arresto degli addestramenti, scelta del preset,
KPI in tempo reale e grafici. Vedi [la guida ai KPI](07-kpi.md).

**Documentazione** (`/docs`) — queste pagine.

## Addestrare da riga di comando

```bash
cd backend
python -m app.engine.train --preset small --name esperimento1 --iterations 60
```

| Opzione | Effetto |
|---|---|
| `--preset` | `tiny`, `small`, `medium`, `large`, `policy-only` |
| `--games` | partite di self-play per iterazione |
| `--simulations` | simulazioni MCTS per mossa (`1` = policy pura) |
| `--workers` | processi di self-play |
| `--device` | `auto`, `cpu`, `cuda` |
| `--resume --run-id <id>` | riprende un run esistente |

I risultati finiscono in `backend/checkpoints/<run-id>/` e i KPI in
`backend/data/training.db`.

## Dimensionamento sulla macchina

| Parametro | Regola pratica |
|---|---|
| `selfplay.workers` | core fisici − 2. Ogni worker usa un thread torch |
| `search.simulations` | il moltiplicatore di forza più diretto; raddoppiarlo raddoppia il tempo per mossa |
| `train.batch_size` | il più grande che entra in VRAM (256 su 8 GB con il preset `small`) |
| `train.replay_buffer_size` | ~460 byte per posizione, quindi 200.000 posizioni ≈ 90 MB |

## Problemi frequenti

| Sintomo | Rimedio |
|---|---|
| `Stockfish binary not found` | riesegui `python scripts/download_assets.py --stockfish`, oppure imposta `AICHESS_STOCKFISH_PATH` |
| `torch.cuda.is_available()` è `False` | hai installato la ruota CPU; reinstalla dall'index CUDA |
| Il training non parte dall'interfaccia | guarda il log in `backend/checkpoints/logs/` |
| I pezzi non si vedono | riesegui `python scripts/download_assets.py --pieces` |
| Il self-play satura il PC | riduci `--workers` |

## Licenze di terze parti

* **Stockfish** — GPL-3.0. Scaricato in `engines/`, non ridistribuito con il codice.
* **Pezzi cburnett** — CC BY-SA 3.0, da Lichess. L'attribuzione è in
  `frontend/public/pieces/ATTRIBUTION.md`.
* La grafica dei pezzi di chess.com è proprietaria e **non** viene usata.
