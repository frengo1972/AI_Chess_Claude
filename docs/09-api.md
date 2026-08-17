# Riferimento API

Base URL di sviluppo: `http://127.0.0.1:8077`. Documentazione interattiva generata da
FastAPI: `http://127.0.0.1:8077/docs`.

> La porta di default è 8077 e non 8000, perché 8000 è quasi sempre già occupata su una
> macchina di sviluppo. Si cambia con la variabile d'ambiente `AICHESS_PORT` o con
> `--port`; se la cambi, aggiorna anche il proxy in `frontend/vite.config.ts`.

## Sistema

| Metodo | Percorso | Descrizione |
|---|---|---|
| `GET` | `/api/health` | stato del servizio |
| `GET` | `/api/system` | versione di torch, disponibilità CUDA, device, stato di Stockfish |
| `GET` | `/api/models` | elenco dei checkpoint disponibili con parametri, Elo e partite |
| `POST` | `/api/models/reload` | invalida la cache dei modelli (dopo un nuovo checkpoint) |

## Partite

### `POST /api/game`

```json
{
  "human_color": "white",
  "opponent": {
    "kind": "neural",
    "model_id": "small-ab12cd34/best",
    "simulations": 200
  },
  "assistance_enabled": true
}
```

`kind` può essere `neural` (la rete) o `stockfish` (con `stockfish_elo` da 1320 a 3190,
oppure `stockfish_skill` da 0 a 20). `human_color` accetta `white`, `black` o `none`
(partita motore contro motore, senza umano).

La risposta contiene lo stato completo della partita e, se l'umano gioca col Nero, la
prima mossa del motore.

### `POST /api/game/{id}/move`

```json
{ "from": "e2", "to": "e4", "promotion": null }
```

Applica la mossa umana, valida la legalità lato server e fa rispondere il motore nella
stessa richiesta. `promotion` è `q`, `r`, `b` o `n`.

### Altri endpoint di partita

| Metodo | Percorso | Descrizione |
|---|---|---|
| `GET` | `/api/game/{id}` | stato corrente |
| `POST` | `/api/game/{id}/engine-move` | forza una mossa del motore |
| `POST` | `/api/game/{id}/undo` | annulla `plies` semimosse (default 2) |
| `POST` | `/api/game/{id}/resign` | abbandono |
| `POST` | `/api/game/{id}/neural-evaluation` | valutazione **della rete** sulla posizione |
| `GET` | `/api/game/{id}/pgn` | PGN della partita |
| `DELETE` | `/api/game/{id}` | elimina la sessione |

### Stato della partita

Il payload `game` contiene fra l'altro:

| Campo | Contenuto |
|---|---|
| `fen`, `turn`, `human_to_move` | stato base |
| `legal_moves` | `{ "e2": ["e3","e4"], … }` — usato dal drag & drop |
| `promotion_moves` | mosse che richiedono la scelta del pezzo |
| `moves_san`, `moves_uci`, `pgn` | notazione |
| `in_check`, `checkers` | evidenziazione del re sotto scacco |
| `material`, `captured` | statistiche laterali |
| `engine_reports` | ultime valutazioni del motore (valore, simulazioni, nodi, PV) |
| `is_over`, `result`, `termination` | esito |

## Analisi (solo per l'umano)

| Metodo | Percorso | Note |
|---|---|---|
| `GET` | `/api/analysis/engine` | disponibilità e percorso del binario Stockfish |
| `POST` | `/api/analysis/position` | analisi libera di una FEN (`depth`, `multipv`) |
| `POST` | `/api/analysis/hint` | analisi della partita corrente — **403** se non c'è un umano o se l'assistenza è disattivata |

Vedi [l'isolamento](08-isolamento-stockfish.md) per la motivazione.

## Training

| Metodo | Percorso | Descrizione |
|---|---|---|
| `GET` | `/api/training/presets` | preset disponibili con dimensione della rete calcolata |
| `POST` | `/api/training/start` | avvia un run (processo separato) |
| `POST` | `/api/training/stop` | stop cooperativo (`force: true` per terminare subito) |
| `GET` | `/api/training/status` | stato corrente, fase, ultima iterazione |
| `GET` | `/api/training/history?run_id=…` | tutte le righe di KPI del run |
| `GET` | `/api/training/events?run_id=…` | log testuale del run |
| `GET` | `/api/training/games?run_id=…` | PGN di partite di esempio |
| `GET` | `/api/training/runs` | elenco dei run |
| `WS` | `/api/training/ws` | push dello stato ogni 1,5 s |
| `GET` | `/api/training/watch?run_id=…` | partite di self-play in corso, una per worker |
| `POST` | `/api/training/watch/settings` | attiva/disattiva la pubblicazione e la pausa fra le mosse |
| `WS` | `/api/training/watch/ws` | push delle scacchiere (cadenza in `interval_ms`, default 500 ms) |

### `POST /api/training/watch/settings`

```json
{ "run_id": null, "enabled": true, "move_delay_ms": 500 }
```

Entrambi i campi sono opzionali: quello omesso resta com'è. Ha effetto entro mezzo
secondo sul run già in corso, senza riavviarlo. Dettagli in
[Osservare il self-play](12-osservare-il-self-play.md).

## Benchmark

### `POST /api/benchmark`

```json
{ "model_id": "small-ab12cd34/best", "games": 20, "stockfish_elo": 1400, "movetime_ms": 50 }
```

Fa giocare la rete contro Stockfish a forza limitata e restituisce vittorie, patte,
sconfitte, score ed Elo stimato con intervallo di confidenza. È una richiesta lunga.

## Documentazione

| Metodo | Percorso | Descrizione |
|---|---|---|
| `GET` | `/api/docs` | indice dei documenti (slug, titolo, sommario) |
| `GET` | `/api/docs/{slug}` | contenuto Markdown |
