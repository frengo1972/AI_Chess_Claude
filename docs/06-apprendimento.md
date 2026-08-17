# L'apprendimento per rinforzo

File: `backend/app/engine/train.py`, `selfplay.py`, `replay.py`, `arena.py`

## Il ciclo

Un'iterazione è composta da quattro fasi:

```
   ┌───────────────────────────────────────────────────────────────┐
   │                                                               │
   │   1. SELF-PLAY          la rete "best" gioca N partite        │
   │      (processi CPU)     contro sé stessa, con rumore          │
   │            │                                                  │
   │            ▼                                                  │
   │   2. REPLAY BUFFER      le posizioni entrano in una coda      │
   │            │            FIFO da ~200.000 elementi             │
   │            ▼                                                  │
   │   3. TRAINING           una copia della rete fa gradient      │
   │      (GPU)              descent su batch campionati           │
   │            │                                                  │
   │            ▼                                                  │
   │   4. ARENA              candidato vs campione: promosso       │
   │            │            solo se score ≥ soglia                │
   │            └───────────────────────────────────────────────┐  │
   │                                                            │  │
   └────────────────────────────────────────────────────────────┘  │
                          la nuova "best" ricomincia ───────────────┘
```

## 1. Self-play

Per ogni posizione della partita si registrano tre cose:

| Cosa | Da dove viene | A cosa serve |
|---|---|---|
| posizione codificata | `PositionHistory.encode()` | input della rete |
| `π`, distribuzione delle visite | MCTS | bersaglio della testa policy |
| `z`, risultato finale | fine partita | bersaglio della testa value |

`z` è assegnato **dal punto di vista di chi muoveva in quella posizione**: se il Bianco
ha vinto, tutte le posizioni con il Bianco al tratto ricevono `+1`, quelle con il Nero
`−1`.

Le partite girano in processi separati (`ProcessPoolExecutor` con contesto `spawn`, come
richiesto su Windows). Ogni worker riceve il **percorso** del checkpoint, non il modello:
i tensori CUDA non attraversano i confini di processo. I worker girano su CPU con un
thread torch ciascuno — molti processi piccoli battono un'unica GPU contesa, e la GPU
resta libera per il training.

Sorgenti di varietà, indispensabili perché la rete non si avviti su sé stessa:

* rumore di Dirichlet sui prior della radice;
* campionamento con temperatura nelle prime ~20 semimosse;
* opzionalmente, alcune semimosse iniziali completamente casuali.

## 2. Replay buffer

Coda FIFO a dimensione fissa. Serve a due scopi:

* **decorrelare** i campioni: le posizioni consecutive di una partita sono quasi identiche,
  e allenarsi su un batch di posizioni consecutive dà gradienti pessimi;
* **stabilizzare**: mantenendo dati di più generazioni si evita che la rete insegua
  troppo velocemente sé stessa (un classico caso di instabilità nell'RL).

Ogni iterazione viene anche salvata su disco come shard `.npz` compresso, così un run
interrotto può ripartire senza buttare via i dati.

## 3. Training

Su un batch campionato uniformemente dal buffer:

```
loss  =  −Σ π(a)·log p(a)        (cross-entropy policy)
       +  (z − v)²               (MSE value)
       +  λ‖θ‖²                  (weight decay)
```

Note implementative:

* la cross-entropy usa il softmax **completo** su 4672: i logit illegali ricevono un
  gradiente verso il basso;
* il gradiente viene clippato a `grad_clip` per evitare che un batch anomalo distrugga
  i pesi;
* l'ottimizzatore di default è **AdamW** perché converge più in fretta su run brevi;
  `optimizer: "sgd"` riproduce la ricetta originale di AlphaZero (SGD + momentum) ed è
  preferibile su run lunghi.

Se il buffer non ha ancora `min_buffer_before_training` posizioni, l'iterazione salta il
training e si limita a raccogliere dati. È normale nelle prime iterazioni.

## 4. Arena e promozione

Il candidato gioca `arena.games` partite contro il campione, a colori alternati e con
poche semimosse iniziali casuali (altrimenti sarebbe sempre la stessa partita). Viene
promosso solo se lo score raggiunge `win_threshold` (0.55 di default). Altrimenti i pesi
vengono riportati al campione e l'ottimizzatore reinizializzato.

Questo gating è ciò che rende il progresso **monotono**: senza, un'iterazione sfortunata
peggiora la rete, che genera dati peggiori, che peggiorano ulteriormente la rete.

La differenza Elo si ricava dallo score:

```
ΔElo = −400 · log₁₀(1/score − 1)
```

e viene accumulata: la curva Elo della dashboard è una scala *relativa* alla rete
iniziale. Per un'ancora assoluta serve il benchmark contro Stockfish
(vedi [KPI](07-kpi.md)).

## Come lanciarlo

Dalla pagina **Training** dell'interfaccia, oppure da riga di comando:

```bash
cd backend
python -m app.engine.train --preset small --name esperimento1 --iterations 60
```

Opzioni utili: `--games`, `--simulations`, `--workers`, `--device`, `--resume --run-id <id>`.

L'arresto è cooperativo: l'interfaccia (o `touch checkpoints/<run>/stop.flag`) chiede lo
stop, e il trainer esce alla fine dell'iterazione corrente lasciando tutto consistente.

## Problemi tipici e cosa guardare

| Sintomo | KPI da guardare | Causa probabile |
|---|---|---|
| La rete non migliora mai | `arena_score` sempre ~0.5 | troppe poche partite per iterazione, o `steps_per_iteration` troppo basso |
| Tutte patte, sempre | `draw_rate` → 1.0, `avg_plies` al massimo | `max_game_plies` troppo basso, oppure rete troppo debole per convertire |
| `value_loss` scende, la forza no | `policy_entropy` che crolla | collasso della policy: alza `dirichlet_epsilon` o `temperature_moves` |
| `policy_loss` esplode | `learning_rate` | riduci il learning rate o abbassa `grad_clip` |
| Iterazioni lentissime | `games_per_minute` | riduci `simulations`, o aumenta `workers` |
