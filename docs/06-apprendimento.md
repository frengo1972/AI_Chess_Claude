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

### Il bersaglio della testa value

`z` da solo è un bersaglio corretto ma **rumorosissimo**: un unico esito appiccicato
identico a tutte le ~80 posizioni della partita, tutte fortemente correlate fra loro.
Una svista alla mossa 60 marchia come "persa" anche l'apertura giocata bene. La testa
value passa così le prime iterazioni a inseguire l'esito delle *partite* invece del
merito delle *posizioni*.

Il rimedio è mescolare `z` con `q`, il valore che l'albero ha effettivamente concluso
per quella posizione:

```
bersaglio = (1 − w) · z  +  w · q          w = train.value_search_weight
```

`q` è un'opinione a varianza molto più bassa e — punto essenziale — viene dalla
**ricerca della rete stessa**: non entra nessuna conoscenza scacchistica esterna, quindi
l'obiettivo resta "impara da sola". È la stessa idea del bootstrap TD, ed è ciò che
usano KataGo e Leela in varie forme.

| `w` | Effetto |
|---|---|
| `0.0` | bersaglio AlphaZero originale (solo esito). Default della dataclass |
| `0.5` | valore dei preset: dimezza la varianza mantenendo l'ancoraggio al risultato reale |
| `1.0` | solo ricerca: la rete non vede più l'esito, rischia di convincersi di sé stessa |

Viene **ignorato quando `simulations == 1`** (preset `policy-only`): senza albero `q` è
l'output stesso della testa value, e farle inseguire la propria previsione non insegna
niente. Il KPI *scarto ricerca / risultato* mostra la media di `|z − q|`, cioè quanto
lavoro il mix sta effettivamente facendo.

Attenzione a cosa **non** è: non è un reward parziale su materiale, centro o sicurezza
del re. Quelli sono euristiche umane, e sommati al bersaglio della value distorcono la
politica ottima — la rete smetterebbe di considerare i sacrifici, e l'arena non se ne
accorgerebbe perché candidato e campione sarebbero distorti allo stesso modo.

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

## Interrompere e riprendere

Una rete forte non nasce in una sessione. Un run è pensato per essere fermato e ripreso
anche giorni dopo, accumulando forza nel tempo invece di ricominciare da zero.

Nella cartella del run c'è tutto il necessario:

| File | Contenuto | Perché serve al ritorno |
|---|---|---|
| `best.pt` | il campione | è la rete che gioca e che genera i dati |
| `trainer.pt` | momenti dell'ottimizzatore, posizione dello scheduler, stato dei generatori casuali | fa *continuare* l'iterazione N+1 invece di farla ripartire |
| `state.json` | iterazione, partite, posizioni, Elo, ore, numero di sessioni | i contatori su cui poggiano i KPI |
| `replay/iter-*.npz` | gli ultimi ~25 shard | il buffer non ricomincia vuoto |
| `watch/settings.json` | preferenze dello spettatore | vedi [osservare il self-play](12-osservare-il-self-play.md) |

Senza `trainer.pt` una ripresa perderebbe silenziosamente tre cose, tutte costose:

* i **momenti di AdamW**, che vanno ricostruiti in qualche centinaio di passi rumorosi;
* la **posizione dello scheduler**, cioè il learning rate tornerebbe al valore iniziale
  — troppo alto per una rete già formata, capace di rovinare in un'iterazione il lavoro
  di venti;
* lo **stato del generatore casuale**, quindi il self-play ripartirebbe con gli stessi
  semi e produrrebbe dati quasi duplicati.

Lo stato viene riscritto a **ogni iterazione**, comprese quelle che si limitano a
raccogliere dati perché il buffer è ancora sotto la soglia: un run interrotto in
qualsiasi momento riprende dall'ultima iterazione completata, senza rifarla.

Dall'interfaccia: seleziona il run e usa il pulsante **Riprendi**, che mostra da dove
ripartirebbe (iterazione, partite, Elo, ore di calcolo). Il campo *Iterazioni* diventa
"quante altre". Da riga di comando basta l'id del run:

```bash
python -m app.engine.train --run-id small-ab12cd34 --resume --iterations 40
```

La configurazione non va ripetuta: viene riletta da `checkpoints/<run>/config.json`,
perché la forma della rete deve combaciare con il checkpoint. I run precedenti a questa
funzione riprendono comunque dai pesi, con il learning rate riportato al punto giusto in
base al numero di iterazioni e un avviso nel registro.

Se invece vuoi cambiare la *forma* della rete (più blocchi, più filtri) devi avviare un
run nuovo: quei pesi non sono trasferibili così come sono.

Un avvertimento sui run "fantasma": se il trainer viene ucciso, la sua riga resta a
`running` perché è lui a scriverne lo stato finale. L'interfaccia lo riconosce da
`status.json` non più aggiornato da mezz'ora e sblocca comunque **Riprendi**, avvisando.
L'API però **non può vedere un trainer che non ha avviato lei** — per esempio dopo un
riavvio del server con l'addestramento in corso: in quel caso controlla che il processo
sia davvero morto prima di riprendere, perché due trainer sulla stessa cartella si
sovrascriverebbero i checkpoint a vicenda.

## Problemi tipici e cosa guardare

| Sintomo | KPI da guardare | Causa probabile |
|---|---|---|
| La rete non migliora mai | `arena_score` sempre ~0.5 | troppe poche partite per iterazione, o `steps_per_iteration` troppo basso |
| Tutte patte, sempre | `draw_rate` → 1.0, `avg_plies` al massimo | `max_game_plies` troppo basso, oppure rete troppo debole per convertire |
| `value_loss` scende, la forza no | `policy_entropy` che crolla | collasso della policy: alza `dirichlet_epsilon` o `temperature_moves` |
| `policy_loss` esplode | `learning_rate` | riduci il learning rate o abbassa `grad_clip` |
| Iterazioni lentissime | `games_per_minute` | riduci `simulations`, o aumenta `workers` |
