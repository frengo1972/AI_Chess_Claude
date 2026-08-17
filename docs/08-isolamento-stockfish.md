# Perché la rete non può vedere Stockfish

Il requisito è netto: la valutazione della posizione e il suggerimento delle mosse
migliori servono **all'essere umano**, per poter giocare a buon livello contro la rete.
La rete neurale non deve poterli usare, altrimenti l'esperimento perde senso — non
starebbe più imparando dal self-play, starebbe copiando.

Questa pagina descrive come il vincolo è imposto dal codice, non affidato alla buona fede.

## 1. Separazione strutturale

```
app/engine/           ← il motore neurale. NON importa mai Stockfish.
   encoding.py
   network.py
   evaluator.py
   mcts.py
   selfplay.py
   arena.py
   rules.py
   replay.py
   train.py           ← unica eccezione: import LAZY dentro _run_benchmark()

app/services/
   nn_engine.py       ← serve le mosse della rete. Nessun riferimento a Stockfish.
   stockfish_service.py  ← l'unico modulo che può avviare un processo UCI.
   benchmark.py       ← fa giocare rete vs Stockfish, ma restituisce solo uno SCORE.

app/api/
   routes_game.py     ← mosse della rete
   routes_analysis.py ← aiuto umano (Stockfish)
```

I due percorsi non si incontrano mai. Il motore neurale non ha un client HTTP, non legge
file prodotti da Stockfish e non riceve nessun parametro che possa iniettare una
valutazione esterna.

## 2. Test automatici che lo verificano

`backend/tests/test_engine_isolation.py`:

* **controllo statico**: l'AST di ogni modulo "di gioco" viene analizzato e si verifica
  che non contenga import che inizino con `chess.engine`, `app.services` o `stockfish`;
* **controllo su `train.py`**: nessun import proibito a livello di modulo, e l'unica
  funzione che importa `app.services` deve chiamarsi `_run_benchmark`;
* **controllo a runtime**: in un interprete pulito si importa tutto lo stack di gioco e si
  verifica che `app.services.stockfish_service` non compaia in `sys.modules`;
* **controllo di firma**: `NeuralEngineService.choose_move` non deve avere parametri
  chiamati `stockfish`, `engine`, `assist` o `hint` — una difesa contro il "lo aggiungo
  solo per provare" di domani.

Se qualcuno dovesse in futuro collegare i due mondi, la suite fallisce.

> Nota tecnica: il modulo `chess.engine` compare comunque in `sys.modules`, perché
> `chess.pgn` lo importa incondizionatamente come parte di python-chess. Il test lo
> ignora deliberatamente: ciò che conta è che `stockfish_service` — l'unico codice capace
> di avviare davvero un processo Stockfish — non venga mai caricato.

## 3. Controlli lato API

Gli endpoint di assistenza applicano ulteriori vincoli:

| Endpoint | Regola |
|---|---|
| `POST /api/analysis/hint` | 403 se la partita non ha un giocatore umano (`human_color = null`) |
| `POST /api/analysis/hint` | 403 se l'assistenza è stata disattivata per quella partita |
| `POST /api/analysis/position` | analisi libera di una FEN: è la scacchiera di analisi, non è legata a una partita |

Il caso più significativo è il primo: nelle partite **rete contro Stockfish** — quelle che
servono a misurare la forza — non esiste giocatore umano, quindi l'assistenza è negata
in modo esplicito.

Nel frontend c'è un interruttore *Assistenza* nel pannello di gioco: spegnendolo si gioca
contro la rete senza alcun aiuto, che è il modo corretto di valutare onestamente la
propria partita.

## 4. E il benchmark contro Stockfish?

È l'unico punto in cui i due motori si incontrano durante l'apprendimento, e il flusso di
informazione è rigorosamente a senso unico:

```
   rete ──── gioca N partite ────► Stockfish (Elo limitato)
                    │
                    ▼
              punteggio (vittorie/patte/sconfitte)
                    │
                    ▼
              KPI "Elo vs Stockfish"     ← si ferma qui
```

Nessuna mossa, valutazione o variante di Stockfish entra nel replay buffer. La funzione
`benchmark_against_stockfish` restituisce solo numeri aggregati, e `train.py` la importa
in modo lazy dentro un `try`, così che il motore resti utilizzabile anche su una macchina
dove Stockfish non è installato.

Il benchmark è disattivato di default (`benchmark.enabled = false`): va acceso solo
quando la rete è abbastanza forte da non perdere tutte le partite, altrimenti consuma
tempo per produrre sempre lo stesso zero.

## 5. Cosa la rete *può* mostrarti

Nel pannello laterale trovi anche la valutazione della **rete stessa** (`root_value`,
mosse più visitate, variante principale). Non è assistenza esterna: è l'output del
modello che stai affrontando, ed è utile proprio per capire dove sbaglia.
