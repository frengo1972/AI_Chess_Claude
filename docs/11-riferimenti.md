# Riferimenti e scelte progettuali

## Fonti principali

| Riferimento | Cosa ne è stato preso |
|---|---|
| Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (2017), arXiv:1712.01815 | l'intera impostazione: rete a due teste, MCTS con PUCT, codifica a piani 8×8, spazio d'azione 4672, ciclo self-play → training |
| Silver et al., *Mastering the game of Go without human knowledge* (2017) | rumore di Dirichlet alla radice, temperatura nell'apertura, gating del candidato in arena |
| Leela Chess Zero — `lczero.org/dev/lc0/search/alphazero/` | virtual loss, batching delle foglie, First Play Urgency, blocchi Squeeze-Excitation, `c_puct` dipendente dalle visite |
| Czech et al., *Representation Matters for Mastering Chess* (2023), arXiv:2304.14918 | conferma sperimentale dell'importanza della codifica di input rispetto all'architettura |
| Documentazione di `python-chess` — `python-chess.readthedocs.io` | generazione mosse legali, FEN/PGN, protocollo UCI |
| Protocollo UCI e opzioni di Stockfish | `UCI_LimitStrength` / `UCI_Elo`, `Skill Level`, `MultiPV` |

## Differenze consapevoli rispetto ad AlphaZero

| AlphaZero | Qui | Perché |
|---|---|---|
| `T = 8` passi di storia fissi | `T` configurabile, 4 di default | 63 piani invece di 119: meno memoria e self-play più veloce su un portatile |
| 19-20 blocchi × 256 filtri | 6 × 96 di default | con poche migliaia di partite una rete grande non ha abbastanza dati per riempirsi |
| 800 simulazioni per mossa | 96 di default, `1` in modalità policy pura | il self-play è il collo di bottiglia; meno simulazioni = più partite a parità di tempo |
| Nessun gating (nella versione finale) | gating in arena attivo | con pochi dati un'iterazione sfortunata fa danni; il gate rende il progresso monotono |
| SGD + momentum | AdamW di default | converge prima su run brevi; `sgd` resta disponibile via config |
| Patte reclamabili | ripetizione e 50 mosse **automatiche** | altrimenti la rete impara a rimescolare all'infinito |
| Nessuna resa | resa automatica con campione di controllo | risparmia molto tempo; il 10% di partite giocate fino in fondo misura i falsi positivi |
| TPU cluster | 1 GPU + processi CPU | i worker girano su CPU per non contendere la GPU, che resta per il training |

## Perché non NNUE

NNUE (la rete di valutazione di Stockfish) è una rete piccola e incrementale accoppiata a
una ricerca alfa-beta profondissima. È molto più forte a parità di hardware, ma è un
paradigma diverso: viene addestrata in modo supervisionato su valutazioni prodotte da un
motore già esistente. Qui l'obiettivo esplicito è **imparare dal nulla giocando contro sé
stessi**, quindi l'impianto AlphaZero è l'unico coerente con la richiesta.

## Perché non DQN

Il deep Q-learning classico richiede uno spazio d'azione piccolo e fisso. Negli scacchi
l'insieme delle mosse legali cambia a ogni posizione e l'orizzonte di ricompensa è
lunghissimo (una singola ricompensa alla fine della partita). Le implementazioni DQN per
scacchi restano storicamente molto deboli. La combinazione policy/value + ricerca risolve
entrambi i problemi: la ricerca fornisce il credit assignment che al DQN manca.

## File chiave del codice

| File | Contenuto |
|---|---|
| `backend/app/engine/encoding.py` | posizione ↔ tensori, mossa ↔ indice, generazione delle mosse legali |
| `backend/app/engine/network.py` | la rete residuale a due teste |
| `backend/app/engine/mcts.py` | ricerca PUCT con batching e virtual loss |
| `backend/app/engine/selfplay.py` | generazione delle partite in multiprocesso |
| `backend/app/engine/train.py` | il ciclo di apprendimento |
| `backend/app/engine/arena.py` | match di gating e stima Elo |
| `backend/app/engine/rules.py` | condizioni di fine partita |
| `backend/app/engine/replay.py` | replay buffer compresso |
| `backend/app/services/stockfish_service.py` | motore classico (solo per l'umano) |
| `backend/app/config.py` | tutti i parametri e i preset |
| `frontend/src/components/Board.tsx` | scacchiera con drag & drop |
| `frontend/src/pages/TrainingPage.tsx` | dashboard dei KPI |
