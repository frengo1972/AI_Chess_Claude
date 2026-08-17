# Regole del gioco e generazione delle mosse legali

## Chi conosce le regole

Le regole degli scacchi non sono implementate a mano e non vengono apprese: sono
delegate a **`python-chess`**, una libreria matura che gestisce arrocco, en passant,
promozioni, scacco, scacco matto, stallo, materiale insufficiente, ripetizioni e regola
delle 50 mosse.

Il motore neurale interagisce con le regole in un solo punto:

```python
# backend/app/engine/encoding.py
def legal_move_indices(board: chess.Board) -> Dict[int, chess.Move]:
    white_to_move = board.turn == chess.WHITE
    return {move_to_index(move, white_to_move): move for move in board.legal_moves}
```

`board.legal_moves` è l'unica sorgente di mosse dell'intero sistema. Ogni valutazione
della rete viene ristretta a quell'insieme prima di essere usata
(`masked_softmax` in `evaluator.py`).

## Conseguenze pratiche

* Una rete con pesi casuali gioca comunque mosse **sempre legali**: sbaglia la scelta,
  non la regola.
* Non serve penalizzare le mosse illegali nella loss.
* Il frontend riceve dal backend l'elenco delle destinazioni legali per ogni casa di
  partenza, quindi non deve reimplementare le regole per il drag & drop.

## Fine partita: cosa conta come terminale

FIDE distingue tra patta *automatica* e patta *reclamabile*. In self-play la distinzione
è dannosa: se la ripetizione non è automatica, la rete impara a rimescolare i pezzi
all'infinito. Perciò `backend/app/engine/rules.py` rende automatiche tutte le patte.

| Condizione | Riconosciuta come | Valore per chi deve muovere |
|---|---|---|
| Scacco matto | fine partita | `-1` (ha perso) |
| Stallo | patta | `0` |
| Materiale insufficiente | patta | `0` |
| 50 mosse senza catture né mosse di pedone (`halfmove_clock >= 100`) | patta **automatica** | `0` |
| Terza ripetizione della posizione | patta **automatica** | `0` |
| Superato `max_game_plies` | patta per adjudication | `0` |

Il valore è sempre espresso **dal punto di vista di chi ha il tratto**. È la stessa
convenzione usata dalla testa *value* della rete e dal backup dell'albero MCTS: mantenerla
uniforme evita l'errore di segno più comune in queste implementazioni.

## Conteggio efficiente delle ripetizioni

Verificare la ripetizione con `board.is_repetition(3)` è costoso e l'MCTS lo farebbe
centinaia di volte per mossa. `PositionHistory` mantiene invece un `Counter` sulla
*transposition key* della posizione, aggiornato in `push()` e ripristinato in `pop()`:
il controllo diventa una singola lettura da dizionario.

```python
def repetition_count(self) -> int:
    return self._repetitions[self.board._transposition_key()]
```

Lo stesso contatore alimenta i due piani di input "ripetizione" descritti nella
[codifica](03-codifica.md).

## Adjudication per lunghezza

Le partite di self-play vengono troncate a `selfplay.max_game_plies` (240 semimosse nel
preset `small`) e registrate come patta. Serve a due cose: evitare che una rete acerba
consumi ore in finali senza senso, e mantenere prevedibile il tempo di un'iterazione.
Il rovescio della medaglia è che una rete che sta vincendo un finale lungo riceve un
segnale di patta: è un bias noto, mitigato alzando `max_game_plies` quando la rete
diventa più forte.

## Resa automatica

Se la valutazione alla radice resta sotto `selfplay.resign_threshold` (`-0.92`) per
`resign_consecutive` mosse consecutive, la partita viene troncata con una sconfitta.
Questo fa risparmiare molto tempo. Per controllare che la soglia non stia buttando via
partite recuperabili, una frazione delle partite (`resign_disable_fraction`, 10%) viene
sempre giocata fino in fondo.
