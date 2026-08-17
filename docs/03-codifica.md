# Codifica: dalla scacchiera ai tensori

Questo è il contratto fra le regole degli scacchi e la rete neurale. Un errore silenzioso
qui rovina un intero addestramento senza dare errori, quindi il modulo
`backend/app/engine/encoding.py` è coperto da 14 test dedicati.

## 1. Input: la posizione diventa piani 8×8

La rete riceve un tensore `(14·T + 7, 8, 8)` di `float32`, dove `T` è la lunghezza della
storia (`network.history_length`). Con `T = 8` si ottengono i canonici **119 piani** di
AlphaZero; il preset `small` usa `T = 4` (63 piani) per risparmiare memoria e tempo.

### Per ognuno dei T passi di storia: 14 piani

| Piani | Contenuto |
|---|---|
| `+0 … +5` | pezzi di **chi muove**: P, N, B, R, Q, K |
| `+6 … +11` | pezzi dell'**avversario**: P, N, B, R, Q, K |
| `+12` | 1.0 se questa posizione era già comparsa una volta |
| `+13` | 1.0 se era già comparsa due volte |

### Poi 7 piani costanti

| Piano | Contenuto |
|---|---|
| `+0` | 1.0 se chi muove è il Bianco (colore assoluto) |
| `+1` | numero di mossa / 100 |
| `+2` | chi muove può arroccare corto |
| `+3` | chi muove può arroccare lungo |
| `+4` | l'avversario può arroccare corto |
| `+5` | l'avversario può arroccare lungo |
| `+6` | contatore della regola delle 50 mosse / 100 |

### Orientamento: la rete gioca sempre "verso l'alto"

Tutto è espresso dal punto di vista di chi deve muovere. Se tocca al Nero, la scacchiera
viene specchiata verticalmente (`square ^ 56`) e i colori scambiati. In questo modo la
rete deve imparare *una* strategia invece di due speculari, dimezzando di fatto il lavoro.

```
Tocca al Bianco                Tocca al Nero (dopo l'orientamento)
    8  r n b q k b n r             8  R N B Q K B N R
    7  p p p p p p p p             7  P P P P P P P P
    …                              …
    1  R N B Q K B N R             1  r n b q k b n r
       ↑ i "miei" pezzi               ↑ i "miei" pezzi (erano i neri)
```

Il test `test_orientation_makes_mirrored_positions_identical` verifica che una posizione
e la sua speculare a colori invertiti producano piani identici.

### Perché la storia serve

Con `T = 1` la rete non può distinguere una posizione ripetuta da una nuova, né capire
"da dove viene" il pezzo appena mosso. AlphaZero usa `T = 8`. Il costo è lineare:
ogni passo aggiuntivo sono 14 piani in più in ingresso.

### Implementazione efficiente

I 14 piani di ogni semimossa vengono calcolati **una sola volta**, in frame assoluto
(Bianco in basso), e memorizzati in `PositionHistory._planes`. L'orientamento è un flip
di un asse più uno scambio di blocchi, applicato solo al momento della codifica. Così
`push()` e `pop()` restano O(1), cosa indispensabile perché l'MCTS li chiama centinaia
di volte per mossa.

## 2. Output: la policy come 4672 numeri

La testa *policy* produce `73 × 64 = 4672` logit, indicizzati come
`piano · 64 + casa_di_partenza` (nel frame orientato). I 73 piani sono:

| Indici | Tipo di mossa | Dettaglio |
|---|---|---|
| `0 … 55` | mosse "di donna" | 8 direzioni × 7 distanze |
| `56 … 63` | mosse di cavallo | 8 salti |
| `64 … 72` | sottopromozioni | 3 pezzi (N, B, R) × 3 direzioni (cattura sx, dritto, cattura dx) |

Le direzioni di donna, in ordine fisso: `N, NE, E, SE, S, SW, W, NW`.

### Casi particolari

* **Promozione a Donna**: *non* ha piani propri. È codificata come la normale mossa di
  pedone in avanti (o in diagonale). In fase di decodifica, se un pedone raggiunge la
  traversa 8 nel frame orientato, la promozione viene inferita come Donna. È esattamente
  la scelta di AlphaZero e fa risparmiare 3 piani su 73.
* **Arrocco**: è codificato come il movimento del Re di due case (`e1→g1`), cioè una
  normale mossa di donna in direzione E a distanza 2. Nessun piano dedicato.
* **En passant**: è una normale cattura diagonale del pedone.

### Perché `piano · 64 + casa` e non `casa · 73 + piano`

Perché la testa policy è una convoluzione con 73 canali di uscita: il tensore risultante
ha forma `(73, 8, 8)` e appiattirlo in ordine C dà esattamente `piano · 64 + rank·8 + file`.
Nessuna permutazione, nessuna trasposizione, nessun errore di indicizzazione.

## 3. Il ponte fra i due mondi

```python
mapping = legal_move_indices(board)      # {indice: chess.Move} — solo mosse legali
logits, value = evaluator.evaluate(planes)
priors = masked_softmax(logits, mapping.keys())   # softmax ristretto alle legali
```

`masked_softmax` normalizza **solo** sugli indici legali. I logit delle 4600+ mosse
impossibili non vengono mai letti in fase di gioco. In fase di training, invece, la loss
usa il softmax completo: i logit illegali ricevono così un gradiente che li spinge verso
il basso, il che rende la policy più pulita ma non è strettamente necessario.

## 4. Compressione per il replay buffer

Un tensore di input grezzo pesa ~16 KB. Con 200.000 posizioni sarebbero 3 GB. Ma:

* i `14·T` piani di storia sono **binari** → si impacchettano a bit (`np.packbits`);
* i 7 piani finali sono **costanti sulla scacchiera** → bastano 7 scalari.

Il risultato è ~460 byte per posizione, cioè una compressione di circa 35×, senza perdita
(test `test_pack_unpack_is_lossless`). Vedi `backend/app/engine/replay.py`.
