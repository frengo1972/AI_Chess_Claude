# La ricerca: MCTS con PUCT

File: `backend/app/engine/mcts.py`

Una rete da sola sceglie la mossa che "sembra" migliore. Una rete più una ricerca sceglie
la mossa che *risulta* migliore dopo aver guardato avanti. La differenza in forza è di
diverse centinaia di punti Elo, a parità di rete.

## Che cos'è un albero MCTS qui

Non è il Monte Carlo classico: **non ci sono partite casuali di rollout**. Ogni foglia è
valutata dalla testa *value* della rete. È il contributo centrale di AlphaZero.

Ogni nodo memorizza:

| Campo | Significato |
|---|---|
| `prior` (P) | probabilità data dalla policy della rete |
| `visit_count` (N) | quante simulazioni sono passate di qui |
| `value_sum` (W) | somma dei valori riportati indietro |
| `Q = W / N` | valore medio, dal punto di vista di chi muove **in quel nodo** |

## Le quattro fasi di una simulazione

```
        SELEZIONE                ESPANSIONE         VALUTAZIONE        BACKUP
   scendi scegliendo il      genera i figli       la rete stima     risali sommando
   figlio con PUCT più alto  (mosse legali)       il valore         il valore, con
                                                                    segno alternato
        radice                     ○                   ○                 ○ +0.3
          │                        │                   │                 │
          ●                        ●                   ●                 ● -0.3
          │                        │                   │                 │
          ●                        ● ─┬─┬─┬─           ●                 ● +0.3
                                      ○ ○ ○         valore = +0.3
```

L'alternanza di segno nel backup è ciò che rende l'albero *negamax*: un valore di `+0.3`
per chi muove alla foglia vale `-0.3` per il suo avversario un livello sopra.

## La formula PUCT

A ogni nodo si sceglie il figlio che massimizza, **dal punto di vista del genitore**:

```
                                            √( Σ N_fratelli )
    score(figlio)  =  −Q(figlio)  +  c_puct · P(figlio) · ─────────────────
                                                            1 + N(figlio)
```

* `−Q(figlio)` è lo **sfruttamento**: il valore del figlio visto dal genitore (segno
  invertito perché Q è espresso dal lato di chi muove nel figlio).
* Il secondo termine è l'**esplorazione**: alto per mosse con prior alto e poche visite,
  decade come `1/N` man mano che il ramo viene esplorato.
* `c_puct` bilancia i due. Con `c_puct_base > 0` si usa la versione di AlphaZero che lo
  fa crescere lentamente con il numero di visite:
  `c = log((N + c_base + 1)/c_base) + c_init`.

### First Play Urgency

Un figlio mai visitato non ha `Q`. Assumere `Q = 0` renderebbe le mosse inesplorate
troppo attraenti in posizioni perdenti e troppo poco in quelle vincenti. Si usa invece
il valore del genitore meno un margine di pessimismo:
`Q_stimato = Q(genitore) − fpu_reduction`.

## Batch e virtual loss

Valutare una posizione alla volta lascia la GPU quasi ferma. La ricerca raccoglie perciò
fino a `max_batch_size` foglie prima di chiamare la rete. Ma senza correttivi tutti i
percorsi scenderebbero sullo stesso ramo, il più promettente.

La **virtual loss** risolve il problema: mentre una foglia è "in volo", i nodi sul suo
percorso contano una sconfitta fittizia per il genitore, il che li rende meno attraenti
per i percorsi successivi dello stesso batch. Quando arriva il valore vero, la virtual
loss viene rimossa.

Nel codice questo si traduce in una convenzione di segno precisa: la virtual loss è una
**vittoria per chi muove nel nodo**, cioè una sconfitta per il genitore che sta scegliendo.

```python
@property
def q(self) -> float:
    visits = self.visit_count + self.virtual_loss
    return (self.value_sum + self.virtual_loss) / visits if visits else 0.0
```

## Nodi terminali

Se durante la discesa si raggiunge scacco matto, stallo, ripetizione o 50 mosse, il valore
riportato indietro è quello **esatto** (`±1` o `0`), non la stima della rete. È il motivo
per cui, con qualche centinaio di simulazioni, anche una rete con pesi casuali trova un
matto in una: il test `test_finds_mate_in_one` lo verifica.

## Dalla ricerca alla mossa

Finito il budget di simulazioni, la distribuzione delle visite alla radice è la **policy
migliorata**:

```
π(a) = N(a) / Σ N
```

* In **self-play**, per le prime `temperature_moves` semimosse la mossa viene *campionata*
  da `π^(1/τ)` con `τ = 1`: serve varietà, altrimenti tutte le partite sarebbero identiche.
  Dopo, si gioca la mossa più visitata.
* Contro un umano o in arena, `τ = 0`: sempre la mossa più visitata, gioco deterministico.
* `π` è anche il bersaglio di addestramento della testa policy.

## Rumore di Dirichlet alla radice

Solo in self-play, i prior della radice vengono mescolati con rumore di Dirichlet:

```
P(a) ← (1 − ε) · P(a) + ε · η(a),      η ~ Dir(α),  α = 0.3,  ε = 0.25
```

Garantisce che ogni mossa alla radice abbia una probabilità non nulla di essere provata,
anche quando la rete è convinta del contrario. Senza, l'addestramento collassa su un
repertorio ristretto e non ne esce più. **Non viene mai applicato quando la rete gioca
contro di te**: `nn_engine.py` imposta esplicitamente `dirichlet_epsilon = 0`.

## Il caso `simulations = 1`

La ricerca espande la radice, non visita nessun figlio e restituisce direttamente i
prior della rete come `policy_target`, con `used_search = False`. È la modalità "policy
pura": una sola forward pass per mossa. Il preset `policy-only` la usa per generare
partite ~50-100× più in fretta, al prezzo di un tetto di forza molto più basso.

## Costo

Il tempo per mossa è dominato da `simulations` forward pass, meno i colpi di cache.
Indicativamente, con il preset `small` su CPU: ~40-80 ms con 96 simulazioni. La ricerca
non muta mai lo stato passato (`test_search_never_mutates_the_state`), quindi lo stesso
oggetto può essere riusato per tutta la partita.
