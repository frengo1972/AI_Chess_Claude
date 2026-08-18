# I KPI della pagina Training

Tutti i valori sono scritti dal trainer in SQLite (`backend/data/training.db`) e letti
dall'API. La dashboard si aggiorna via WebSocket ogni 1,5 secondi.

## Indicatori di avanzamento

| KPI | Significato | Cosa vuoi vedere |
|---|---|---|
| **Iterazione** | cicli self-play → training → arena completati | cresce |
| **Partite giocate** | totale cumulativo di partite di self-play | cresce; è il vero "carburante" dell'apprendimento |
| **Posizioni** | totale di esempi di training generati | ~40-120 per partita |
| **Replay buffer** | posizioni attualmente in memoria | si stabilizza sulla capacità configurata |
| **Tempo di calcolo** | ore di addestramento accumulate, sommate su tutte le sessioni | cresce anche dopo un'interruzione: il run riprende, non riparte |

## Indicatori di forza

| KPI | Significato | Cosa vuoi vedere |
|---|---|---|
| **Elo (relativo)** | somma dei ΔElo delle promozioni, partendo da 0 | curva crescente. È relativo alla rete iniziale, non è un Elo FIDE |
| **Arena score** | score del candidato contro il campione (0.5 = pari) | ≥ 0.55 abbastanza spesso. Sempre 0.5 = l'apprendimento è fermo |
| **Promosso** | il candidato ha superato il gate | qualche promozione ogni poche iterazioni |
| **Elo vs Stockfish** | stima *assoluta* dal benchmark contro Stockfish a Elo noto | è l'unico numero confrontabile con il mondo reale |

Il ΔElo dell'arena si ricava da `ΔElo = −400·log₁₀(1/score − 1)`. La dashboard mostra
anche un intervallo di confidenza al 95%: con 20 partite l'errore è di ±150 Elo circa, per
cui **le singole misure vanno lette come tendenza, non come verità**.

## Indicatori di apprendimento

| KPI | Significato | Cosa vuoi vedere |
|---|---|---|
| **Policy loss** | cross-entropy fra rete e visite MCTS | scende, poi si stabilizza. Parte da ~ln(35) ≈ 3,5-6,5 |
| **Value loss** | MSE fra valore previsto e bersaglio value | scende da ~1,0 verso 0,6-0,8. Sotto 0,3 con poche partite = overfitting. Con `value_search_weight > 0` parte più bassa: il bersaglio è meno rumoroso |
| **Scarto ricerca / risultato** | media di \|z − q\|: quanto la ricerca si sbagliava sull'esito | scende. È la misura di quanto la rete sta imparando a *valutare*, non solo a giocare |
| **Learning rate** | passo dell'ottimizzatore | scalini in corrispondenza delle milestone |
| **Entropia della policy** | quanto è "indecisa" la ricerca | deve scendere lentamente. Un crollo rapido = collasso, la rete gioca sempre le stesse mosse |

## Indicatori sul gioco prodotto

| KPI | Significato | Cosa vuoi vedere |
|---|---|---|
| **Lunghezza media** | semimosse per partita | cresce all'inizio (impara a non farsi mattare), poi si stabilizza |
| **Tasso di patte** | quota di `1/2-1/2` | 20-60% è sano. 100% = adjudication troppo aggressiva o rete incapace di vincere |
| **Vittorie Bianco / Nero** | bilanciamento | leggero vantaggio al Bianco atteso. Uno sbilanciamento forte segnala un bug di orientamento |
| **Rese** | quota di partite chiuse per resa automatica | 20-50%. Se ~0, la soglia è troppo severa e stai sprecando tempo |
| **Valore medio alla radice** | quanto la rete si crede in vantaggio | vicino a 0 in media: gioca contro sé stessa, quindi in media pareggia |

## Indicatori di costo

| KPI | Significato |
|---|---|
| **Partite/minuto** | throughput del self-play. È il collo di bottiglia principale |
| **Posizioni/secondo** | derivato dal precedente |
| **Secondi self-play / training / arena** | dove sta andando il tempo. In genere il self-play domina (70-90%) |
| **Cache hit rate** | quota di valutazioni servite dalla cache dell'evaluator |

## Indicatori sulla rete

| KPI | Significato |
|---|---|
| **Parametri** | numero di pesi. Non è un indicatore di forza, è un indicatore di *costo* |
| **Dimensione (MB)** | peso del checkpoint |
| **Blocchi × filtri** | forma della torre residuale |
| **Piani di input** | `14·T + 7` |
| **Simulazioni/mossa** | budget MCTS. È il moltiplicatore di forza più diretto a parità di rete |

## Come leggere l'insieme

Una run sana, nell'ordine:

1. le prime iterazioni saltano il training (buffer sotto la soglia): è normale;
2. `value_loss` scende in fretta — la rete impara il valore del materiale;
3. `avg_plies` cresce — smette di farsi mattare in apertura;
4. `arena_score` comincia a superare 0.55 a intermittenza → l'Elo relativo sale a scalini;
5. `policy_entropy` scende lentamente — la rete sviluppa preferenze;
6. il benchmark contro Stockfish inizia a raccogliere qualche patta, poi qualche vittoria.

Il segnale d'allarme più utile è la combinazione **`arena_score` fermo a 0.5 +
`policy_entropy` in caduta**: significa che la rete si sta convincendo di sé stessa senza
migliorare. Le contromisure sono più esplorazione (`dirichlet_epsilon`,
`temperature_moves`) o più dati per iterazione.
