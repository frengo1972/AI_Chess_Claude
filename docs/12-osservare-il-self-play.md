# Osservare il self-play

File: `backend/app/engine/watch.py`, `frontend/src/components/SelfPlayWatch.tsx`

Durante l'addestramento la rete gioca migliaia di partite che normalmente nessuno
vede: finiscono nel replay buffer come tensori e nei KPI come medie. Il pannello
**Self-play in diretta** della pagina Apprendimento apre una finestra su quelle
partite mentre vengono giocate, una scacchiera per worker.

Serve a due cose molto concrete: capire *cosa* sta imparando la rete (una media di
1,4 pezzi di vantaggio non dice se sta arroccando o solo spingendo pedoni), e
accorgersi subito delle patologie tipiche delle prime iterazioni — partite che
finiscono sempre per limite di semimosse, regine buttate via alla terza mossa,
pezzi che oscillano avanti e indietro.

## Come arriva la partita al browser

Il self-play gira in processi worker generati dal trainer, che a sua volta è un
processo separato dal server web: non c'è memoria condivisa da guardare. I worker
pubblicano quindi come pubblica il trainer i suoi KPI, attraverso il filesystem.

```
 worker 0 ─┐
 worker 1 ─┼─► checkpoints/<run>/watch/slot-NN.json ─► API ─► WebSocket ─► scacchiere
 worker N ─┘                    ▲
                                └── settings.json (lo leggono i worker, lo scrive l'API)
```

Ogni worker possiede **uno slot** e lo riscrive dopo ogni mossa. Il file viene
scritto in una copia temporanea e spostato con `os.replace`, che è atomico anche su
Windows: chi legge vede lo scatto precedente o quello successivo, mai metà di uno.

Il flusso è a senso unico. Niente di ciò che viene pubblicato rientra
nell'addestramento, e un test lo verifica giocando due volte la stessa partita con
lo stesso seme, con e senza spettatore, e confrontando i campioni prodotti.

## Le due velocità

Sono indipendenti, ed è il punto del pannello:

| Controllo | Su cosa agisce | Costo |
|---|---|---|
| **Riproduzione** | solo su ciò che vedi | nessuno |
| **Pausa del trainer** | i worker si fermano fra una mossa e l'altra | rallenta l'addestramento in proporzione |

Ogni slot non pubblica solo la posizione attuale ma una **finestra scorrevole**
delle ultime 20 posizioni. Il browser le percorre al ritmo scelto (da mezza mossa
al secondo a "istantaneo") mentre il trainer continua a piena velocità: si ottengono
partite leggibili senza pagare nulla in termini di throughput. Il contatore `−N`
accanto al numero di mossa dice quante semimosse di ritardo si stanno accumulando; se
il ritardo supera la finestra la scacchiera salta avanti, invece di desincronizzarsi
per sempre.

La **pausa del trainer** serve quando si vuole seguire davvero una partita — a 500 ms
per mossa una partita da 120 semimosse dura un minuto — o quando la si mostra a
qualcuno. Con 8 worker e 500 ms di pausa il self-play diventa parecchie volte più
lento: è un ausilio all'osservazione, non un parametro di addestramento.

## Attivarla

È spenta per default: pubblicare costa una piccola scrittura per semimossa, e non ha
senso pagarla quando nessuno guarda.

* **Dal pannello**: l'interruttore *Pubblica le partite* agisce sul run già in corso.
  I worker rileggono `settings.json` un paio di volte al secondo, quindi ha effetto
  entro mezzo secondo senza riavviare niente. Se il pannello è aperto con
  l'interruttore attivo, anche i run avviati da quella pagina partono già osservabili.
* **Da configurazione**: `selfplay.watch_enabled` e `selfplay.watch_move_delay_ms`,
  utili da riga di comando o in un preset personalizzato.
* **Da API**: `POST /api/training/watch/settings` — vedi il
  [riferimento HTTP](09-api.md).

## Cosa mostra ogni scacchiera

* la posizione, con l'ultima mossa evidenziata;
* **worker** e **iterazione · partita**: due scacchiere non mostrano mai la stessa
  partita, perché ogni worker gioca la propria;
* la **valutazione della rete** per quella posizione, dal punto di vista del Bianco.
  È il valore della radice della ricerca, cioè l'opinione della rete stessa: non ha
  niente a che vedere con la barra di valutazione della pagina Gioca, che è Stockfish
  e serve solo all'essere umano ([isolamento](08-isolamento-stockfish.md));
* il **bilancio materiale**, come promemoria del fatto che alla rete nessuno ha detto
  quanto vale un alfiere;
* a partita finita, risultato e motivo della fine (`abbandono` compare quando scatta
  la soglia di resa).

Le scacchiere si attenuano quando il loro slot non viene aggiornato da qualche
secondo: succede fra un'iterazione e l'altra, quando il trainer è nella fase di
`train`, `arena` o `benchmark` e nessun worker sta giocando.
