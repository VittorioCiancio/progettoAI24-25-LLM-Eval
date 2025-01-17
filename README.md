 **Valutazione Automatica di Modelli IA per Dialoghi Uomo-Macchina**

**Descrizione del Progetto:**

*   Questo progetto si concentra sulla valutazione automatica della qualità dei dialoghi generati da chatbot, utilizzando il framework **LLM-EVAL**. L'obiettivo principale è analizzare l'efficacia di diversi modelli di IA nell'assegnare punteggi di qualità ai dialoghi tra bot e umani e l'impatto di diversi dataset sulle prestazioni del modello. Il progetto è diviso in due fasi principali:
    *   **Fase 1**: Valutazione di quattro modelli di IA (GPT4o-mini, GPT4o, Claude 3.5 e Claude 3) su un singolo dataset (convai2_data.json).
    *   **Fase 2**: Valutazione dell'impatto di diversi dataset (PC, TC, DSTC9 e FED) sulle prestazioni del modello Claude 3.

**Contesto del Progetto:**

*   L'intelligenza artificiale sta trasformando il modo in cui interagiamo con la tecnologia, in particolare attraverso i chatbot. È essenziale disporre di metodi affidabili per valutare le prestazioni di questi sistemi. Questo progetto affronta questa sfida implementando e analizzando il framework LLM-EVAL per la valutazione automatica dei dialoghi.

**Obiettivi:**

*   Comprendere e implementare la metrica LLM-EVAL.
*   Valutare l'efficacia di LLM-EVAL nel misurare la qualità delle risposte generate rispetto ai giudizi umani.
*   Confrontare le prestazioni di diversi modelli di IA nella valutazione di dialoghi.
*   Analizzare l'impatto di diversi dataset sulle prestazioni di un modello di IA.
*   Valutare l'efficacia di un prompt specifico per la valutazione della qualità dei dialoghi.

**Struttura della Relazione (Documentazione.pdf):**

*   **Capitolo 1**: Introduzione al contesto e agli obiettivi del progetto.
*   **Capitolo 2**: Background teorico, inclusa una descrizione di LLM-EVAL.
*   **Capitolo 3**: Descrizione della Fase 1, con la valutazione dei modelli di IA.
*   **Capitolo 4**: Descrizione della Fase 2, con la valutazione dei dataset.
*   **Capitolo 5**: Conclusioni e riepilogo dei risultati.

**Fase 1: Valutazione dei Modelli di IA**

*   **Dataset**: convai2_data.json, contenente dialoghi con punteggi di qualità (score_eval).
*   **Modelli di IA**: GPT4o-mini, GPT4o, Claude 3.5, Claude 3.
*   **Metodologia di Valutazione**: Utilizzo di metriche come accuratezza, Kappa di Cohen, correlazione di Spearman, correlazione di Pearson e correlazione di Kendall-Tau.
*   **Prompt Utilizzato**: "Score the following dialogue generated on a continuous scale from 1 to 5. Dialogue: {dialogue}".
*   **Risultati Principali**:
    *   Claude 3.5 ha ottenuto la migliore accuratezza.
    *   GPT4o-mini ha mostrato la migliore correlazione con i giudizi umani.
    *   L'accuratezza complessiva dei modelli è risultata bassa.

**Fase 2: Valutazione dei Dataset**

*   **Dataset**: PC, TC, DSTC9 e FED.
*   **Modello di IA**: Claude 3.
*   **Metodologia di Valutazione**: Utilizzo degli stessi metriche della Fase 1.
*   **Prompt Utilizzati**:
    *   Per dataset turn-level: "Score the following dialogue response generated on a continuous scale from 1 to 5. Context: {context} Dialogue response: {response}".
    *   Per dataset dialogue-level: "Score the following dialogue generated on a continuous scale from 1 to 5. Dialogue: {dialog}".
*   **Risultati Principali**:
    *   Claude 3 ha mostrato prestazioni migliori su dataset con dialoghi task-oriented e annotazioni dettagliate (FED, PC, TC).
    *   Le prestazioni su DSTC9, un dataset open-domain, sono state basse.

**Risultati Principali del Progetto:**

*   **Claude 3.5** è il modello più promettente per la valutazione automatica dei dialoghi, grazie alla sua accuratezza superiore.
*   La scelta del dataset influenza significativamente le prestazioni del modello. I modelli di IA tendono a ottenere risultati migliori su dataset con dialoghi task-oriented e annotazioni dettagliate.
*   La valutazione automatica dei dialoghi è un compito complesso che richiede ulteriori ricerche per migliorare l'accuratezza e l'affidabilità dei modelli.

**Come Usare il Codice:**

*   **Linguaggio di Programmazione:** Python.
*   **Librerie**: `os`, `json`, `re`, `dotenv`, `anthropic`, `tqdm`.
*   **Installazione Dipendenze:** Assicurati di avere le librerie necessarie (es. `pip install -r requirements.txt`).
*   **Configurazione:** Crea un file `.env` con la chiave API di Anthropic.
*   **Esecuzione:** Esegui gli script Python per valutare i modelli e i dataset.

**Autori:**

*   Arcangeli Giovanni, Ciancio Vittorio, Di Maio Marco.
