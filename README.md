🚀 LLM-EVAL: Valutazione Multidimensionale dei Modelli AI
LLM-EVAL è un framework per la valutazione automatica dei modelli di Large Language Models (LLM) nelle conversazioni open-domain. Il progetto analizza le performance di modelli di intelligenza artificiale nella valutazione della qualità del dialogo, confrontando diverse metriche di correlazione con i giudizi umani.

📌 Obiettivi del progetto
📊 Analisi della metrica LLM-EVAL: Comprendere la metodologia per la valutazione automatica dei dialoghi.
🤖 Confronto tra modelli di IA: Valutare e confrontare i modelli GPT4o-mini, GPT4o, Claude 3.5 e Claude 3.
🔍 Analisi dataset: Testare l’impatto di dataset differenti sulla valutazione dei modelli.
🎯 Benchmark delle metriche: Utilizzare metriche avanzate (Spearman, Pearson, Kendall-Tau, Kappa di Cohen) per l’analisi dei risultati.
🔧 Setup del progetto
Per eseguire i test e le valutazioni, clona il repository e installa le dipendenze:

bash
Copia
Modifica
git clone https://github.com/TuoUsername/LLM-EVAL.git
cd LLM-EVAL
pip install -r requirements.txt
⚠️ Nota: Alcune funzionalità richiedono credenziali API per interrogare modelli LLM. Assicurati di impostare il file .env con la chiave API corretta.

📂 Struttura del Repository
data/ → Dataset utilizzati per la valutazione.
notebooks/ → Notebook Jupyter per l’analisi.
scripts/ → Codice per la valutazione automatica.
results/ → Output e report dei test.
🚀 Esegui un test
Per valutare un modello, esegui:

bash
Copia
Modifica
python scripts/evaluate.py --model claude-3.5 --dataset convai2_data.json
I risultati verranno salvati nella cartella results/.

📊 Risultati principali
📌 Modello più accurato: Claude 3.5 ha ottenuto la migliore accuratezza nella valutazione dei dialoghi.

📌 Dataset più adatto: I dataset FED e PC hanno fornito risultati più stabili rispetto a DSTC9, evidenziando l’importanza delle annotazioni nei dati.

📌 Metriche di valutazione: La Correlazione di Spearman e il Kappa di Cohen sono risultate metriche efficaci per valutare la coerenza con i giudizi umani.

📢 Contributi e Futuri Sviluppi
Siamo aperti a contributi! Prossimi sviluppi:

✅ Ottimizzazione dei prompt per ciascun modello.
✅ Estensione del framework a nuovi dataset.
✅ Implementazione di tecniche avanzate per la gestione dei bias.
Autori: Arcangeli Giovanni, Ciancio Vittorio, Di Maio Marco
📅 Data: 14/01/2025
