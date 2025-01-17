import json
import scipy.stats
from sklearn.metrics import cohen_kappa_score
from collections import OrderedDict

# Carica i risultati generati da Claude 3
with open("risultati_claude3.json", "r") as file:
    risultati_modello = json.load(file)

# Carica il dataset originale
with open("C:/Users/marco/Desktop/Marco/Universita/Magistrale/IA/LLMprog/convai2_data.json", "r") as file:
    dataset = json.load(file)

# Funzione per confrontare i risultati
def confronta_risultati(risultati_modello, dataset):
    corretti = 0
    errati = 0
    non_confrontabili = 0
    dettagli = []
    annotations = []
    evaluations = []

    # Creazione di un dizionario per mappare dialog_id al eval_score del dataset
    eval_score_dataset = {dialog["dialog_id"]: dialog.get("eval_score") for dialog in dataset}

    for risultato in risultati_modello:
        dialog_id = risultato["dialog_id"]
        eval_score_generato = risultato["eval_score"]
        eval_score_atteso = eval_score_dataset.get(dialog_id)

        if eval_score_atteso is None:  # Caso in cui il dialogo non ha un eval_score
            non_confrontabili += 1
            dettagli.append({
                "dialog_id": dialog_id,
                "eval_score_generato": eval_score_generato,
                "eval_score_atteso": None,
                "corretto": False
            })
        elif eval_score_generato == eval_score_atteso:  # Caso in cui il dialogo ha un eval_score
            corretti += 1
            dettagli.append({
                "dialog_id": dialog_id,
                "eval_score_generato": eval_score_generato,
                "eval_score_atteso": eval_score_atteso,
                "corretto": True
            })
            annotations.append(int(eval_score_atteso))  # Conversione in interi
            evaluations.append(int(eval_score_generato))
        else:
            errati += 1
            dettagli.append({
                "dialog_id": dialog_id,
                "eval_score_generato": eval_score_generato,
                "eval_score_atteso": eval_score_atteso,
                "corretto": False
            })
            annotations.append(int(eval_score_atteso))  
            evaluations.append(int(eval_score_generato))

    # Calcola le metriche di performance
    spearman_corr, _ = scipy.stats.spearmanr(annotations, evaluations) if annotations and evaluations else (None, None)
    pearson_corr, _ = scipy.stats.pearsonr(annotations, evaluations) if annotations and evaluations else (None, None)
    kendall_tau, _ = scipy.stats.kendalltau(annotations, evaluations) if annotations and evaluations else (None, None)

    # Calcola Cohen Kappa solo se i valori sono validi
    try:
        kappa_score = cohen_kappa_score(annotations, evaluations) if annotations and evaluations else None
    except ValueError as e:
        print(f"Errore durante il calcolo di Cohen Kappa: {e}")
        kappa_score = None

    # Organizza l'output
    risultati_finali = OrderedDict([
        ("totale_dialoghi", len(risultati_modello)),
        ("corretti", corretti),
        ("errati", errati),
        ("non_confrontabili", non_confrontabili),
        ("accuratezza", round((corretti / len(risultati_modello)) * 100, 2)),
        ("metriche", {
            "Spearman": spearman_corr,
            "Pearson": pearson_corr,
            "Kendall-Tau": kendall_tau,
            "Cohen Kappa": kappa_score
        }),
        ("dettagli", dettagli)
    ])
    return risultati_finali

# Confronta i risultati
risultati_confronto = confronta_risultati(risultati_modello, dataset)

# Salva i risultati del confronto in un file
with open("confronto_risultati_claude3.json", "w") as output_file:
    json.dump(risultati_confronto, output_file, indent=4)

# Stampa il riassunto
print(f"Totale Dialoghi: {risultati_confronto['totale_dialoghi']}")
print(f"Corretti: {risultati_confronto['corretti']}")
print(f"Errati: {risultati_confronto['errati']}")
print(f"Non Confrontabili: {risultati_confronto['non_confrontabili']}")
print(f"Accuratezza: {risultati_confronto['accuratezza']}%")
print("Metriche di Performance:")
print(f"  Spearman Correlation: {risultati_confronto['metriche']['Spearman']}")
print(f"  Pearson Correlation: {risultati_confronto['metriche']['Pearson']}")
print(f"  Kendall-Tau Correlation: {risultati_confronto['metriche']['Kendall-Tau']}")
print(f"  Cohen Kappa: {risultati_confronto['metriche']['Cohen Kappa']}")
