import json
import scipy.stats
from sklearn.metrics import cohen_kappa_score
from collections import OrderedDict

# Funzione per convertire valori continui in categorie discrete
def discretizza_score(score, num_classes=5):
    """
    Converte uno score continuo in una categoria discreta basata sul numero di classi.
    Ad esempio, con num_classes=5, score tra 1 e 5 verranno mappati a [1, 2, 3, 4, 5].
    """
    return max(1, min(num_classes, round(score)))

# Carica i risultati generati dal modello per il livello turn
with open("turn_results.json", "r") as file:
    risultati_turn = json.load(file)

# Carica i risultati generati dal modello per il livello dialogo
with open("dialogue_results.json", "r") as file:
    risultati_dialogue = json.load(file)

# Carica il dataset originale
with open("fed_data.json", "r") as file:
    dataset = json.load(file)

# Funzione per confrontare i risultati
def confronta_risultati(risultati_modello, dataset, livello="turn"):
    corretti = 0
    errati = 0
    non_confrontabili = 0
    dettagli = []
    annotations = []
    evaluations = []

    for i, risultato in enumerate(risultati_modello):
        dialog_id = risultato["dialog_id"]
        eval_score_generato = risultato["predicted_score"]

        # Recupera il punteggio umano medio
        example = dataset[dialog_id]
        if "annotations" in example and "Overall" in example["annotations"]:
            mean_annotation = sum(example["annotations"]["Overall"]) / len(example["annotations"]["Overall"])
        else:
            non_confrontabili += 1
            dettagli.append({
                "dialog_id": dialog_id,
                "eval_score_generato": eval_score_generato,
                "eval_score_atteso": None,
                "corretto": False
            })
            continue

        if discretizza_score(eval_score_generato) == discretizza_score(mean_annotation):
            corretti += 1
            dettagli.append({
                "dialog_id": dialog_id,
                "eval_score_generato": eval_score_generato,
                "eval_score_atteso": mean_annotation,
                "corretto": True
            })
            annotations.append(mean_annotation)
            evaluations.append(eval_score_generato)
        else:
            errati += 1
            dettagli.append({
                "dialog_id": dialog_id,
                "eval_score_generato": eval_score_generato,
                "eval_score_atteso": mean_annotation,
                "corretto": False
            })
            annotations.append(mean_annotation)
            evaluations.append(eval_score_generato)

    # Calcolo delle metriche globali
    spearman_corr, _ = scipy.stats.spearmanr(annotations, evaluations) if annotations and evaluations else (None, None)
    pearson_corr, _ = scipy.stats.pearsonr(annotations, evaluations) if annotations and evaluations else (None, None)
    kendall_tau, _ = scipy.stats.kendalltau(annotations, evaluations) if annotations and evaluations else (None, None)

    # Calcolo di Cohen Kappa
    try:
        annotations_discrete = [discretizza_score(x) for x in annotations]
        evaluations_discrete = [discretizza_score(x) for x in evaluations]
        kappa_score = cohen_kappa_score(annotations_discrete, evaluations_discrete) if annotations and evaluations else None
    except ValueError as e:
        print(f"Errore durante il calcolo di Cohen Kappa: {e}")
        kappa_score = None

    # Organizza l'output
    risultati_finali = OrderedDict([
        ("totale_dialoghi", len(risultati_modello)),
        ("corretti", corretti),
        ("errati", errati),
        ("non_confrontabili", non_confrontabili),
        ("accuratezza", round((corretti / len(risultati_modello)) * 100, 2) if len(risultati_modello) > 0 else 0),
        ("metriche", {
            "Spearman": spearman_corr,
            "Pearson": pearson_corr,
            "Kendall-Tau": kendall_tau,
            "Cohen Kappa": kappa_score
        }),
        ("dettagli", dettagli)
    ])
    return risultati_finali

# Confronta i risultati per il livello turn
risultati_confronto_turn = confronta_risultati(risultati_turn, dataset, livello="turn")

# Salva i risultati del confronto per il livello turn
with open("confronto_risultati_turn_fed.json", "w") as output_file:
    json.dump(risultati_confronto_turn, output_file, indent=4)

# Stampa il riassunto per il livello turn
print("Turn Level Results:")
print(f"Totale Dialoghi: {risultati_confronto_turn['totale_dialoghi']}")
print(f"Corretti: {risultati_confronto_turn['corretti']}")
print(f"Errati: {risultati_confronto_turn['errati']}")
print(f"Non Confrontabili: {risultati_confronto_turn['non_confrontabili']}")
print(f"Accuratezza: {risultati_confronto_turn['accuratezza']}%")
print("Metriche di Performance:")
print(f"  Spearman Correlation: {risultati_confronto_turn['metriche']['Spearman']}")
print(f"  Pearson Correlation: {risultati_confronto_turn['metriche']['Pearson']}")
print(f"  Kendall-Tau Correlation: {risultati_confronto_turn['metriche']['Kendall-Tau']}")
print(f"  Cohen Kappa: {risultati_confronto_turn['metriche']['Cohen Kappa']}")

# Confronta i risultati per il livello dialogo
risultati_confronto_dialogue = confronta_risultati(risultati_dialogue, dataset, livello="dialogue")

# Salva i risultati del confronto per il livello dialogo
with open("confronto_risultati_dialogue_fed.json", "w") as output_file:
    json.dump(risultati_confronto_dialogue, output_file, indent=4)

# Stampa il riassunto per il livello dialogo
print("\nDialogue Level Results:")
print(f"Totale Dialoghi: {risultati_confronto_dialogue['totale_dialoghi']}")
print(f"Corretti: {risultati_confronto_dialogue['corretti']}")
print(f"Errati: {risultati_confronto_dialogue['errati']}")
print(f"Non Confrontabili: {risultati_confronto_dialogue['non_confrontabili']}")
print(f"Accuratezza: {risultati_confronto_dialogue['accuratezza']}%")
print("Metriche di Performance:")
print(f"  Spearman Correlation: {risultati_confronto_dialogue['metriche']['Spearman']}")
print(f"  Pearson Correlation: {risultati_confronto_dialogue['metriche']['Pearson']}")
print(f"  Kendall-Tau Correlation: {risultati_confronto_dialogue['metriche']['Kendall-Tau']}")
print(f"  Cohen Kappa: {risultati_confronto_dialogue['metriche']['Cohen Kappa']}")
