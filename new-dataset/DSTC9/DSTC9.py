import os
import json
import re
import scipy.stats
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
import anthropic

# Carica la chiave API dal file .env
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Inizializza il client di Anthropic
client = anthropic.Anthropic(api_key=anthropic_api_key)

# Funzione per ripulire e normalizzare l'output del modello
def estrai_json(risposta_text, dialog_id):
    try:
        # Rimuove caratteri di controllo non validi
        risposta_text = re.sub(r"[\x00-\x1F\x7F]", "", risposta_text)

        # Cerca il JSON con una regex
        match = re.search(r"{.*}", risposta_text, re.DOTALL)
        if match:
            raw_json = match.group(0)
            parsed = json.loads(raw_json)
            
            # Normalizza il campo del punteggio
            possible_keys = ["quality_score", "score", "overall_quality_score"]
            for key in possible_keys:
                if key in parsed:
                    parsed["eval_score"] = parsed.pop(key)
                    break
            
            # Controlla se il campo eval_score è stato trovato
            if "eval_score" not in parsed:
                raise KeyError("Missing eval_score in parsed response.")
            
            # Costruisci il risultato finale con solo i campi richiesti
            return {
                "eval_score": parsed["eval_score"],
                "dialog_id": dialog_id
            }
        return None
    except json.JSONDecodeError as e:
        print(f"Errore durante l'estrazione del JSON per il dialogo {dialog_id}: {e}")
        print(f"Risposta grezza: {risposta_text}")
        return None
    except Exception as e:
        print(f"Errore generico per il dialogo {dialog_id}: {e}")
        print(f"Risposta grezza: {risposta_text}")
        return None

# Funzione per valutare un intero dialogo
def valuta_dialogo(dialog, dialog_id):
    messaggi = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
Score the following dialogue generated on a continuous scale from 1 to 5.

Dialogue: {dialog}
"""
                }
            ]
        }
    ]

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            temperature=0,
            system="You are an assistant that evaluates the overall quality of dialogues. Respond only with the structured JSON output.",
            messages=messaggi,
        )

        risposta_text = response.content[0].text
        return estrai_json(risposta_text, dialog_id)
    except Exception as e:
        print(f"Errore durante l'elaborazione del dialogo {dialog_id}: {e}")
        return None

# Funzione per analizzare il dataset in modo incrementale
def analizza_dataset_incrementale(dstc9_data, output_file, error_file):
    if os.path.exists(output_file):
        with open(output_file, "r") as file:
            risultati = json.load(file)
    else:
        risultati = []

    if os.path.exists(error_file):
        with open(error_file, "r") as file:
            errori = json.load(file)
    else:
        errori = []

    dialoghi_processati = {res["dialog_id"] for res in risultati}
    dialoghi_non_processati = {err["dialog_id"] for err in errori}

    annotations = []
    evaluations = []

    for i in tqdm(range(len(dstc9_data['contexts']))):
        dialog_id = i
        if dialog_id in dialoghi_processati or dialog_id in dialoghi_non_processati:
            continue

        dialog = " ".join(dstc9_data['contexts'][i])
        annotation = dstc9_data['scores'][i]

        # Valuta il dialogo
        model_evaluation = valuta_dialogo(dialog, dialog_id)

        if model_evaluation:
            annotations.append(annotation)
            evaluations.append(model_evaluation["eval_score"])
            risultati.append({
                "dialog_id": dialog_id,
                "human_score": annotation,
                "predicted_score": model_evaluation["eval_score"]
            })
            with open(output_file, "w") as file:
                json.dump(risultati, file, indent=4)
        else:
            print(f"Errore durante la valutazione del dialogo con ID {dialog_id}.")
            errori.append({"dialog_id": dialog_id, "dialog": dialog})
            with open(error_file, "w") as file:
                json.dump(errori, file, indent=4)

    # Calcolo delle metriche globali
    spearman_corr, _ = scipy.stats.spearmanr(annotations, evaluations) if annotations and evaluations else (None, None)
    pearson_corr, _ = scipy.stats.pearsonr(annotations, evaluations) if annotations and evaluations else (None, None)
    kendall_tau, _ = scipy.stats.kendalltau(annotations, evaluations) if annotations and evaluations else (None, None)
    kappa_score = cohen_kappa_score([round(a) for a in annotations], [round(e) for e in evaluations]) if annotations and evaluations else None
    
    # Stampa le metriche
    print(f"Spearman: {spearman_corr}")
    print(f"Pearson: {pearson_corr}")
    print(f"Kendall-Tau: {kendall_tau}")
    print(f"Cohen Kappa: {kappa_score}")

    return {
        "results": risultati,
        "metrics": {
            "Spearman": spearman_corr,
            "Pearson": pearson_corr,
            "Kendall-Tau": kendall_tau,
            "Cohen Kappa": kappa_score
        }
    }

# Carica il dataset
with open("dstc9_data.json", "r") as file:
    dstc9_data = json.load(file)

# Imposta i file di output
output_file = "results_dstc9.json"
error_file = "errors_dstc9.json"

# Analizza il dataset
risultati = analizza_dataset_incrementale(dstc9_data, output_file, error_file)

# Salva le metriche
with open("metrics_dstc9.json", "w") as file:
    json.dump(risultati["metrics"], file, indent=4)
