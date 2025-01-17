import os
import json
import re
import scipy.stats
from sklearn.metrics import cohen_kappa_score
from dotenv import load_dotenv
from tqdm import tqdm
import anthropic

# Carica la chiave API dal file .env
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Inizializza il client di Anthropic
client = anthropic.Anthropic(api_key=anthropic_api_key)

# Funzione per estrarre JSON dalla risposta del modello
def estrai_json(risposta_text, dialog_id):
    try:
        match = re.search(r"{.*}", risposta_text, re.DOTALL)
        if match:
            raw_json = match.group(0)
            parsed = json.loads(raw_json)
            if "score" in parsed:
                parsed["eval_score"] = parsed.pop("score")
            elif "quality_score" in parsed:
                parsed["eval_score"] = parsed.pop("quality_score")
            if "eval_score" not in parsed:
                raise KeyError("eval_score mancante nella risposta.")
            return {"eval_score": parsed["eval_score"], "dialog_id": dialog_id}
    except Exception as e:
        print(f"Errore durante l'estrazione del JSON per il dialogo {dialog_id}: {e}")
        return None

# Funzione per valutare un esempio di turn-level
def valuta_turn(context, response, dialog_id):
    messaggi = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
Score the following dialogue response generated on a continuous scale from 1 to 5.

Context: {context}
Dialogue response: {response}
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
            system="You are an assistant that evaluates turn-level responses. Respond only with structured JSON output.",
            messages=messaggi,
        )
        return estrai_json(response.content[0].text, dialog_id)
    except Exception as e:
        print(f"Errore durante la valutazione del turn {dialog_id}: {e}")
        return None

# Funzione per valutare un esempio di dialogue-level
def valuta_dialogo(conversation, dialog_id):
    messaggi = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
Score the following dialogue generated on a continuous scale from 1 to 5.

Dialogue: {conversation}
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
            system="You are an assistant that evaluates dialogue-level quality. Respond only with structured JSON output.",
            messages=messaggi,
        )
        return estrai_json(response.content[0].text, dialog_id)
    except Exception as e:
        print(f"Errore durante la valutazione del dialogo {dialog_id}: {e}")
        return None

# Funzione per analizzare il dataset incrementale
def analizza_fed_data(fed_data, output_file, error_file, livello="turn"):
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            risultati = json.load(f)
    else:
        risultati = []

    if os.path.exists(error_file):
        with open(error_file, "r") as f:
            errori = json.load(f)
    else:
        errori = []

    dialoghi_processati = {res["dialog_id"] for res in risultati}
    annotations, evaluations = [], []

    for dialog_id, example in enumerate(tqdm(fed_data)):
        if dialog_id in dialoghi_processati:
            continue

        if livello == "turn" and "response" in example:
            context = example["context"].replace("\n", " ").strip()
            response = example["response"].strip()
            valutazione = valuta_turn(context, response, dialog_id)
        elif livello == "dialogue" and "response" not in example:
            conversation = example["context"].replace("\n", " ").strip()
            valutazione = valuta_dialogo(conversation, dialog_id)
        else:
            continue

        mean_annotation = sum(example["annotations"]["Overall"]) / len(example["annotations"]["Overall"])
        if valutazione:
            annotations.append(mean_annotation)
            evaluations.append(valutazione["eval_score"])
            risultati.append({"dialog_id": dialog_id, "human_score": mean_annotation, "predicted_score": valutazione["eval_score"]})
            with open(output_file, "w") as f:
                json.dump(risultati, f, indent=4)
        else:
            errori.append({"dialog_id": dialog_id, "example": example})
            with open(error_file, "w") as f:
                json.dump(errori, f, indent=4)

    # Calcolo delle metriche
    spearman_corr, _ = scipy.stats.spearmanr(annotations, evaluations)
    pearson_corr, _ = scipy.stats.pearsonr(annotations, evaluations)
    kendall_tau, _ = scipy.stats.kendalltau(annotations, evaluations)
    kappa_score = cohen_kappa_score([round(a) for a in annotations], [round(e) for e in evaluations]) if annotations and evaluations else None

    print(f"Spearman Correlation: {spearman_corr}")
    print(f"Pearson Correlation: {pearson_corr}")
    print(f"Kendall-Tau Correlation: {kendall_tau}")
    print(f"Cohen Kappa: {kappa_score}")

    return {
        "results": risultati,
        "metrics": {"Spearman": spearman_corr, "Pearson": pearson_corr, "Kendall-Tau": kendall_tau, "Cohen Kappa": kappa_score},
    }

# Carica il dataset
with open("fed_data.json", "r") as f:
    fed_data = json.load(f)

# Analizza dati turn-level
turn_results = analizza_fed_data(fed_data, "turn_results.json", "turn_errors.json", livello="turn")

# Salva le metriche
with open("turn_metrics.json", "w") as f:
    json.dump(turn_results["metrics"], f, indent=4)

# Analizza dati dialogue-level
dialogue_results = analizza_fed_data(fed_data, "dialogue_results.json", "dialogue_errors.json", livello="dialogue")

# Salva le metriche
with open("dialogue_metrics.json", "w") as f:
    json.dump(dialogue_results["metrics"], f, indent=4)
