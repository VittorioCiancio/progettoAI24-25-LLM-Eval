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

# Funzione per analizzare il dataset incrementale
def analizza_pc_usr(dataset, output_file, error_file):
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

    for i in tqdm(range(len(dataset))):
        dialogue_data = dataset[i]
        context = dialogue_data["context"].replace("\n", " ")
        responses = dialogue_data["responses"]

        for response_data in responses:
            response = response_data["response"].split("\n")[0]
            human_score = sum(response_data["Overall"]) / len(response_data["Overall"])

            if i in dialoghi_processati:
                continue

            valutazione = valuta_turn(context, response, i)

            if valutazione:
                annotations.append(human_score)
                evaluations.append(valutazione["eval_score"])
                risultati.append({
                    "dialog_id": i,
                    "context": context,
                    "response": response,
                    "human_score": human_score,
                    "predicted_score": valutazione["eval_score"]
                })
                with open(output_file, "w") as f:
                    json.dump(risultati, f, indent=4)
            else:
                errori.append({"dialog_id": i, "context": context, "response": response})
                with open(error_file, "w") as f:
                    json.dump(errori, f, indent=4)

    # Calcolo delle metriche globali
    spearman_corr, _ = scipy.stats.spearmanr(annotations, evaluations)
    pearson_corr, _ = scipy.stats.pearsonr(annotations, evaluations)
    kendall_tau, _ = scipy.stats.kendalltau(annotations, evaluations)
    kappa_score = cohen_kappa_score(
        [round(a) for a in annotations],
        [round(e) for e in evaluations]
    ) if annotations and evaluations else None

    print(f"Spearman Correlation: {spearman_corr}")
    print(f"Pearson Correlation: {pearson_corr}")
    print(f"Kendall-Tau Correlation: {kendall_tau}")
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
with open("pc_usr_data.json", "r") as f:
    pc_usr_data = json.load(f)

# Imposta i file di output
output_file = "pc_usr_results.json"
error_file = "pc_usr_errors.json"

# Analizza il dataset
risultati = analizza_pc_usr(pc_usr_data, output_file, error_file)

# Salva le metriche
with open("pc_usr_metrics.json", "w") as f:
    json.dump(risultati["metrics"], f, indent=4)
