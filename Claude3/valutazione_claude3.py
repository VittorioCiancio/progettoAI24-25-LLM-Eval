import os
import json
import re
from dotenv import load_dotenv
import anthropic
from tqdm import tqdm

# Carica la chiave API dal file .env
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Inizializza il client di Anthropic
client = anthropic.Anthropic(api_key=anthropic_api_key)

# Funzione per ripulire e normalizzare l'output del modello
def estrai_json(risposta_text, dialog_id):
    try:
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
    except Exception as e:
        print(f"Errore durante l'estrazione del JSON: {e}")
        return None

# Funzione per valutare un intero dialogo
def valuta_dialogo(dialog, dialog_id):
    testo_completo = "\n".join([turno["text"] for turno in dialog])

    messaggi = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
Score the following dialogue generated on a continuous scale from 1 to 5.

Dialogue: {testo_completo}
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
        risultato = estrai_json(risposta_text, dialog_id)

        if risultato:
            return risultato
        print(f"Error: Invalid result. Raw response: {risposta_text}")
        return None
    except Exception as e:
        print(f"Error during processing: {e}")
        return None

# Funzione per valutare il dataset in modo incrementale
def valuta_intero_dataset(dataset, output_file, error_file):
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

    dialoghi_processati = {ris["dialog_id"] for ris in risultati}
    dialoghi_non_processati = {err["dialog_id"] for err in errori}

    for i, dialogo in enumerate(dataset):
        dialog_id = dialogo["dialog_id"]
        if dialog_id in dialoghi_processati or dialog_id in dialoghi_non_processati:
            continue

        print(f"Processando dialogo {i + 1} di {len(dataset)} con ID: {dialog_id}")

        risultato = valuta_dialogo(dialogo["dialog"], dialog_id)
        if risultato:
            risultati.append(risultato)
            with open(output_file, "w") as file:
                json.dump(risultati, file, indent=4)
        else:
            errori.append({"dialog_id": dialog_id, "dialog": dialogo["dialog"]})
            with open(error_file, "w") as file:
                json.dump(errori, file, indent=4)

    return risultati

# Carica il dataset
with open("C:/Users/marco/Desktop/Marco/Universita/Magistrale/IA/LLMprog/convai2_data.json", "r") as file:
    dataset = json.load(file)

# Valuta il dataset
output_file = "risultati_claude3.json"
error_file = "errori_claude3.json"
risultati = valuta_intero_dataset(dataset, output_file, error_file)

# Stampa messaggio finale
print(f"Risultati salvati in '{output_file}'")
print(f"Dialoghi non processati salvati in '{error_file}'")
print(f"Totale dialoghi processati: {len(risultati)}")
