import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# Carica la chiave API
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Funzione per ripulire e normalizzare l'output del modello
def estrai_json(risposta_text, dialog_id):
    try:
        # Tenta di trovare un JSON valido nella risposta
        match = re.search(r"{.*}", risposta_text, re.DOTALL)
        if match:
            raw_json = match.group(0)
            parsed = json.loads(raw_json)
            if "eval_score" in parsed:
                return {
                    "eval_score": parsed["eval_score"],
                    "dialog_id": dialog_id
                }
        
        # Estrattore manuale per risposte non JSON (include decimali e varianti)
        match = re.search(
            r"(?:score(?:s|d)?(?:.*)?|rating|rate(?:s|d)?(?:.*)? as|this dialogue scores(?:.*)?|"
            r"I would rate(?:.*)?|this dialogue would be rated(?:.*)?|I'd rate(?:.*)?)\s*(\d+(\.\d+)?)(?:\s*out of\s*\d+)?", 
            risposta_text, 
            re.IGNORECASE
        )
        if match:
            eval_score = round(float(match.group(1)))  # Converte e arrotonda a intero
            return {
                "eval_score": eval_score,
                "dialog_id": dialog_id
            }
        
        raise ValueError("Punteggio non trovato nella risposta.")
    except Exception as e:
        print(f"Errore durante l'estrazione del JSON per il dialogo {dialog_id}: {e}")
        print(f"Risposta grezza: {risposta_text}")
        return None

# Funzione per valutare un intero dialogo
def valuta_dialogo(dialog, dialog_id):
    testo_completo = "\n".join([f'{turno["sender"]}: "{turno["text"]}"' for turno in dialog])

    prompt = f"""
Score the following dialogue generated on a continuous scale from 1 to 5.

Dialogue:
{testo_completo}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt.strip()}],
            timeout=30
        )
        risposta_text = response.choices[0].message.content.strip()
        return estrai_json(risposta_text, dialog_id)
    except Exception as e:
        print(f"Errore durante l'elaborazione del dialogo {dialog_id}: {e}")
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
            print(f"Errore: Il dialogo con ID {dialog_id} non è stato processato correttamente.")
            errori.append({"dialog_id": dialog_id, "dialog": dialogo["dialog"]})
            with open(error_file, "w") as file:
                json.dump(errori, file, indent=4)

    return risultati

# Carica il dataset
with open("C:/Users/marco/Desktop/Marco/Universita/Magistrale/IA/LLMprog/convai2_data.json", "r") as file:
    dataset = json.load(file)

# Valuta il dataset
output_file = "risultati_chatgpt4o.json"
error_file = "errori_chatgpt4o.json"
risultati = valuta_intero_dataset(dataset, output_file, error_file)

# Messaggio finale
print(f"Risultati salvati in '{output_file}'")
print(f"Dialoghi non processati salvati in '{error_file}'")
print(f"Totale dialoghi processati: {len(risultati)}")
