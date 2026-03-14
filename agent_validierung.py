import torch
import torch.nn.functional as F
import numpy as np
import requests


def test_neuron_spezialisierung(model, test_data, device,
                                neuron_idx, layer="layer1", top_n=200):
    """
    Testet ob ein Neuron auf eine bestimmte Ziffer spezialisiert ist.
    Gibt zurück: bei welchen Ziffern feuert es am stärksten?
    """
    model.eval()
    aktivierungen_pro_ziffer = {i: [] for i in range(10)}

    for i in range(min(top_n, len(test_data))):
        image, label = test_data[i]
        with torch.no_grad():
            _ = model(image.to(device).unsqueeze(0))

        if layer == "layer1":
            akt = model.act1.squeeze().cpu().numpy()[neuron_idx]
        else:
            akt = model.act2.squeeze().cpu().numpy()[neuron_idx]

        aktivierungen_pro_ziffer[label].append(float(akt))

    # Durchschnitt pro Ziffer
    durchschnitt = {
        ziffer: float(np.mean(werte)) if werte else 0.0
        for ziffer, werte in aktivierungen_pro_ziffer.items()
    }

    # Welche Ziffer aktiviert das Neuron am stärksten?
    top_ziffer = sorted(durchschnitt.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "neuron":      neuron_idx,
        "layer":       layer,
        "top_ziffern": top_ziffer,
        "alle":        durchschnitt,
    }


def agent_validierung(model, test_data, hypothesen, profile, api_key, device):
    """
    Validierungs-Agent: Testet die Hypothesen des Synthese-Agenten
    automatisch und gibt ein Urteil zurück.
    """
    u = profile["uncertainty"]
    c = profile["causal"]

    # Teste die top kausalen Neuronen auf Spezialisierung
    test_ergebnisse = []
    for neuron_info in c["top_causal"][:3]:
        neuron_idx = neuron_info["neuron"]
        ergebnis = test_neuron_spezialisierung(
            model, test_data, device, neuron_idx, layer="layer1"
        )
        test_ergebnisse.append(ergebnis)

    # Test-Ergebnisse als Text
    test_text = ""
    for e in test_ergebnisse:
        ziffern_text = ", ".join([
            f"Ziffer {z}: {a:.3f}" for z, a in e["top_ziffern"]
        ])
        test_text += f"Neuron {e['neuron']} (Layer 1) feuert am stärksten bei: {ziffern_text}\n"

    hypothesen_text = "\n".join(hypothesen)

    prompt = f"""Du bist ein KI-Forscher der Hypothesen über neuronale Netze validiert.

HYPOTHESEN:
{hypothesen_text}

TESTERGEBNISSE (gemessen über 200 Beispiele):
{test_text}

KONTEXT:
- Das NN hat Ziffer {u['prediction']} vorhergesagt
- Konfidenz: {u['confidence']*100:.1f}%

Bewerte jede Hypothese:
- BESTÄTIGT: wenn die Daten die Hypothese stützen
- WIDERLEGT: wenn die Daten dagegen sprechen
- UNKLAR: wenn die Daten nicht ausreichen

Format: "Hypothese X: [BESTÄTIGT/WIDERLEGT/UNKLAR] – [kurze Begründung]"
Nur die Bewertungen, keine Einleitung."""

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
        headers={"content-type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    urteil = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return {
        "urteil":          urteil,
        "test_ergebnisse": test_ergebnisse,
    }