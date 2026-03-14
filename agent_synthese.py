import requests
from agent_kausal import format_kausal


def agent_synthese(profile, api_key):
    """
    Synthese-Agent: Bekommt das vollständige Messprofil
    und formuliert konkrete Hypothesen über das NN.

    Gibt strukturierte Hypothesen zurück die der
    Validierungs-Agent dann testen kann.
    """
    u = profile["uncertainty"]
    s = profile["saliency"]
    c = profile["causal"]
    t = profile["top_neurons"]

    # Top Neuronen als Text
    top_l1 = ", ".join([f"Neuron {n['neuron']} ({n['activation']:.2f})"
                        for n in t["layer1"]])
    top_l2 = ", ".join([f"Neuron {n['neuron']} ({n['activation']:.2f})"
                        for n in t["layer2"]])

    kausal_text = format_kausal({"layer1": c, "layer2": c})

    prompt = f"""Du bist ein Forscher der neuronale Netze analysiert.
Dir liegen folgende Messdaten vor:

ENTSCHEIDUNG:
- Vorhergesagte Ziffer: {u['prediction']}
- Echte Ziffer: {s['true_label']}
- Korrekt: {'Ja' if u['prediction'] == s['true_label'] else 'Nein'}
- Konfidenz: {u['confidence']*100:.1f}%
- Unsicherheit (Entropie): {u['entropy']:.4f}
- Wichtige Pixel: {s['top_pixel_count']}/784

AKTIVE NEURONEN:
- Layer 1 (top 5): {top_l1}
- Layer 2 (top 5): {top_l2}

KAUSAL-ANALYSE:
{kausal_text}

Formuliere 2-3 konkrete, testbare Hypothesen über dieses NN.
Beispiel: "Hypothese: Neuron 17 in Layer 1 ist spezialisiert auf horizontale Linien,
weil es bei der Ziffer 7 stark feuert und kausal wichtig ist."

Antworte NUR mit den Hypothesen, eine pro Zeile, beginnend mit "Hypothese X:".
Keine Einleitung, kein Abschluss."""

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
        headers={"content-type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    # Hypothesen parsen
    hypothesen = [
        line.strip()
        for line in text.strip().split("\n")
        if line.strip().startswith("Hypothese")
    ]

    return {
        "raw":       text,
        "hypothesen": hypothesen,
        "prediction": u["prediction"],
        "true_label": s["true_label"],
    }