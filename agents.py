import torch
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────
#  AGENT 1 – DER BEOBACHTER
#  Misst welche Pixel die Entscheidung beeinflusst haben
#  Methode: Saliency Map via Input-Gradienten
# ─────────────────────────────────────────────────

def agent_beobachter(model, image, label, device):
    """
    Berechnet für jeden der 784 Pixel wie stark er
    die finale Entscheidung beeinflusst hat.
    Gibt eine 28x28 Heatmap zurück.
    """
    model.eval()
    image = image.to(device).unsqueeze(0)
    image.requires_grad_(True)

    out  = model(image)
    pred = out.argmax(dim=1).item()

    # Gradient zur vorhergesagten Klasse
    model.zero_grad()
    out[0, pred].backward()

    # Absolute Gradienten = Wichtigkeit jedes Pixels
    saliency = image.grad.data.abs().squeeze()
    saliency = saliency.cpu().numpy().reshape(28, 28)

    # Normalisieren auf 0-1
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return {
        "saliency_map":    saliency,
        "prediction":      pred,
        "true_label":      label,
        "top_pixel_count": int((saliency > 0.5).sum()),
    }


# ─────────────────────────────────────────────────
#  AGENT 2 – DER ZWEIFLER
#  Analysiert die Unsicherheit der Entscheidung
#  Methode: Softmax Wahrscheinlichkeiten + Entropie
# ─────────────────────────────────────────────────

def agent_zweifler(model, image, device):
    """
    Schaut wie sicher/unsicher das NN war.
    Hohe Entropie = unsicher = interessanter Fall.
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        out   = model(image)
        probs = F.softmax(out, dim=1).squeeze().cpu().numpy()

    pred       = int(probs.argmax())
    confidence = float(probs.max())

    # Entropie = Maß für Unsicherheit (höher = unsicherer)
    entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

    # Top 3 Kandidaten
    top3_idx  = probs.argsort()[::-1][:3]
    top3      = [(int(i), float(probs[i])) for i in top3_idx]

    return {
        "prediction":  pred,
        "confidence":  confidence,
        "entropy":     entropy,
        "is_uncertain": entropy > 1.0,   # Schwellwert – experimentiere damit
        "top3":        top3,
        "all_probs":   probs,
    }


# ─────────────────────────────────────────────────
#  AGENT 3 – DER ERKLÄRER
#  Nutzt Claude um eine menschliche Erklärung
#  aus den Messungen der anderen Agenten zu machen
# ─────────────────────────────────────────────────

def agent_erklaerer(beobachter_result, zweifler_result):
    """
    Baut einen strukturierten Prompt für Claude
    aus den Messungen der anderen zwei Agenten.
    Claude erklärt dann in normalem Deutsch.
    """
    b = beobachter_result
    z = zweifler_result

    top3_text = ", ".join(
        [f"Ziffer {cls} ({prob*100:.1f}%)" for cls, prob in z["top3"]]
    )

    prompt = f"""Du bist ein Experte für neuronale Netze und erklärst Laien 
wie ein NN eine Entscheidung getroffen hat.

Hier sind die Messdaten:

ENTSCHEIDUNG:
- Das NN hat die Ziffer {z['prediction']} erkannt
- Echte Ziffer: {b['true_label']}
- Korrekt: {'Ja ✓' if z['prediction'] == b['true_label'] else 'Nein ✗'}

SICHERHEIT:
- Konfidenz: {z['confidence']*100:.1f}%
- Entropie (Unsicherheit): {z['entropy']:.3f} {'(das NN war unsicher!)' if z['is_uncertain'] else '(das NN war sicher)'}
- Top 3 Kandidaten: {top3_text}

PIXEL-ANALYSE:
- {b['top_pixel_count']} von 784 Pixeln haben die Entscheidung stark beeinflusst
- Das sind {b['top_pixel_count']/784*100:.1f}% der gesamten Bildfläche

Erkläre in 3-4 Sätzen auf Deutsch:
1. Wie hat das NN entschieden?
2. War es sicher oder unsicher – und was bedeutet das?
3. Was könnte der Grund für einen Fehler sein (falls falsch)?

Schreibe klar und verständlich, ohne Fachbegriffe."""

    return prompt