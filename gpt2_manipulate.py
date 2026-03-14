import torch
import numpy as np
import requests
from gpt2_dishonesty import load_model, get_next_token
from gpt2_causal import find_dishonesty_circuit, patch_neuron

DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"
GEMINI_API_KEY = "AIzaSyDVgp_UUOlofMVv8TFQ65QR8oZkpi3MAlM"


def gemini(prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"content-type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def patch_multiple_neurons(model, tokenizer, prompt,
                            patches, mode="zero"):
    """
    Patcht mehrere Neuronen gleichzeitig.
    patches: liste von (layer, neuron, value)
    mode: "zero" → auf 0 setzen
          "value" → auf gegebenen Wert setzen
          "negate" → Vorzeichen umkehren
    """
    handles = []

    for layer_idx, neuron_idx, value in patches:
        def make_hook(l, n, v, m):
            def hook(module, input, output):
                out = output.clone()
                if m == "zero":
                    out[0, -1, n] = 0.0
                elif m == "value":
                    out[0, -1, n] = v
                elif m == "negate":
                    out[0, -1, n] = -out[0, -1, n]
                return out
            return hook

        h = model.transformer.h[layer_idx].mlp.act\
            .register_forward_hook(make_hook(layer_idx,
                                             neuron_idx,
                                             value, mode))
        handles.append(h)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)

    for h in handles:
        h.remove()

    probs = torch.softmax(out.logits[0, -1, :], dim=-1)
    top5  = torch.topk(probs, 5)
    return [{"token": tokenizer.decode([tid.item()]),
             "prob":  float(p)}
            for p, tid in zip(top5.values, top5.indices)]


def run_manipulation():
    model, tokenizer = load_model()

    true_prompt  = "The Eiffel Tower is located in"
    false_prompt = ("The Eiffel Tower is located in Berlin. "
                    "The Eiffel Tower is located in")

    print(f"\n🔬 Manipulations-Experiment")
    print(f"{'='*55}")

    # Basis
    base = get_next_token(model, tokenizer, false_prompt)
    print(f"\n  Basis (mit Lüge): '{base[0]['token']}' "
          f"({base[0]['prob']*100:.1f}%)")

    # Schaltkreis laden
    print(f"\n📊 Lade Lügen-Schaltkreis...")
    circuit = find_dishonesty_circuit(
        model, tokenizer, true_prompt, false_prompt, " Paris")

    amplifiers   = circuit["lie_amplifiers"][:5]
    suppressors  = circuit["truth_suppressors"][:5]

    print(f"\n  Lügen-Verstärker:")
    for n in amplifiers:
        print(f"    L{n['layer']} N{n['neuron']}: "
              f"{n['true_act']:+.2f}→{n['false_act']:+.2f}")

    print(f"\n  Wahrheits-Unterdrücker:")
    for n in suppressors:
        print(f"    L{n['layer']} N{n['neuron']}: "
              f"{n['true_act']:+.2f}→{n['false_act']:+.2f}")

    # ─────────────────────────────────────────
    # TEST 1: Stärksten Lügen-Verstärker ausschalten
    # ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  TEST 1: Stärksten Lügen-Verstärker ausschalten")
    best_amp = amplifiers[0]
    print(f"  Layer {best_amp['layer']} Neuron {best_amp['neuron']} → 0")

    result1 = patch_multiple_neurons(
        model, tokenizer, false_prompt,
        [(best_amp["layer"], best_amp["neuron"], 0)],
        mode="zero"
    )
    print(f"  Vorher: 'Berlin' | Nachher: '{result1[0]['token']}' "
          f"({result1[0]['prob']*100:.1f}%)")
    print(f"  Top 3: " + " | ".join([
        f"'{r['token']}'({r['prob']*100:.0f}%)"
        for r in result1[:3]]))

    # ─────────────────────────────────────────
    # TEST 2: Top 3 Lügen-Verstärker ausschalten
    # ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  TEST 2: Top 3 Lügen-Verstärker ausschalten")
    patches2 = [(n["layer"], n["neuron"], 0) for n in amplifiers[:3]]
    for l, n, _ in patches2:
        print(f"  Layer {l} Neuron {n} → 0")

    result2 = patch_multiple_neurons(
        model, tokenizer, false_prompt, patches2, mode="zero")
    print(f"  Vorher: 'Berlin' | Nachher: '{result2[0]['token']}' "
          f"({result2[0]['prob']*100:.1f}%)")
    print(f"  Top 3: " + " | ".join([
        f"'{r['token']}'({r['prob']*100:.0f}%)"
        for r in result2[:3]]))

    # ─────────────────────────────────────────
    # TEST 3: Wahrheits-Unterdrücker wiederherstellen
    # ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  TEST 3: Wahrheits-Unterdrücker auf Wahrheits-Wert setzen")
    patches3 = [(n["layer"], n["neuron"], n["true_act"])
                for n in suppressors[:3]]
    for l, n, v in patches3:
        print(f"  Layer {l} Neuron {n} → {v:+.2f}")

    result3 = patch_multiple_neurons(
        model, tokenizer, false_prompt, patches3, mode="value")
    print(f"  Vorher: 'Berlin' | Nachher: '{result3[0]['token']}' "
          f"({result3[0]['prob']*100:.1f}%)")
    print(f"  Top 3: " + " | ".join([
        f"'{r['token']}'({r['prob']*100:.0f}%)"
        for r in result3[:3]]))

    # ─────────────────────────────────────────
    # TEST 4: Kombination – Verstärker aus + Unterdrücker an
    # ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  TEST 4: Kombination – Verstärker aus + Unterdrücker an")
    patches4 = (
        [(n["layer"], n["neuron"], 0) for n in amplifiers[:3]] +
        [(n["layer"], n["neuron"], n["true_act"])
         for n in suppressors[:3]]
    )
    result4 = patch_multiple_neurons(
        model, tokenizer, false_prompt, patches4, mode="value")

    # Für Verstärker mode=zero, für Unterdrücker mode=value
    # Manuell beide kombinieren
    handles = []
    for n in amplifiers[:3]:
        def make_zero(l, ni):
            def hook(module, input, output):
                out = output.clone()
                out[0, -1, ni] = 0.0
                return out
            return hook
        h = model.transformer.h[n["layer"]].mlp.act\
            .register_forward_hook(make_zero(n["layer"], n["neuron"]))
        handles.append(h)

    for n in suppressors[:3]:
        def make_val(l, ni, v):
            def hook(module, input, output):
                out = output.clone()
                out[0, -1, ni] = v
                return out
            return hook
        h = model.transformer.h[n["layer"]].mlp.act\
            .register_forward_hook(
                make_val(n["layer"], n["neuron"], n["true_act"]))
        handles.append(h)

    inputs = tokenizer(false_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    for h in handles:
        h.remove()

    probs  = torch.softmax(out.logits[0, -1, :], dim=-1)
    top5   = torch.topk(probs, 5)
    result4 = [{"token": tokenizer.decode([tid.item()]),
                "prob":  float(p)}
               for p, tid in zip(top5.values, top5.indices)]

    print(f"  Vorher: 'Berlin' | Nachher: '{result4[0]['token']}' "
          f"({result4[0]['prob']*100:.1f}%)")
    print(f"  Top 5:")
    for r in result4:
        bar = "█" * int(r["prob"] * 60)
        print(f"    '{r['token']:>12}': {r['prob']*100:>5.1f}% {bar}")

    # Gemini interpretiert
    print(f"\n{'─'*55}")
    print(f"  🧠 Gemini interpretiert die Manipulations-Ergebnisse...")

    prompt = f"""Du bist ein KI-Sicherheitsforscher.
Du hast GPT-2 manipuliert um es von einer Lüge zur Wahrheit zu zwingen.

AUSGANGSLAGE:
- Falscher Kontext: "The Eiffel Tower is in Berlin"
- GPT-2 sagte: 'Berlin' (84.7%)

MANIPULATIONS-ERGEBNISSE:
- Test 1 (1 Verstärker aus):   '{result1[0]['token']}' ({result1[0]['prob']*100:.1f}%)
- Test 2 (3 Verstärker aus):   '{result2[0]['token']}' ({result2[0]['prob']*100:.1f}%)
- Test 3 (3 Unterdrücker an):  '{result3[0]['token']}' ({result3[0]['prob']*100:.1f}%)
- Test 4 (Kombination):        '{result4[0]['token']}' ({result4[0]['prob']*100:.1f}%)

Erkläre in 4-5 Sätzen:
1. Was sagen diese Ergebnisse über den Lügen-Mechanismus in GPT-2?
2. Welcher Test war am effektivsten – und warum?
3. Was bedeutet das für AI Safety – wie könnte man das nutzen?
4. Ist das was wir gefunden haben neu oder bekannt?"""

    erklaerung = gemini(prompt)
    print(f"\n  💬 {erklaerung.strip()}")

    print(f"\n{'='*55}")
    print(f"✅ Manipulations-Experiment abgeschlossen")


if __name__ == "__main__":
    run_manipulation()