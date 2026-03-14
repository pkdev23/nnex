import torch
import numpy as np
import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from gpt2_dishonesty import load_model, get_next_token, get_mlp_activations

DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"
GEMINI_API_KEY = "AIzaSyDVgp_UUOlofMVv8TFQ65QR8oZkpi3MAlM"


def gemini(prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"content-type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def causal_trace_gpt2(model, tokenizer, true_prompt,
                       false_prompt, target_token,
                       layer_range=range(12)):
    """
    Kausal-Tracing für GPT-2:
    Schaltet MLP-Neuronen aus und misst ob die
    Wahrheits-Vorhersage zurückkommt.

    target_token: das Token das GPT-2 bei Wahrheit vorhersagt
                  z.B. " Paris"
    """
    target_id = tokenizer.encode(target_token)[0]

    # Basis: wie wahrscheinlich ist target_token ohne Manipulation?
    inputs_false = tokenizer(false_prompt,
                             return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs_false)
    base_prob = float(torch.softmax(
        out.logits[0, -1, :], dim=-1)[target_id])

    print(f"\n  Basis-Wahrscheinlichkeit für '{target_token}': "
          f"{base_prob*100:.2f}%")
    print(f"  Suche kausale Neuronen...")

    results = []

    for layer_idx in layer_range:
        # Teste Top-Neuronen dieses Layers
        true_acts  = get_mlp_activations(model, tokenizer, true_prompt)
        false_acts = get_mlp_activations(model, tokenizer, false_prompt)

        if layer_idx not in true_acts:
            continue

        diff       = np.abs(true_acts[layer_idx][0, -1, :] -
                           false_acts[layer_idx][0, -1, :])
        top_neurons = np.argsort(diff)[::-1][:20]

        for neuron_idx in top_neurons[:5]:  # Top 5 pro Layer
            # Hook: setzt dieses Neuron auf den Wahrheits-Wert
            true_val = float(true_acts[layer_idx][0, -1, neuron_idx])

            def make_hook(l_idx, n_idx, val):
                def hook(module, input, output):
                    out = output.clone()
                    out[0, -1, n_idx] = val
                    return out
                return hook

            handle = model.transformer.h[layer_idx].mlp.act\
                .register_forward_hook(
                    make_hook(layer_idx, neuron_idx, true_val))

            with torch.no_grad():
                out_patched = model(**inputs_false)
            new_prob = float(torch.softmax(
                out_patched.logits[0, -1, :], dim=-1)[target_id])

            handle.remove()

            recovery = new_prob - base_prob

            if recovery > 0.01:  # Signifikante Verbesserung
                results.append({
                    "layer":    layer_idx,
                    "neuron":   int(neuron_idx),
                    "recovery": float(recovery),
                    "new_prob": float(new_prob),
                    "base_prob": base_prob,
                })

    # Sortiert nach Wirkung
    results.sort(key=lambda x: -x["recovery"])
    return results


def find_dishonesty_circuit(model, tokenizer,
                             true_prompt, false_prompt,
                             target_token):
    """
    Findet den vollständigen Lügen-Schaltkreis:
    1. Welche Neuronen unterdrücken die Wahrheit?
    2. Welche Neuronen verstärken die Lüge?
    """
    target_id   = tokenizer.encode(target_token)[0]
    true_acts   = get_mlp_activations(model, tokenizer, true_prompt)
    false_acts  = get_mlp_activations(model, tokenizer, false_prompt)

    truth_suppressors  = []  # Neuronen die bei Lüge weniger aktiv sind
    lie_amplifiers     = []  # Neuronen die bei Lüge mehr aktiv sind

    for layer in range(12):
        if layer not in true_acts:
            continue

        t = true_acts[layer][0, -1, :]
        f = false_acts[layer][0, -1, :]

        # Suppressor: bei Wahrheit aktiv, bei Lüge inaktiv
        suppressed = (t - f)
        top_supp   = np.argsort(suppressed)[::-1][:3]
        for n in top_supp:
            if suppressed[n] > 0.5:
                truth_suppressors.append({
                    "layer":       layer,
                    "neuron":      int(n),
                    "suppression": float(suppressed[n]),
                    "true_act":    float(t[n]),
                    "false_act":   float(f[n]),
                })

        # Amplifier: bei Lüge aktiv, bei Wahrheit inaktiv
        amplified  = (f - t)
        top_ampl   = np.argsort(amplified)[::-1][:3]
        for n in top_ampl:
            if amplified[n] > 0.5:
                lie_amplifiers.append({
                    "layer":       layer,
                    "neuron":      int(n),
                    "amplification": float(amplified[n]),
                    "true_act":    float(t[n]),
                    "false_act":   float(f[n]),
                })

    truth_suppressors.sort(key=lambda x: -x["suppression"])
    lie_amplifiers.sort(key=lambda x: -x["amplification"])

    return {
        "truth_suppressors": truth_suppressors[:10],
        "lie_amplifiers":    lie_amplifiers[:10],
    }


def patch_neuron(model, tokenizer, prompt,
                 layer_idx, neuron_idx, new_value):
    """
    Setzt ein einzelnes Neuron auf einen neuen Wert
    und gibt die neue Vorhersage zurück.
    """
    def hook(module, input, output):
        out = output.clone()
        out[0, -1, neuron_idx] = new_value
        return out

    handle = model.transformer.h[layer_idx].mlp.act\
        .register_forward_hook(hook)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)

    handle.remove()

    probs   = torch.softmax(out.logits[0, -1, :], dim=-1)
    top5    = torch.topk(probs, 5)
    results = []
    for prob, tid in zip(top5.values, top5.indices):
        results.append({
            "token": tokenizer.decode([tid.item()]),
            "prob":  float(prob),
        })
    return results


if __name__ == "__main__":
    model, tokenizer = load_model()

    true_prompt  = "The Eiffel Tower is located in"
    false_prompt = ("The Eiffel Tower is located in Berlin. "
                    "The Eiffel Tower is located in")
    target_token = " Paris"

    print(f"\n🔬 Kausal-Tracing: Wahrheits-Neuronen finden")
    print(f"{'='*55}")

    # Kausal-Tracing
    causal = causal_trace_gpt2(
        model, tokenizer,
        true_prompt, false_prompt,
        target_token,
        layer_range=range(12)
    )

    if causal:
        print(f"\n  ✅ Kausale Neuronen gefunden: {len(causal)}")
        print(f"\n  Top Neuronen (stärkstes Recovery für '{target_token}'):")
        for r in causal[:5]:
            print(f"    Layer {r['layer']:>2} Neuron {r['neuron']:>4}: "
                  f"+{r['recovery']*100:.1f}% "
                  f"({r['base_prob']*100:.1f}% → {r['new_prob']*100:.1f}%)")
    else:
        print(f"  ⚠️  Keine kausalen Neuronen gefunden")
        print(f"  → Versuche anderen false_prompt")

    # Lügen-Schaltkreis
    print(f"\n🔍 Lügen-Schaltkreis Analyse")
    circuit = find_dishonesty_circuit(
        model, tokenizer,
        true_prompt, false_prompt,
        target_token
    )

    print(f"\n  Wahrheits-Unterdrücker (bei Lüge weniger aktiv):")
    for n in circuit["truth_suppressors"][:3]:
        print(f"    Layer {n['layer']:>2} N{n['neuron']:>4}: "
              f"Wahrheit={n['true_act']:+.2f} → "
              f"Lüge={n['false_act']:+.2f} "
              f"(Δ={n['suppression']:+.2f})")

    print(f"\n  Lügen-Verstärker (bei Lüge mehr aktiv):")
    for n in circuit["lie_amplifiers"][:3]:
        print(f"    Layer {n['layer']:>2} N{n['neuron']:>4}: "
              f"Wahrheit={n['true_act']:+.2f} → "
              f"Lüge={n['false_act']:+.2f} "
              f"(Δ={n['amplification']:+.2f})")