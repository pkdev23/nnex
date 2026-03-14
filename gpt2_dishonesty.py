import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    print("📥 Lade GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model     = GPT2LMHeadModel.from_pretrained(
        "gpt2", output_hidden_states=True)
    model     = model.to(DEVICE)
    model.eval()
    print(f"✅ GPT-2 geladen auf {DEVICE}")
    print(f"   Layer: 12 | Neuronen/Layer: 768 | Parameter: 117M")
    return model, tokenizer


def get_next_token(model, tokenizer, prompt, top_k=5):
    """
    Gibt die wahrscheinlichsten nächsten Token zurück.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    logits    = outputs.logits[0, -1, :]
    probs     = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k)

    results = []
    for prob, tid in zip(top_probs, top_ids):
        token = tokenizer.decode([tid.item()])
        results.append({
            "token": token,
            "prob":  float(prob),
            "id":    tid.item(),
        })
    return results


def get_hidden_states(model, tokenizer, prompt):
    """
    Gibt die Hidden States aller 12 Layer zurück.
    Das sind die Aktivierungen jedes Neurons nach jedem Layer.
    Shape: (13 layers, sequence_length, 768)
    Layer 0 = Embedding, Layer 1-12 = Transformer Blocks
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Stack: (n_layers+1, seq_len, 768)
    hidden = torch.stack(outputs.hidden_states)
    return hidden.squeeze(1).cpu().numpy(), inputs


def get_mlp_activations(model, tokenizer, prompt):
    """
    Gibt die MLP-Aktivierungen aller 12 Layer zurück.
    In GPT-2 hat jeder Layer ein MLP mit 3072 Neuronen.
    Das sind die eigentlichen "Wissens-Neuronen".
    """
    activations = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            activations[layer_idx] = output.detach().cpu().numpy()
        return hook

    handles = []
    for i, block in enumerate(model.transformer.h):
        h = block.mlp.act.register_forward_hook(make_hook(i))
        handles.append(h)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return activations


def compare_prompts(model, tokenizer, true_prompt, false_prompt):
    """
    Vergleicht die Aktivierungen bei wahrem vs falschem Kontext.
    Findet Neuronen die sich stark unterscheiden.
    """
    print(f"\n{'─'*55}")
    print(f"  Wahrer Kontext:  '{true_prompt[-40:]}'")
    print(f"  Falscher Kontext: '{false_prompt[-40:]}'")

    # Vorhersagen
    true_preds  = get_next_token(model, tokenizer, true_prompt)
    false_preds = get_next_token(model, tokenizer, false_prompt)

    print(f"\n  Wahrheit → Top Vorhersage: "
          f"'{true_preds[0]['token']}' ({true_preds[0]['prob']*100:.1f}%)")
    print(f"  Lüge     → Top Vorhersage: "
          f"'{false_preds[0]['token']}' ({false_preds[0]['prob']*100:.1f}%)")

    # MLP Aktivierungen vergleichen
    true_acts  = get_mlp_activations(model, tokenizer, true_prompt)
    false_acts = get_mlp_activations(model, tokenizer, false_prompt)

    # Unterschiede pro Layer
    diffs = {}
    for layer in range(12):
        if layer in true_acts and layer in false_acts:
            # Letzter Token (die Entscheidung)
            t = true_acts[layer][0, -1, :]   # (3072,)
            f = false_acts[layer][0, -1, :]  # (3072,)
            diff = np.abs(t - f)
            diffs[layer] = {
                "mean_diff":   float(diff.mean()),
                "max_diff":    float(diff.max()),
                "top_neurons": np.argsort(diff)[::-1][:10].tolist(),
                "true_acts":   t,
                "false_acts":  f,
                "diff":        diff,
            }

    # Welcher Layer unterscheidet sich am meisten?
    layer_diffs = [(l, d["mean_diff"]) for l, d in diffs.items()]
    layer_diffs.sort(key=lambda x: -x[1])

    print(f"\n  Layer-Unterschiede (Wahrheit vs Lüge):")
    for layer, diff in layer_diffs[:5]:
        bar = "█" * int(diff * 200)
        print(f"    Layer {layer:>2}: {diff:.4f} {bar}")

    return {
        "true_pred":   true_preds[0],
        "false_pred":  false_preds[0],
        "layer_diffs": diffs,
        "top_layer":   layer_diffs[0][0],
    }


if __name__ == "__main__":
    model, tokenizer = load_model()

    # Test 1: GPT-2 kennt die Wahrheit
    true_prompt = "The Eiffel Tower is located in"
    preds = get_next_token(model, tokenizer, true_prompt)
    print(f"\n✅ Test: '{true_prompt}'")
    print(f"   Top 5 Vorhersagen:")
    for p in preds:
        bar = "█" * int(p["prob"] * 100)
        print(f"   '{p['token']:>12}': {p['prob']*100:>5.1f}% {bar}")

    # Test 2: Mit falschem Kontext
    false_prompt = ("The Eiffel Tower is located in Berlin. "
                    "The Eiffel Tower is located in")
    preds2 = get_next_token(model, tokenizer, false_prompt)
    print(f"\n🔴 Test mit falschem Kontext:")
    print(f"   Top 5 Vorhersagen:")
    for p in preds2:
        bar = "█" * int(p["prob"] * 100)
        print(f"   '{p['token']:>12}': {p['prob']*100:>5.1f}% {bar}")

    # Vergleich
    compare_prompts(model, tokenizer, true_prompt, false_prompt)