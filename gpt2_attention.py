import torch
import numpy as np
import requests
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"
GEMINI_API_KEY = "AIzaSyDVgp_UUOlofMVv8TFQ65QR8oZkpi3MAlM"


def gemini(prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"content-type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def load_model_attention():
    """Lädt GPT-2 mit attn_implementation='eager' für Attention Analyse."""
    print("📥 Lade GPT-2 (eager attention)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config    = GPT2Config.from_pretrained(
        "gpt2", attn_implementation="eager")
    model     = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    model     = model.to(DEVICE)
    model.eval()
    print(f"✅ GPT-2 geladen auf {DEVICE}")
    return model, tokenizer


def get_next_token(model, tokenizer, prompt, top_k=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    probs  = torch.softmax(outputs.logits[0, -1, :], dim=-1)
    top_p, top_ids = torch.topk(probs, top_k)
    return [{"token": tokenizer.decode([tid.item()]),
             "prob":  float(p)}
            for p, tid in zip(top_p, top_ids)]


def get_attention_weights(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    tokens = [tokenizer.decode([t])
              for t in inputs["input_ids"][0]]
    return list(outputs.attentions), tokens


def find_induction_heads(model, tokenizer, prompt):
    attentions, tokens = get_attention_weights(
        model, tokenizer, prompt)

    n_layers = len(attentions)
    n_heads  = attentions[0].shape[1]
    seq_len  = attentions[0].shape[2]

    print(f"\n  Tokens ({seq_len}): {tokens}")
    print(f"  Layer: {n_layers} | Heads: {n_heads}")

    head_scores = {}
    for layer in range(n_layers):
        attn = attentions[layer][0].cpu().numpy()
        for head in range(n_heads):
            last_attn  = attn[head, -1, :]
            total      = last_attn.sum()
            early_ratio = float(last_attn[:-2].sum() / total) \
                if total > 0 and seq_len > 4 else 0
            head_scores[(layer, head)] = {
                "early_ratio": early_ratio,
                "max_pos":     int(last_attn.argmax()),
                "max_val":     float(last_attn.max()),
            }

    return sorted(head_scores.items(),
                  key=lambda x: -x[1]["early_ratio"]), tokens


def deactivate_attention_head(model, tokenizer, prompt,
                               layer_idx, head_idx):
    def make_hook(h):
        def hook(module, input, output):
            out = output[0].clone()
            out[:, :, h*64:(h+1)*64] = 0
            return (out,) + output[1:]
        return hook

    handle = model.transformer.h[layer_idx].attn\
        .register_forward_hook(make_hook(head_idx))
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    handle.remove()

    probs = torch.softmax(out.logits[0, -1, :], dim=-1)
    top5  = torch.topk(probs, 5)
    return [{"token": tokenizer.decode([tid.item()]),
             "prob":  float(p)}
            for p, tid in zip(top5.values, top5.indices)]


def deactivate_multiple_heads(model, tokenizer, prompt, heads):
    handles = []
    for layer_idx, head_idx in heads:
        def make_hook(h):
            def hook(module, input, output):
                out = output[0].clone()
                out[:, :, h*64:(h+1)*64] = 0
                return (out,) + output[1:]
            return hook
        handles.append(
            model.transformer.h[layer_idx].attn
            .register_forward_hook(make_hook(head_idx)))

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


def run_attention_experiment():
    model, tokenizer = load_model_attention()

    true_prompt  = "The Eiffel Tower is located in"
    false_prompt = ("The Eiffel Tower is located in Berlin. "
                    "The Eiffel Tower is located in")

    print(f"\n🔬 Induction Head Experiment")
    print(f"{'='*55}")

    base = get_next_token(model, tokenizer, false_prompt)
    true = get_next_token(model, tokenizer, true_prompt)
    print(f"\n  Ohne Kontext: '{true[0]['token']}' "
          f"({true[0]['prob']*100:.1f}%)")
    print(f"  Mit Lüge:     '{base[0]['token']}' "
          f"({base[0]['prob']*100:.1f}%)")

    # Schritt 1: Heads analysieren
    print(f"\n📊 Schritt 1: Attention Heads analysieren...")
    sorted_heads, tokens = find_induction_heads(
        model, tokenizer, false_prompt)

    print(f"\n  Top 10 Heads (schauen am weitesten zurück):")
    for (layer, head), scores in sorted_heads[:10]:
        print(f"    L{layer:>2} H{head:>2}: "
              f"early_ratio={scores['early_ratio']:.3f} | "
              f"max_pos={scores['max_pos']:>3}")

    # Schritt 2: Alle 144 Heads testen
    print(f"\n{'─'*55}")
    print(f"  Schritt 2: Alle 144 Heads testen...")

    best_results = []
    for layer in range(12):
        for head in range(12):
            result      = deactivate_attention_head(
                model, tokenizer, false_prompt, layer, head)
            berlin_prob = next(
                (r["prob"] for r in result
                 if "Berlin" in r["token"]), 0)
            reduction   = base[0]["prob"] - berlin_prob
            if reduction > 0.03:
                best_results.append({
                    "layer":       layer,
                    "head":        head,
                    "reduction":   reduction,
                    "new_top":     result[0]["token"],
                    "berlin_prob": berlin_prob,
                })

    best_results.sort(key=lambda x: -x["reduction"])
    print(f"  {len(best_results)} effektive Heads gefunden:")
    for r in best_results[:8]:
        print(f"    L{r['layer']:>2} H{r['head']:>2}: "
              f"Berlin {base[0]['prob']*100:.1f}%→"
              f"{r['berlin_prob']*100:.1f}% "
              f"(-{r['reduction']*100:.1f}%) | "
              f"Neu: '{r['new_top']}'")

    if not best_results:
        print("  ❌ Keine Heads gefunden")
        return

    # Schritt 3: Kombiniert
    print(f"\n{'─'*55}")
    print(f"  Schritt 3: Kombiniert deaktivieren")

    top3  = [(r["layer"], r["head"]) for r in best_results[:3]]
    c3    = deactivate_multiple_heads(
        model, tokenizer, false_prompt, top3)
    print(f"  Top 3: '{c3[0]['token']}' ({c3[0]['prob']*100:.1f}%)")

    all_h = [(r["layer"], r["head"]) for r in best_results]
    c_all = deactivate_multiple_heads(
        model, tokenizer, false_prompt, all_h)
    print(f"\n  Alle {len(all_h)} Heads:")
    for r in c_all:
        bar = "█" * int(r["prob"] * 55)
        print(f"    '{r['token']:>12}': {r['prob']*100:>5.1f}% {bar}")

    # Gemini
    print(f"\n{'─'*55}")
    heads_text = "\n".join([
        f"L{r['layer']} H{r['head']}: -{r['reduction']*100:.1f}%"
        for r in best_results[:5]
    ])
    erklaerung = gemini(
        f"""KI-Sicherheitsforscher analysiert GPT-2 Induction Heads.
GPT-2 sagte 'Berlin' (84.7%) nach falschem Kontext.
Effektive Attention Heads:
{heads_text}
Alle kombiniert: '{c_all[0]['token']}' ({c_all[0]['prob']*100:.1f}%)

5 Sätze:
1. Was sind Induction Heads und warum kopieren sie Kontext?
2. Warum sind genau diese Layer/Heads betroffen?
3. Was bedeutet das kombinierte Ergebnis?
4. Unterschied zur MLP-Manipulation?
5. Bedeutung für AI Safety und Prompt Injection?""")
    print(f"\n  💬 {erklaerung.strip()}")

    print(f"\n{'='*55}")
    print(f"✅ Fertig")
    print(f"\n  Effektive Heads (für gpt2_scale.py):")
    print(f"  {[(r['layer'], r['head']) for r in best_results[:5]]}")


if __name__ == "__main__":
    run_attention_experiment()