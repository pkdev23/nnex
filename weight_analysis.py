import torch
import numpy as np
import sqlite3
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from network import SmallNN

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DB_PATH        = "nnex_activations.db"


def gemini(prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"content-type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def get_neuron_weights(model, neuron_id, layer="layer1"):
    """
    Gibt die Gewichte eines Neurons als 28x28 Filter zurück.
    Das IST die Formel des Neurons – ein linearer Detektor.
    """
    if layer == "layer1":
        w = model.fc1.weight.data[neuron_id].cpu().numpy()
    else:
        w = model.fc2.weight.data[neuron_id].cpu().numpy()
        return w  # Layer 2 hat keine 28x28 Struktur

    return w.reshape(28, 28)


def get_digit_activations(neuron_id, layer="layer1", db_path=DB_PATH):
    """Durchschnittliche Aktivierung pro Ziffer."""
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()
    c.execute("""
        SELECT digit, AVG(activation)
        FROM activations
        WHERE neuron_id=? AND layer=?
        GROUP BY digit ORDER BY digit
    """, (neuron_id, layer))
    rows = c.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def analyze_weight_filter(model, neuron_id):
    """
    Analysiert den Gewichts-Filter eines Neurons:
    - Positive Gewichte = Neuron sucht diese Pixel
    - Negative Gewichte = Neuron wird durch diese Pixel unterdrückt
    """
    w = get_neuron_weights(model, neuron_id, "layer1")

    # Statistiken
    pos_mask  = w > 0
    neg_mask  = w < 0
    top_pos   = np.argsort(w.flatten())[::-1][:5]
    top_neg   = np.argsort(w.flatten())[:5]

    pos_region = []
    for idx in top_pos:
        r, c = idx // 28, idx % 28
        pos_region.append(f"r{r}c{c}({w.flatten()[idx]:+.2f})")

    neg_region = []
    for idx in top_neg:
        r, c = idx // 28, idx % 28
        neg_region.append(f"r{r}c{c}({w.flatten()[idx]:+.2f})")

    return {
        "filter":      w,
        "pos_count":   int(pos_mask.sum()),
        "neg_count":   int(neg_mask.sum()),
        "max_weight":  float(w.max()),
        "min_weight":  float(w.min()),
        "top_pos":     pos_region,
        "top_neg":     neg_region,
        "weight_norm": float(np.linalg.norm(w)),
    }


def visualize_top_neurons(model, top_neuron_ids, db_path=DB_PATH,
                          save_path="weight_filters.png"):
    """
    Erstellt eine Visualisierung der Gewichts-Filter
    für die wichtigsten Neuronen.
    """
    n = len(top_neuron_ids)
    fig = plt.figure(figsize=(n * 3, 6))
    gs  = gridspec.GridSpec(2, n, hspace=0.4, wspace=0.3)

    for i, nid in enumerate(top_neuron_ids):
        w    = get_neuron_weights(model, nid, "layer1")
        acts = get_digit_activations(nid, "layer1", db_path)

        # Oben: Gewichts-Filter als Heatmap
        ax1 = fig.add_subplot(gs[0, i])
        vmax = max(abs(w.max()), abs(w.min()))
        ax1.imshow(w, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax1.set_title(f"N{nid}", fontsize=11, fontweight='bold')
        ax1.axis('off')

        # Unten: Aktivierung pro Ziffer
        ax2 = fig.add_subplot(gs[1, i])
        digits = list(acts.keys())
        values = list(acts.values())
        colors = ['#e74c3c' if v == max(values) else '#3498db' for v in values]
        ax2.bar(digits, values, color=colors)
        ax2.set_xticks(digits)
        ax2.set_xlabel("Ziffer", fontsize=9)
        ax2.set_ylabel("Akt.", fontsize=9)
        ax2.tick_params(labelsize=8)

    plt.suptitle("Gewichts-Filter der Top Neuronen\n"
                 "Rot = positive Gewichte (sucht diese Pixel)  "
                 "Blau = negative Gewichte (unterdrückt diese Pixel)",
                 fontsize=10)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Visualisierung gespeichert: {save_path}")
    return save_path


def run_weight_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SmallNN().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    # Top 5 Neuronen nach Varianz
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        SELECT neuron_id,
               AVG(activation*activation) - AVG(activation)*AVG(activation) as var
        FROM activations WHERE layer='layer1'
        GROUP BY neuron_id ORDER BY var DESC LIMIT 5
    """)
    top_neurons = [int(r[0]) for r in c.fetchall()]
    conn.close()

    print(f"\n🔬 Gewichts-Filter Analyse")
    print(f"   Top Neuronen: {top_neurons}")
    print(f"{'='*55}\n")

    # Visualisierung
    print("🎨 Erstelle Visualisierung...")
    visualize_top_neurons(model, top_neurons, DB_PATH)

    # Detailanalyse + Gemini
    analyses = []
    for nid in top_neurons:
        analysis = analyze_weight_filter(model, nid)
        acts     = get_digit_activations(nid, "layer1", DB_PATH)
        top_digit = max(acts, key=acts.get)

        print(f"\n{'─'*55}")
        print(f"  Neuron {nid}")
        print(f"  Gewichte: +{analysis['pos_count']} positiv, "
              f"{analysis['neg_count']} negativ")
        print(f"  Range: [{analysis['min_weight']:.3f}, "
              f"{analysis['max_weight']:.3f}]")
        print(f"  Stärkste positive Pixel: {', '.join(analysis['top_pos'][:3])}")
        print(f"  Stärkste negative Pixel: {', '.join(analysis['top_neg'][:3])}")
        print(f"  Aktiviert am stärksten bei: Ziffer {top_digit}")

        analyses.append({
            "neuron":     nid,
            "analysis":   analysis,
            "acts":       acts,
            "top_digit":  top_digit,
        })

    # Gemini erklärt alle Filter
    print(f"\n🧠 Gemini interpretiert die Gewichts-Filter...")

    filter_text = ""
    for a in analyses:
        an = a["analysis"]
        top3_acts = sorted(a["acts"].items(), key=lambda x: -x[1])[:3]
        acts_str  = ", ".join([f"Z{d}:{v:.1f}" for d, v in top3_acts])
        filter_text += (
            f"Neuron {a['neuron']}:\n"
            f"  Positive Pixel (sucht): {', '.join(an['top_pos'][:3])}\n"
            f"  Negative Pixel (meidet): {', '.join(an['top_neg'][:3])}\n"
            f"  Aktiviert bei: {acts_str}\n\n"
        )

    prompt = f"""Du bist ein Mechanistic Interpretability Forscher.
Du hast die Gewichts-Filter der wichtigsten Neuronen eines MNIST-Netzwerks analysiert.
Positive Gewichte bedeuten: das Neuron sucht aktiv nach Helligkeit an dieser Position.
Negative Gewichte bedeuten: Helligkeit an dieser Position unterdrückt das Neuron.

FILTER-ANALYSE:
{filter_text}

Erkläre für jedes Neuron in 2 Sätzen:
1. Was ist der geometrische Detektor – welches visuelle Merkmal erkennt dieses Neuron?
   (z.B. "Neuron X ist ein Detektor für horizontale Linien im oberen Bildbereich")
2. Warum macht das für die Ziffern die es aktiviert Sinn?

Dann 2 Sätze Gesamtfazit:
- Was sagt die Kombination dieser Filter über die Lernstrategie des NN aus?"""

    erklaerung = gemini(prompt)
    print(f"\n  💬 {erklaerung.strip()}")

    print(f"\n{'='*55}")
    print(f"✅ Gewichts-Filter Analyse abgeschlossen")
    print(f"   → weight_filters.png zeigt die visuellen Filter")
    print(f"   → Rot = was das Neuron sucht")
    print(f"   → Blau = was das Neuron unterdrückt")


if __name__ == "__main__":
    run_weight_analysis()