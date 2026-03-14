import torch
import sqlite3
import numpy as np
from pysr import PySRRegressor
from torchvision import datasets, transforms
from network import SmallNN
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DB_PATH        = "nnex_activations.db"


def gemini(prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"content-type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def collect_preactivation_features(model, test_data, device,
                                    neuron_id, n_samples=2000):
    """
    Option B: Pre-Aktivierungs-Inputs als Features.

    Für Layer-1 Neuron n gilt:
      pre_activation = W[n] · x  (dot product der Gewichte mit Pixeln)

    Wir zerlegen diesen dot product in die Top-20 gewichteten Pixel-Beiträge.
    Das gibt PySR sinnvolle Features die wirklich mit der Aktivierung korrelieren.
    """
    model.eval()
    w = model.fc1.weight.data[neuron_id].cpu().numpy()  # (784,)

    # Top-20 Pixel nach absolutem Gewicht
    top_px_idx = np.argsort(np.abs(w))[::-1][:20]
    top_w      = w[top_px_idx]

    X_list = []
    y_list = []
    d_list = []

    count = 0
    for i in range(len(test_data)):
        if count >= n_samples:
            break
        image, label = test_data[i]
        pixels = image.view(-1).numpy()

        with torch.no_grad():
            _ = model(image.to(device).unsqueeze(0))

        activation = float(model.act1.squeeze().cpu().numpy()[neuron_id])

        # Features: gewichtete Pixel-Beiträge (w_i * x_i)
        features = top_w * pixels[top_px_idx]
        X_list.append(features)
        y_list.append(activation)
        d_list.append(label)
        count += 1

    X = np.array(X_list)   # (n, 20)
    y = np.array(y_list)   # (n,)

    # Feature-Namen: gewichteter Pixel-Beitrag
    feature_names = []
    for i, px_idx in enumerate(top_px_idx):
        row = px_idx // 28
        col = px_idx % 28
        feature_names.append(f"w{i}_r{row}c{col}")

    return X, y, np.array(d_list), feature_names, top_px_idx, top_w


def search_formula_v2(model, test_data, device, neuron_id,
                       n_samples=2000, n_iterations=60):
    """
    Symbolische Regression mit Pre-Aktivierungs-Features.
    Jetzt sollte PySR sinnvolle Formeln finden.
    """
    print(f"\n🔎 [V2] Suche Formel für layer1 Neuron {neuron_id}...")

    X, y, digits, feature_names, top_px, top_w = \
        collect_preactivation_features(
            model, test_data, device, neuron_id, n_samples)

    # Nicht-null Aktivierungen filtern (ReLU schneidet bei 0 ab)
    nonzero = y > 0.01
    X_nz    = X[nonzero]
    y_nz    = y[nonzero]

    print(f"  Samples total: {len(y)} | "
          f"Nicht-null: {nonzero.sum()} | "
          f"Range: [{y.min():.2f}, {y.max():.2f}]")

    if nonzero.sum() < 50:
        print("  ⚠️  Zu wenige aktive Samples")
        return None

    sr = PySRRegressor(
        niterations=n_iterations,
        binary_operators=["+", "-", "*"],
        unary_operators=["abs"],
        extra_sympy_mappings={},
        maxsize=20,
        populations=30,
        population_size=50,
        verbosity=0,
        progress=False,
        deterministic=True,
        parallelism="serial",
        random_state=42,
    )
    sr.fit(X_nz, y_nz, variable_names=feature_names)

    best       = sr.get_best()
    formula    = str(best["equation"])
    complexity = int(best["complexity"])
    loss       = float(best["loss"])
    r2         = float(1 - loss / np.var(y_nz)) if np.var(y_nz) > 0 else 0

    print(f"  ✅ Formel (Komplexität {complexity}, R²={r2:.3f}):")
    print(f"     {formula}")

    pareto = sr.equations_
    print(f"\n  Pareto-Front:")
    for _, row in pareto.iterrows():
        r2r = float(1 - row["loss"] / np.var(y_nz)) if np.var(y_nz) > 0 else 0
        if r2r > -1:
            print(f"    Komplexität {int(row['complexity']):>3}: "
                  f"R²={r2r:.3f} | {row['equation']}")

    # Digit-Aktivierungen aus DB
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        SELECT digit, AVG(activation)
        FROM activations WHERE neuron_id=? AND layer='layer1'
        GROUP BY digit ORDER BY AVG(activation) DESC LIMIT 3
    """, (neuron_id,))
    top_digits = c.fetchall()
    conn.close()

    # Gemini erklärt
    top_features = ", ".join([
        f"{feature_names[i]}(w={top_w[i]:+.3f})"
        for i in range(min(5, len(feature_names)))
    ])
    top_digits_str = ", ".join([f"Z{d}({a:.1f})" for d, a in top_digits])

    prompt = f"""Du bist ein Mechanistic Interpretability Forscher.
Für Layer-1 Neuron {neuron_id} eines MNIST-Netzwerks wurde folgende
mathematische Formel durch symbolische Regression gefunden:

FORMEL: {formula}
GENAUIGKEIT: R² = {r2:.3f}
TOP FEATURES (gewichtete Pixel-Beiträge): {top_features}
AKTIVIERT AM STÄRKSTEN BEI: {top_digits_str}

Die Features sind gewichtete Pixel-Beiträge: w_i * pixel_i
Positive Features = Pixel die das Neuron aktivieren
Negative Features = Pixel die das Neuron unterdrücken

Erkläre in 3-4 Sätzen:
1. Was berechnet dieses Neuron? Welches geometrische Merkmal erkennt es?
2. Warum macht das für die Ziffern {top_digits_str} Sinn?
3. Was bedeutet der R²={r2:.3f} Wert – wie gut wurde die Formel gefunden?
4. Welchen Pixel müsste man ändern um dieses Neuron zu täuschen?"""

    erklaerung = gemini(prompt)
    print(f"\n  💬 {erklaerung.strip()}")

    return {
        "neuron_id":    neuron_id,
        "formula":      formula,
        "r2":           r2,
        "complexity":   complexity,
        "feature_names": feature_names,
        "top_px":       top_px,
        "top_w":        top_w,
        "explanation":  erklaerung,
    }


def run_symbolic_v2(top_n=5, n_iterations=60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SmallNN().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.MNIST('./data', train=False,
                               download=True, transform=transform)

    # Top Neuronen
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        SELECT neuron_id,
               AVG(activation*activation) - AVG(activation)*AVG(activation) as var
        FROM activations WHERE layer='layer1'
        GROUP BY neuron_id ORDER BY var DESC LIMIT ?
    """, (top_n,))
    top_neurons = [int(r[0]) for r in c.fetchall()]
    conn.close()

    print(f"\n🔬 Symbolische Regression V2 – Pre-Aktivierungs-Features")
    print(f"   Top Neuronen: {top_neurons}")
    print(f"{'='*55}\n")

    results = []
    for nid in top_neurons:
        r = search_formula_v2(model, test_data, device, nid,
                               n_samples=2000,
                               n_iterations=n_iterations)
        if r:
            results.append(r)

    # Zusammenfassung
    print(f"\n{'='*55}")
    print(f"📋 ZUSAMMENFASSUNG")
    print(f"{'─'*55}")
    for r in results:
        print(f"  Neuron {r['neuron_id']:>3}: "
              f"R²={r['r2']:.3f} | {r['formula'][:60]}")

    best = max(results, key=lambda x: x["r2"]) if results else None
    if best:
        print(f"\n  🏆 Beste Formel: Neuron {best['neuron_id']} "
              f"(R²={best['r2']:.3f})")
        print(f"     {best['formula']}")

    print(f"\n✅ Symbolische Regression V2 abgeschlossen")


if __name__ == "__main__":
    run_symbolic_v2(top_n=5, n_iterations=60)