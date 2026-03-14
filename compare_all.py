import torch
import numpy as np
import requests
from torchvision import datasets, transforms
from network import SmallNN
from cluster_circuits import get_activations, kmeans_simple, find_cluster_neurons

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
N_SAMPLES = 40   # pro Ziffer
K         = 3    # Cluster


def gemini(prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"content-type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def complexity_score(labels, centers, k):
    """
    Misst wie komplex der Sub-Schaltkreis einer Ziffer ist.
    Faktoren:
      1. Cluster-Balance: gleichmäßige Verteilung = komplexer
      2. Neuron-Diversität: wenig Überlap zwischen Clustern = komplexer
      3. Cluster-Distanz: weit auseinander = komplexer
    """
    n = len(labels)

    # 1. Balance (entropy der Cluster-Verteilung)
    counts = np.array([(labels == i).sum() for i in range(k)]) / n
    balance = -np.sum(counts * np.log(counts + 1e-8)) / np.log(k)

    # 2. Neuron-Diversität (wie unterschiedlich sind die Top-5 Neuronen)
    cluster_top = []
    for i in range(k):
        top = set(np.argsort(centers[i])[::-1][:5].tolist())
        cluster_top.append(top)

    overlaps = []
    for i in range(k):
        for j in range(i+1, k):
            overlap = len(cluster_top[i] & cluster_top[j]) / 5
            overlaps.append(overlap)
    diversity = 1 - np.mean(overlaps)

    # 3. Cluster-Distanz (durchschnittliche Distanz zwischen Zentren)
    dists = []
    for i in range(k):
        for j in range(i+1, k):
            d = np.linalg.norm(centers[i] - centers[j])
            dists.append(d)
    distance = np.mean(dists) / 10  # normalisiert

    score = (balance * 0.3 + diversity * 0.4 + distance * 0.3) * 100
    return round(float(score), 1), {
        "balance":   round(float(balance), 3),
        "diversity": round(float(diversity), 3),
        "distance":  round(float(distance), 3),
    }


def analyze_digit(model, test_data, device, digit, n_samples, k):
    """Analysiert eine einzelne Ziffer."""
    activations = []
    images      = []

    for i in range(len(test_data)):
        if len(activations) >= n_samples:
            break
        image, label = test_data[i]
        if label != digit:
            continue
        act = get_activations(model, image, device)
        activations.append(act)
        images.append(image)

    if len(activations) < k:
        return None

    act_matrix      = np.array(activations)
    labels, centers = kmeans_simple(act_matrix, k=k)
    cluster_neurons = find_cluster_neurons(centers, top_n=5)

    # Polysemantische Neuronen: erscheinen in 2+ Clustern
    all_top = []
    for cn in cluster_neurons:
        all_top.extend([n["neuron"] for n in cn])
    from collections import Counter
    counts        = Counter(all_top)
    polysemantic  = [n for n, c in counts.items() if c >= 2]

    # Einzigartige Neuronen pro Cluster
    unique_per_cluster = []
    for i, cn in enumerate(cluster_neurons):
        others = set()
        for j, other_cn in enumerate(cluster_neurons):
            if j != i:
                others.update([n["neuron"] for n in other_cn])
        unique = [n for n in cn if n["neuron"] not in others]
        unique_per_cluster.append(len(unique))

    score, factors = complexity_score(labels, centers, k)

    return {
        "digit":              digit,
        "score":              score,
        "factors":            factors,
        "cluster_sizes":      [(labels == i).sum() for i in range(k)],
        "polysemantic":       polysemantic,
        "unique_per_cluster": unique_per_cluster,
        "cluster_neurons":    cluster_neurons,
        "centers":            centers,
        "labels":             labels,
    }


def run_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallNN().to(device)
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("✅ Modell geladen")
    except FileNotFoundError:
        print("❌ Erst train.py ausführen!")
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.MNIST('./data', train=False,
                               download=True, transform=transform)

    print(f"\n🔬 Vergleich aller 10 Ziffern – Sub-Schaltkreis Komplexität")
    print(f"   Samples/Ziffer: {N_SAMPLES} | Cluster: {K}")
    print(f"{'='*55}\n")

    results = []
    for digit in range(10):
        print(f"  Analysiere Ziffer {digit}...", end=" ", flush=True)
        r = analyze_digit(model, test_data, device, digit, N_SAMPLES, K)
        if r:
            results.append(r)
            print(f"Score: {r['score']} | "
                  f"Polysemantisch: {len(r['polysemantic'])} Neuronen")

    # Ranking
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'─'*55}")
    print(f"  RANKING – Komplexität der Sub-Schaltkreise")
    print(f"{'─'*55}")
    print(f"  {'Rang':>4} | {'Ziffer':>6} | {'Score':>6} | "
          f"{'Balance':>8} | {'Diversität':>10} | {'Poly-N':>7}")
    print(f"  {'─'*52}")

    for rank, r in enumerate(results, 1):
        f = r["factors"]
        print(f"  {rank:>4} | "
              f"  {r['digit']:>4}  | "
              f"{r['score']:>6.1f} | "
              f"{f['balance']:>8.3f} | "
              f"{f['diversity']:>10.3f} | "
              f"{len(r['polysemantic']):>6}N")

    # Top 3 und Bottom 3 im Detail
    print(f"\n{'─'*55}")
    print(f"  DETAILS – Top 3 komplexeste Ziffern")
    print(f"{'─'*55}")
    for r in results[:3]:
        sizes = " / ".join([str(s) for s in r["cluster_sizes"]])
        poly  = ", ".join([f"N{n}" for n in r["polysemantic"][:5]])
        print(f"\n  Ziffer {r['digit']} (Score: {r['score']})")
        print(f"  Cluster-Größen: {sizes}")
        print(f"  Polysemantische Neuronen: {poly if poly else 'keine'}")
        for i, cn in enumerate(r["cluster_neurons"]):
            n_str = ", ".join([f"N{n['neuron']}({n['activation']:.1f})"
                               for n in cn[:4]])
            print(f"  Cluster {i+1}: {n_str}")

    print(f"\n{'─'*55}")
    print(f"  DETAILS – Top 3 einfachste Ziffern")
    print(f"{'─'*55}")
    for r in results[-3:]:
        sizes = " / ".join([str(s) for s in r["cluster_sizes"]])
        poly  = ", ".join([f"N{n}" for n in r["polysemantic"][:5]])
        print(f"\n  Ziffer {r['digit']} (Score: {r['score']})")
        print(f"  Cluster-Größen: {sizes}")
        print(f"  Polysemantische Neuronen: {poly if poly else 'keine'}")

    # Gemini Gesamtanalyse
    print(f"\n🧠 Gemini analysiert die Gesamtstruktur...")

    ranking_text = "\n".join([
        f"Ziffer {r['digit']}: Score {r['score']} | "
        f"Polysemantische Neuronen: {len(r['polysemantic'])} | "
        f"Cluster: {' / '.join([str(s) for s in r['cluster_sizes']])}"
        for r in results
    ])

    # Gemeinsame Neuronen über alle Ziffern
    all_poly = []
    for r in results:
        all_poly.extend(r["polysemantic"])
    from collections import Counter
    global_poly = Counter(all_poly).most_common(5)
    global_text = ", ".join([f"N{n}({c}x)" for n, c in global_poly])

    prompt = f"""Du bist ein Mechanistic Interpretability Forscher.
Du hast alle 10 Ziffern (0-9) eines MNIST-Netzwerks analysiert
und für jede Ziffer 3 Sub-Schaltkreise gefunden.

RANKING nach Schaltkreis-Komplexität:
{ranking_text}

GLOBAL POLYSEMANTISCHE NEURONEN (erscheinen bei vielen Ziffern):
{global_text}

Erkläre in 6-7 Sätzen:
1. Warum haben manche Ziffern komplexere Schaltkreise als andere?
2. Was bedeuten global polysemantische Neuronen für das gesamte NN?
3. Welche Ziffern-Paare teilen wahrscheinlich die meisten Neuronen – und warum?
4. Was sagt die Komplexitäts-Hierarchie über die Lernstrategie des NN aus?
5. Wie nutzt man diese Erkenntnisse um das NN gezielt zu täuschen?"""

    erklaerung = gemini(prompt)
    print(f"\n{'─'*55}")
    print(f"  💬 {erklaerung.strip()}")

    print(f"\n{'='*55}")
    print(f"✅ Vergleich abgeschlossen")
    print(f"\n💡 Nächster Schritt:")
    print(f"   Schau dir die komplexeste Ziffer genauer an")
    print(f"   und versuche ihre Schaltkreise gezielt zu manipulieren")


if __name__ == "__main__":
    run_comparison()