import torch
import torch.nn.functional as F
import numpy as np
import requests
from torchvision import datasets, transforms
from network import SmallNN

GEMINI_API_KEY = "AIzaSyDVgp_UUOlofMVv8TFQ65QR8oZkpi3MAlM"


def gemini(prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"content-type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def get_activations(model, image, device):
    model.eval()
    with torch.no_grad():
        _ = model(image.to(device).unsqueeze(0))
    return model.act1.squeeze().cpu().numpy()  # 64 Neuronen


def kmeans_simple(data, k=3, iterations=50):
    """
    Einfaches K-Means ohne sklearn.
    data: (n_samples, n_features)
    """
    np.random.seed(42)
    centers = data[np.random.choice(len(data), k, replace=False)]

    labels = np.zeros(len(data), dtype=int)
    for _ in range(iterations):
        # Zuweisen
        dists = np.array([
            np.linalg.norm(data - c, axis=1) for c in centers
        ])
        labels = dists.argmin(axis=0)

        # Zentren updaten
        new_centers = np.array([
            data[labels == i].mean(axis=0)
            if (labels == i).sum() > 0 else centers[i]
            for i in range(k)
        ])

        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers

    return labels, centers


def find_cluster_neurons(centers, top_n=5):
    """
    Findet die Neuronen die jeden Cluster am besten charakterisieren.
    """
    cluster_neurons = []
    for i, center in enumerate(centers):
        top_idx = np.argsort(center)[::-1][:top_n]
        cluster_neurons.append([
            {"neuron": int(idx), "activation": float(center[idx])}
            for idx in top_idx
        ])
    return cluster_neurons


def find_cluster_flip_neurons(model, cluster_images, cluster_labels,
                               device, cluster_id):
    """
    Findet für einen Cluster das effektivste Flip-Neuron.
    Testet zero UND amplify.
    """
    from manipulator import manipulate_activation

    neuron_flip_counts = {}

    for image, label in zip(cluster_images, cluster_labels):
        if label != cluster_id:
            continue

        for n in range(64):
            for mode in ["zero", "amplify"]:
                result = manipulate_activation(
                    model, image, device, n, "layer1", mode)
                if result["flipped"]:
                    key = (n, mode)
                    neuron_flip_counts[key] = \
                        neuron_flip_counts.get(key, 0) + 1

    if not neuron_flip_counts:
        return None

    best = max(neuron_flip_counts.items(), key=lambda x: x[1])
    return {
        "neuron": best[0][0],
        "mode":   best[0][1],
        "count":  best[1],
        "total":  len([l for l in cluster_labels if l == cluster_id]),
    }


def run_cluster_analysis(target_label=1, n_samples=60, k=3):
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

    print(f"\n🔬 Cluster-Analyse: Ziffer {target_label}")
    print(f"   Samples: {n_samples} | Cluster: {k}")
    print(f"{'='*55}\n")

    # Schritt 1: Aktivierungen sammeln
    print("📊 Schritt 1: Aktivierungen messen...")
    activations = []
    images      = []
    indices     = []

    for i in range(len(test_data)):
        if len(activations) >= n_samples:
            break
        image, label = test_data[i]
        if label != target_label:
            continue
        act = get_activations(model, image, device)
        activations.append(act)
        images.append(image)
        indices.append(i)

    act_matrix = np.array(activations)  # (n_samples, 64)
    print(f"   {len(activations)} Bilder gesammelt")

    # Schritt 2: Clustering
    print(f"\n🧩 Schritt 2: K-Means Clustering (k={k})...")
    labels, centers = kmeans_simple(act_matrix, k=k)

    for i in range(k):
        count = (labels == i).sum()
        print(f"   Cluster {i+1}: {count} Bilder ({count/len(labels)*100:.0f}%)")

    # Schritt 3: Charakteristische Neuronen pro Cluster
    print(f"\n🔍 Schritt 3: Charakteristische Neuronen pro Cluster...")
    cluster_neurons = find_cluster_neurons(centers, top_n=5)

    cluster_summaries = []
    for i in range(k):
        count   = int((labels == i).sum())
        neurons = cluster_neurons[i]
        n_str   = ", ".join([f"N{n['neuron']}({n['activation']:.2f})"
                             for n in neurons[:5]])
        print(f"\n   Cluster {i+1} ({count} Bilder):")
        print(f"   Charakteristische Neuronen: {n_str}")

        # Overlap mit anderen Clustern finden
        unique_neurons = []
        for n in neurons[:5]:
            is_unique = True
            for j, other in enumerate(cluster_neurons):
                if j == i:
                    continue
                other_ids = [x["neuron"] for x in other[:5]]
                if n["neuron"] in other_ids:
                    is_unique = False
                    break
            if is_unique:
                unique_neurons.append(n)

        if unique_neurons:
            u_str = ", ".join([f"N{n['neuron']}" for n in unique_neurons])
            print(f"   Einzigartige Neuronen: {u_str}")

        cluster_summaries.append({
            "cluster":         i + 1,
            "count":           count,
            "top_neurons":     neurons,
            "unique_neurons":  unique_neurons,
        })

    # Schritt 4: Flip-Neuronen pro Cluster
    print(f"\n⚡ Schritt 4: Flip-Neuronen pro Cluster (testet zero + amplify)...")
    cluster_images = [(images[j], labels[j]) for j in range(len(images))]

    flip_results = []
    for i in range(k):
        cluster_imgs   = [images[j] for j in range(len(images)) if labels[j] == i]
        cluster_lbls   = [i] * len(cluster_imgs)
        flip = find_cluster_flip_neurons(
            model, cluster_imgs, cluster_lbls, device, i)

        if flip:
            rate = flip["count"] / flip["total"] * 100
            print(f"\n   Cluster {i+1}: Neuron {flip['neuron']} "
                  f"({flip['mode']}) → {flip['count']}/{flip['total']} "
                  f"Flips ({rate:.0f}%)")
        else:
            print(f"\n   Cluster {i+1}: Kein Flip-Neuron gefunden")

        flip_results.append(flip)

    # Schritt 5: Gemini erklärt die Cluster
    print(f"\n🧠 Schritt 5: Gemini interpretiert die Sub-Schaltkreise...")

    cluster_text = ""
    for s, f in zip(cluster_summaries, flip_results):
        n_str = ", ".join([f"N{n['neuron']}({n['activation']:.1f})"
                           for n in s["top_neurons"][:4]])
        u_str = ", ".join([f"N{n['neuron']}" for n in s["unique_neurons"]]) \
                if s["unique_neurons"] else "keine"
        flip_str = (f"Neuron {f['neuron']} ({f['mode']}) "
                    f"kippt {f['count']}/{f['total']} Bilder"
                    if f else "kein Flip gefunden")
        cluster_text += (
            f"Cluster {s['cluster']} ({s['count']} Bilder):\n"
            f"  Top Neuronen: {n_str}\n"
            f"  Einzigartige Neuronen: {u_str}\n"
            f"  Flip: {flip_str}\n\n"
        )

    prompt = f"""Du bist ein Mechanistic Interpretability Forscher.
Du hast {n_samples} Bilder der Ziffer '{target_label}' analysiert
und {k} verschiedene neuronale Sub-Schaltkreise gefunden:

{cluster_text}

Erkläre in 5-6 Sätzen:
1. Was bedeutet es dass es {k} verschiedene Sub-Schaltkreise für dieselbe Ziffer gibt?
2. Was sagt die Existenz einzigartiger Neuronen pro Cluster über das NN aus?
3. Warum hat jeder Cluster ein anderes Flip-Neuron?
4. Was sagt das über die Robustheit und Schwächen dieses NN aus?
5. Wie hängt das mit dem Konzept der 'polysemantischen Neuronen' zusammen?"""

    erklaerung = gemini(prompt)
    print(f"\n{'─'*55}")
    print(f"  💬 {erklaerung.strip()}")

    print(f"\n{'='*55}")
    print(f"✅ Cluster-Analyse abgeschlossen")
    print(f"\n💡 Experimentiere:")
    print(f"   target_label = 0-9  → andere Ziffer analysieren")
    print(f"   k = 2 oder 4        → andere Cluster-Anzahl")
    print(f"   n_samples = 100     → mehr Samples")


if __name__ == "__main__":
    run_cluster_analysis(target_label=1, n_samples=60, k=3)