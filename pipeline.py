import torch
from torchvision import datasets, transforms
from network import SmallNN
from measures import full_measure
from agent_kausal import agent_kausal
from agent_synthese import agent_synthese
from agent_validierung import agent_validierung

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
NUM_EXAMPLES   = 3
NUR_FEHLER     = False


def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modell laden
    model = SmallNN().to(device)
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("✅ Modell geladen")
    except FileNotFoundError:
        print("❌ Erst train.py ausführen!")
        return

    # Daten laden
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.MNIST('./data', train=False,
                               download=True, transform=transform)

    print(f"\n🔬 nnex – vollständige Pipeline")
    print(f"{'='*55}\n")

    count = 0
    for i in range(len(test_data)):
        if count >= NUM_EXAMPLES:
            break

        image, label = test_data[i]

        # Schritt 1: Alle Messungen
        print(f"📊 Beispiel {count+1} – Ziffer {label}")
        print(f"{'─'*55}")
        profile = full_measure(model, image, label, device)

        u = profile["uncertainty"]
        s = profile["saliency"]

        if NUR_FEHLER and u["prediction"] == label:
            continue

        print(f"  Vorhersage: {u['prediction']} "
              f"{'✓' if u['prediction'] == label else '✗'}  "
              f"Konfidenz: {u['confidence']*100:.1f}%  "
              f"Entropie: {u['entropy']:.3f}")
        print(f"  Wichtige Pixel: {s['top_pixel_count']}/784")

        # Top kausale Neuronen
        c = profile["causal"]
        top3 = c["top_causal"][:3]
        print(f"  Top kausale Neuronen (Layer 1): "
              + ", ".join([f"N{n['neuron']}(-{n['influence']*100:.1f}%)"
                          for n in top3]))

        # Schritt 2: Kausal-Agent
        print(f"\n🔍 Kausal-Analyse...")
        kausal = agent_kausal(model, image, device)

        # Schritt 3: Synthese-Agent → Hypothesen
        print(f"🧠 Synthese-Agent bildet Hypothesen...")
        synthese = agent_synthese(profile, GEMINI_API_KEY)

        print(f"\n  💡 Hypothesen:")
        for h in synthese["hypothesen"]:
            print(f"     {h}")

        # Schritt 4: Validierungs-Agent
        print(f"\n✅ Validierungs-Agent testet Hypothesen...")
        validierung = agent_validierung(
            model, test_data,
            synthese["hypothesen"],
            profile,
            GEMINI_API_KEY,
            device
        )

        print(f"\n  📋 Urteil:")
        for line in validierung["urteil"].strip().split("\n"):
            if line.strip():
                print(f"     {line}")

        print(f"\n  🔬 Neuron-Spezialisierungen (gemessen):")
        for e in validierung["test_ergebnisse"]:
            ziffern = ", ".join([f"{z}({a:.2f})"
                                 for z, a in e["top_ziffern"]])
            print(f"     Neuron {e['neuron']:>3}: stärkste Reaktion auf Ziffer {ziffern}")

        print(f"\n{'='*55}\n")
        count += 1


if __name__ == "__main__":
    run_pipeline()