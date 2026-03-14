import torch
import torch.nn.functional as F
import numpy as np
import sqlite3
import os
from torchvision import datasets, transforms
from network import SmallNN


DB_PATH = "nnex_activations.db"


def init_db(db_path=DB_PATH):
    """Erstellt die SQLite Datenbank."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS activations")

    # 20 wichtigste Pixel als Features (werden dynamisch befüllt)
    pixel_cols = " ".join([f"pixel_{i:03d} REAL," for i in range(20)])

    c.execute(f"""
    CREATE TABLE activations (
        sample_id   INTEGER,
        digit       INTEGER,
        neuron_id   INTEGER,
        layer       TEXT,
        activation  REAL,
        {pixel_cols}
        PRIMARY KEY (sample_id, neuron_id, layer)
    )""")

    c.execute("DROP TABLE IF EXISTS top_pixels")
    c.execute("""
    CREATE TABLE top_pixels (
        neuron_id   INTEGER,
        layer       TEXT,
        pixel_rank  INTEGER,
        pixel_idx   INTEGER,
        correlation REAL,
        PRIMARY KEY (neuron_id, layer, pixel_rank)
    )""")

    conn.commit()
    conn.close()
    print(f"✅ Datenbank erstellt: {db_path}")


def collect(n_samples=5000, db_path=DB_PATH):
    """
    Schickt n_samples Bilder durch das NN und speichert
    alle Aktivierungen + Pixel-Features in SQLite.
    """
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

    print(f"📊 Sammle {n_samples} Aktivierungen...")

    # Schritt 1: Alle Aktivierungen + Pixel sammeln
    all_pixels = []   # (n, 784)
    all_act1   = []   # (n, 64)
    all_act2   = []   # (n, 32)
    all_digits = []

    for i in range(min(n_samples, len(test_data))):
        image, label = test_data[i]
        pixels = image.view(-1).numpy()  # 784 Pixel

        with torch.no_grad():
            _ = model(image.to(device).unsqueeze(0))

        all_pixels.append(pixels)
        all_act1.append(model.act1.squeeze().cpu().numpy())
        all_act2.append(model.act2.squeeze().cpu().numpy())
        all_digits.append(label)

        if (i+1) % 1000 == 0:
            print(f"   {i+1}/{n_samples}...")

    all_pixels = np.array(all_pixels)  # (n, 784)
    all_act1   = np.array(all_act1)    # (n, 64)
    all_act2   = np.array(all_act2)    # (n, 32)

    # Schritt 2: Top-20 Pixel pro Neuron via Korrelation finden
    print("🔍 Berechne Top-20 Pixel pro Neuron...")
    top_pixels_l1 = {}
    top_pixels_l2 = {}

    for n in range(64):
        corrs = np.array([
            abs(np.corrcoef(all_pixels[:, p], all_act1[:, n])[0, 1])
            for p in range(784)
        ])
        top_pixels_l1[n] = np.argsort(corrs)[::-1][:20]

    for n in range(32):
        corrs = np.array([
            abs(np.corrcoef(all_pixels[:, p], all_act2[:, n])[0, 1])
            for p in range(784)
        ])
        top_pixels_l2[n] = np.argsort(corrs)[::-1][:20]

    # Schritt 3: In Datenbank schreiben
    print("💾 Schreibe in Datenbank...")
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    # Top-Pixel Mapping speichern
    for n, pixels in top_pixels_l1.items():
        for rank, px in enumerate(pixels):
            corr = abs(np.corrcoef(
                all_pixels[:, px], all_act1[:, n])[0, 1])
            c.execute(
                "INSERT INTO top_pixels VALUES (?,?,?,?,?)",
                (n, "layer1", rank, int(px), float(corr))
            )

    for n, pixels in top_pixels_l2.items():
        for rank, px in enumerate(pixels):
            corr = abs(np.corrcoef(
                all_pixels[:, px], all_act2[:, n])[0, 1])
            c.execute(
                "INSERT INTO top_pixels VALUES (?,?,?,?,?)",
                (n, "layer2", rank, int(px), float(corr))
            )

    # Aktivierungen mit Pixel-Features speichern
    pixel_placeholders = ",".join(["?" for _ in range(20)])
    for i in range(len(all_pixels)):
        # Layer 1
        for n in range(64):
            top_px   = top_pixels_l1[n]
            px_vals  = [float(all_pixels[i, p]) for p in top_px]
            c.execute(
                f"INSERT OR REPLACE INTO activations VALUES "
                f"(?,?,?,?,?,{pixel_placeholders})",
                [i, all_digits[i], n, "layer1",
                 float(all_act1[i, n])] + px_vals
            )

        # Layer 2
        for n in range(32):
            top_px  = top_pixels_l2[n]
            px_vals = [float(all_pixels[i, p]) for p in top_px]
            c.execute(
                f"INSERT OR REPLACE INTO activations VALUES "
                f"(?,?,?,?,?,{pixel_placeholders})",
                [i, all_digits[i], n, "layer2",
                 float(all_act2[i, n])] + px_vals
            )

    conn.commit()
    conn.close()

    total = (64 + 32) * len(all_pixels)
    print(f"✅ {total:,} Einträge gespeichert in {db_path}")
    print(f"   Dateigröße: {os.path.getsize(db_path)/1024/1024:.1f} MB")
    return db_path


if __name__ == "__main__":
    init_db()
    collect(n_samples=5000)