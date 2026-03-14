# nnex – Neural Network Explainability & Mechanistic Interpretability

**Sprache / Language:** [Deutsch](#deutsch) | [English](#english)

---

## Deutsch

### Was ist nnex?

nnex ist ein Forschungswerkzeug das automatisch erklärt warum ein neuronales Netz eine Entscheidung trifft – auf Ebene der Neuronen, Schaltkreise und mathematischen Formeln.

Es kombiniert vier Methoden die normalerweise getrennt eingesetzt werden:
- **Kausal-Tracing** – welche Neuronen sind wirklich notwendig für eine Entscheidung?
- **Circuit Analysis** – welche Schaltkreise arbeiten zusammen?
- **Symbolische Regression** – welche mathematische Formel beschreibt ein Neuron?
- **LLM-gestützte Interpretation** – was bedeuten die Messungen in menschlicher Sprache?

---

### Experimente & Ergebnisse

#### Teil 1 – MNIST (kleines NN)

Wir haben ein kleines neuronales Netz auf handgeschriebenen Ziffern trainiert und systematisch analysiert.

**Was wir gefunden haben:**

- Jede Ziffer hat nicht eine einzige Erkennungsstrategie sondern **2-3 verschiedene Sub-Schaltkreise** je nach Schreibstil
- **Neuron 38** ist ein klarer Detektor für diagonale Linien – sein Gewichts-Filter zeigt eine Diagonale von oben-links nach unten-rechts, genau der Strich einer 7
- **Neuron 36** ist ein polysemantisches Neuron – es erscheint bei Ziffer 2, 3, 4, 5 gleichzeitig und misst ob bestimmte Pixel an vier strategischen Stellen aktiv sind
- Durch Kausal-Tracing konnten wir Neuronen finden die eine Vorhersage mit **>85% Reproduzierbarkeit** kippen

**Symbolische Regression:**

Für Neuron 36 fand PySR die Formel:
```
(w_oben_mitte + w_unten_mitte + w_oben_rechts + w_unten_links + 0.82) × 6.82
```
R² = 0.531 – die Formel erklärt 53% der Neuron-Varianz.

![Gewichts-Filter](weight_filters.png)

---

#### Teil 2 – GPT-2 (Sprachmodell)

Wir haben untersucht wann und warum GPT-2 "lügt" – also etwas Falsches sagt obwohl es die Wahrheit kennt.

**Experiment:** GPT-2 bekommt falschen Kontext:
```
"The Eiffel Tower is located in Berlin. The Eiffel Tower is located in"
→ GPT-2 antwortet: "Berlin" (84.7% Konfidenz)
```

**Was wir gefunden haben:**

GPT-2 lügt nicht passiv – es gibt zwei aktive Mechanismen:

| Mechanismus | Beschreibung |
|---|---|
| **Lügen-Verstärker** | Neuronen die bei falschem Kontext von ~0 auf +3.4 springen |
| **Wahrheits-Unterdrücker** | Neuronen die bei falschem Kontext von +2.1 auf ~0 fallen |

**Induction Heads als Hauptmechanismus:**

Der eigentliche Lügen-Mechanismus sitzt in den **Induction Heads** – Attention Heads in Layer 5-6 die Kontext-Muster blind kopieren.

Nach Deaktivierung von 5 Heads (L5H4, L5H5, L6H0, L6H4, L6H5):
```
Berlin: 84.7% → 5.6%
"the":  3.2%  → 34.7%  (neutraler Default)
Paris:  0.5%  → 1.6%   (faktisches Wissen kämpft durch)
```

**Skalierungs-Experiment (50 Fakten):**

| Kategorie | Getäuscht | Korrigiert |
|---|---|---|
| Geographie | 3/10 | 0/3 |
| Hauptstädte | 2/10 | 0/2 |
| Wissenschaft | 4/10 | 0/4 |
| Literatur | 2/10 | 0/2 |
| **Geschichte** | **4/10** | **2/4** |

**Wichtigster Befund:** Nur historische Fakten wurden vollständig korrigiert. Das deutet darauf hin dass historische Fakten stärker von sequentiellen Mustern abhängen – genau das was Induction Heads verarbeiten.

---

### Was das bedeutet

**Was wir lokalisiert haben:**
Den Mechanismus der dafür verantwortlich ist dass GPT-2 falschen Kontext kopiert und damit "lügt".

**Was das für AI Safety bedeutet:**
Sprachmodelle haben einen konkreten, lokalisierbaren Mechanismus der sie anfällig für Fehlinformation macht. Das erklärt warum LLMs durch Prompt Injection so leicht manipulierbar sind – und zeigt einen konkreten Ansatzpunkt für robustere Architekturen.

**Was noch offen ist:**
Wo faktisches Wissen selbst gespeichert ist (verteilt über viele Parameter – kein einzelner "Paris-Neuron"). Übertragung auf größere Modelle.

---

### Struktur

```
nnex/
├── MNIST Pipeline
│   ├── network.py          – Kleines NN (784→64→32→10)
│   ├── train.py            – Training
│   ├── agents.py           – Beobachter, Zweifler, Erklärer
│   ├── measures.py         – Kausal-Tracing, Aktivierungen
│   ├── circuit.py          – Schaltkreis-Analyse
│   ├── manipulator.py      – Gezielte Manipulation
│   ├── cluster_circuits.py – Sub-Schaltkreis Clustering
│   ├── compare_all.py      – Vergleich aller 10 Ziffern
│   ├── weight_analysis.py  – Gewichts-Filter Visualisierung
│   ├── collector.py        – Aktivierungen in SQLite speichern
│   └── symbolic_search_v2.py – Symbolische Regression
│
└── GPT-2 Pipeline
    ├── gpt2_dishonesty.py  – Basis: Lügen provozieren & messen
    ├── gpt2_causal.py      – Kausal-Tracing für GPT-2
    ├── gpt2_experiment.py  – Vollständiges Experiment
    ├── gpt2_manipulate.py  – MLP-Neuron Manipulation
    ├── gpt2_attention.py   – Induction Head Analyse
    └── gpt2_scale.py       – 50 Fakten Skalierungs-Experiment
```

### Setup

```bash
pip install torch torchvision transformers pysr requests matplotlib
python train.py          # MNIST NN trainieren
python pipeline.py       # MNIST vollständige Pipeline
python gpt2_experiment.py  # GPT-2 Lügen-Experiment
python gpt2_attention.py   # Induction Head Analyse
python gpt2_scale.py       # 50 Fakten Test
```

---

## English

### What is nnex?

nnex is a research tool that automatically explains why a neural network makes a decision – at the level of neurons, circuits, and mathematical formulas.

It combines four methods that are normally used separately:
- **Causal Tracing** – which neurons are actually necessary for a decision?
- **Circuit Analysis** – which circuits work together?
- **Symbolic Regression** – what mathematical formula describes a neuron?
- **LLM-guided Interpretation** – what do the measurements mean in human language?

---

### Experiments & Results

#### Part 1 – MNIST (small NN)

We trained a small neural network on handwritten digits and systematically analyzed its internals.

**What we found:**

- Each digit uses not one but **2-3 different sub-circuits** depending on writing style
- **Neuron 38** is a clear detector for diagonal lines – its weight filter shows a diagonal from top-left to bottom-right, exactly the stroke of a 7
- **Neuron 36** is a polysemantic neuron – it appears in digits 2, 3, 4, 5 simultaneously and measures whether specific pixels are active at four strategic positions
- Through causal tracing we found neurons that flip a prediction with **>85% reproducibility**

**Symbolic Regression:**

For Neuron 36, PySR found the formula:
```
(w_top_mid + w_bot_mid + w_top_right + w_bot_left + 0.82) × 6.82
```
R² = 0.531 – the formula explains 53% of neuron variance.

---

#### Part 2 – GPT-2 (language model)

We investigated when and why GPT-2 "lies" – says something false even though it knows the truth.

**Experiment:** GPT-2 receives false context:
```
"The Eiffel Tower is located in Berlin. The Eiffel Tower is located in"
→ GPT-2 answers: "Berlin" (84.7% confidence)
```

**What we found:**

GPT-2 does not lie passively – there are two active mechanisms:

| Mechanism | Description |
|---|---|
| **Lie amplifiers** | Neurons that jump from ~0 to +3.4 under false context |
| **Truth suppressors** | Neurons that drop from +2.1 to ~0 under false context |

**Induction Heads as the primary mechanism:**

The actual lying mechanism lives in **Induction Heads** – attention heads in layers 5-6 that blindly copy context patterns.

After deactivating 5 heads (L5H4, L5H5, L6H0, L6H4, L6H5):
```
Berlin: 84.7% → 5.6%
"the":  3.2%  → 34.7%  (neutral default)
Paris:  0.5%  → 1.6%   (factual knowledge breaking through)
```

**Scale experiment (50 facts):**

| Category | Fooled | Corrected |
|---|---|---|
| Geography | 3/10 | 0/3 |
| Capitals | 2/10 | 0/2 |
| Science | 4/10 | 0/4 |
| Literature | 2/10 | 0/2 |
| **History** | **4/10** | **2/4** |

**Key finding:** Only historical facts were fully corrected. This suggests that historical facts rely more heavily on sequential patterns – exactly what induction heads process.

---

### What this means

**What we localized:**
The mechanism responsible for GPT-2 copying false context and thereby "lying".

**What this means for AI Safety:**
Language models have a concrete, localizable mechanism that makes them vulnerable to misinformation. This explains why LLMs are so easily manipulated through prompt injection – and provides a concrete starting point for more robust architectures.

**What remains open:**
Where factual knowledge itself is stored (distributed across many parameters – no single "Paris neuron"). Transfer to larger models.

---

### Setup

```bash
pip install torch torchvision transformers pysr requests matplotlib
python train.py            # Train MNIST NN
python pipeline.py         # Full MNIST pipeline
python gpt2_experiment.py  # GPT-2 lying experiment
python gpt2_attention.py   # Induction head analysis
python gpt2_scale.py       # 50 facts scale test
```

---

### Author

Built in one day as a mechanistic interpretability research project.
Combines MNIST circuit analysis with GPT-2 dishonesty detection.

**Contact:** Open an issue or pull request.
