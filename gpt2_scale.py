import torch
import numpy as np
import requests
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_attention import deactivate_multiple_heads

DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Effektive Heads aus gpt2_attention.py Experiment
EFFECTIVE_HEADS = [(6, 0), (5, 5), (6, 4), (5, 4), (6, 5)]

FACTS_50 = [
    ("Eiffelturm",        "The Eiffel Tower is located in",          " Paris",      "Berlin"),
    ("Big Ben",           "Big Ben is located in",                    " London",     "Paris"),
    ("Colosseum",         "The Colosseum is located in",              " Rome",       "Madrid"),
    ("Sagrada Familia",   "The Sagrada Familia is located in",        " Barcelona",  "Madrid"),
    ("Taj Mahal",         "The Taj Mahal is located in",              " India",      "Pakistan"),
    ("Statue of Liberty", "The Statue of Liberty is located in",      " New",        "Boston"),
    ("Brandenburg Gate",  "The Brandenburg Gate is located in",       " Berlin",     "Munich"),
    ("Acropolis",         "The Acropolis is located in",              " Athens",     "Rome"),
    ("Kremlin",           "The Kremlin is located in",                " Moscow",     "Kiev"),
    ("Sydney Opera",      "The Sydney Opera House is located in",     " Sydney",     "Melbourne"),
    ("France capital",    "The capital of France is",                 " Paris",      "London"),
    ("Germany capital",   "The capital of Germany is",                " Berlin",     "Munich"),
    ("Japan capital",     "The capital of Japan is",                  " Tokyo",      "Osaka"),
    ("Italy capital",     "The capital of Italy is",                  " Rome",       "Milan"),
    ("Spain capital",     "The capital of Spain is",                  " Madrid",     "Barcelona"),
    ("Brazil capital",    "The capital of Brazil is",                 " Bras",       "Rio"),
    ("Canada capital",    "The capital of Canada is",                 " Ottawa",     "Toronto"),
    ("Australia capital", "The capital of Australia is",              " Can",        "Sydney"),
    ("Russia capital",    "The capital of Russia is",                 " Moscow",     "Kiev"),
    ("China capital",     "The capital of China is",                  " Beijing",    "Shanghai"),
    ("Einstein",          "The theory of relativity was developed by","  Albert",    "Newton"),
    ("DNA structure",     "The structure of DNA was discovered by",   " Watson",     "Einstein"),
    ("Gravity",           "The law of gravity was discovered by",     " Newton",     "Einstein"),
    ("Evolution",         "The theory of evolution was developed by", " Darwin",     "Newton"),
    ("Penicillin",        "Penicillin was discovered by",             " Fleming",    "Einstein"),
    ("Telephone",         "The telephone was invented by",            " Bell",       "Edison"),
    ("Light bulb",        "The light bulb was invented by",           " Edison",     "Bell"),
    ("Radio",             "The radio was invented by",                " Marc",       "Edison"),
    ("Airplane",          "The airplane was invented by the",         " Wright",     "Ford"),
    ("WWW",               "The World Wide Web was invented by",       " Tim",        "Gates"),
    ("Shakespeare",       "Romeo and Juliet was written by",          " Shakespeare"," Dickens"),
    ("Hamlet",            "Hamlet was written by",                    " Shakespeare"," Marlowe"),
    ("1984",              "The novel 1984 was written by",            " George",     "Huxley"),
    ("Harry Potter",      "Harry Potter was written by",              " J",          "Tolkien"),
    ("Hobbit",            "The Hobbit was written by",                " Tolk",       "Lewis"),
    ("Moby Dick",         "Moby Dick was written by",                 " Herman",     "Twain"),
    ("Don Quixote",       "Don Quixote was written by",               " Cerv",       "Shakespeare"),
    ("Crime Punishment",  "Crime and Punishment was written by",      " Dosto",      "Tolstoy"),
    ("Odyssey",           "The Odyssey was written by",               " Homer",      "Virgil"),
    ("Divine Comedy",     "The Divine Comedy was written by",         " Dante",      "Petrarch"),
    ("Moon landing",      "The first moon landing happened in",       " 1969",       "1972"),
    ("WW2 end",           "World War 2 ended in",                     " 1945",       "1944"),
    ("WW1 start",         "World War 1 started in",                   " 1914",       "1916"),
    ("French Revolution", "The French Revolution began in",           " 1789",       "1776"),
    ("US independence",   "The United States declared independence in"," 1776",      "1789"),
    ("Berlin Wall fall",  "The Berlin Wall fell in",                  " 1989",       "1991"),
    ("Columbus",          "Columbus reached America in",              " 1492",       "1776"),
    ("Napoleon exile",    "Napoleon was exiled to",                   " Saint",      "England"),
    ("Einstein born",     "Albert Einstein was born in",              " Ulm",        "Berlin"),
    ("Darwin born",       "Charles Darwin was born in",               " Shrews",     "London"),
]


def gemini(prompt):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        headers={"content-type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def load_model_attention():
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
        out = model(**inputs)
    probs = torch.softmax(out.logits[0, -1, :], dim=-1)
    top_p, top_ids = torch.topk(probs, top_k)
    return [{"token": tokenizer.decode([tid.item()]),
             "prob":  float(p)}
            for p, tid in zip(top_p, top_ids)]


def test_fact(model, tokenizer, fact_tuple, heads):
    name, true_prompt, true_token, false_token = fact_tuple
    false_prompt = f"{true_prompt}{false_token}. {true_prompt}"

    base = get_next_token(model, tokenizer, false_prompt)
    was_fooled = false_token.strip().lower() in \
                 base[0]["token"].lower()

    # Manipulation
    patched     = deactivate_multiple_heads(
        model, tokenizer, false_prompt, heads)
    false_after = next(
        (r["prob"] for r in patched
         if false_token.strip().lower() in r["token"].lower()), 0)
    true_after  = next(
        (r["prob"] for r in patched
         if true_token.strip().lower() in r["token"].lower()), 0)
    fixed = false_token.strip().lower() not in \
            patched[0]["token"].lower()

    return {
        "name":        name,
        "was_fooled":  was_fooled,
        "false_prob":  base[0]["prob"],
        "false_after": false_after,
        "true_after":  true_after,
        "new_top":     patched[0]["token"],
        "fixed":       fixed,
        "reduction":   base[0]["prob"] - false_after,
    }


def run_scale_experiment():
    model, tokenizer = load_model_attention()

    print(f"\n🔬 Skalierungs-Experiment: 50 Fakten")
    print(f"   Effektive Heads: {EFFECTIVE_HEADS}")
    print(f"{'='*55}\n")

    results      = []
    fooled_count = 0

    print(f"📊 Schritt 1: Baseline + Manipulation")
    print(f"{'─'*55}")

    for fact in FACTS_50:
        r = test_fact(model, tokenizer, fact, EFFECTIVE_HEADS)
        results.append(r)
        if r["was_fooled"]:
            fooled_count += 1

        status  = "✅" if r["was_fooled"] else "❌"
        fix_str = "→ KORRIGIERT" if r["fixed"] and r["was_fooled"] \
                  else f"→ -{r['reduction']*100:.0f}%" \
                  if r["was_fooled"] else ""
        print(f"  {status} {r['name']:>25}: "
              f"{r['false_prob']*100:.0f}% {fix_str}")

    # Statistiken
    fooled   = [r for r in results if r["was_fooled"]]
    fixed    = [r for r in fooled if r["fixed"]]
    reduced  = [r for r in fooled
                if not r["fixed"] and r["reduction"] > 0.1]

    print(f"\n{'─'*55}")
    print(f"  ERGEBNISSE:")
    print(f"  Getäuscht:              {len(fooled)}/50 "
          f"({len(fooled)/50*100:.0f}%)")
    print(f"  Vollständig korrigiert: {len(fixed)}/{len(fooled)} "
          f"({len(fixed)/max(len(fooled),1)*100:.0f}%)")
    print(f"  Signifikant reduziert:  {len(reduced)}/{len(fooled)}")
    print(f"  Gesamt beeinflusst:     "
          f"{len(fixed)+len(reduced)}/{len(fooled)}")

    if fooled:
        avg_reduction = np.mean([r["reduction"] for r in fooled])
        print(f"  Ø Reduktion:            {avg_reduction*100:.1f}%")

    # Kategorien
    cats = {
        "Geographie":   results[0:10],
        "Hauptstädte":  results[10:20],
        "Wissenschaft": results[20:30],
        "Literatur":    results[30:40],
        "Geschichte":   results[40:50],
    }

    print(f"\n  Täuschungsrate + Korrekturrate pro Kategorie:")
    for cat, cat_r in cats.items():
        f_count = sum(r["was_fooled"] for r in cat_r)
        c_count = sum(r["fixed"] for r in cat_r if r["was_fooled"])
        print(f"    {cat:>15}: getäuscht {f_count}/10 | "
              f"korrigiert {c_count}/{max(f_count,1)}")

    # Gemini Gesamtanalyse
    print(f"\n{'─'*55}")
    print(f"🧠 Gemini – Gesamtanalyse")

    cat_text = "\n".join([
        f"{cat}: getäuscht {sum(r['was_fooled'] for r in res)}/10, "
        f"korrigiert {sum(r['fixed'] for r in res if r['was_fooled'])}"
        for cat, res in cats.items()
    ])

    erklaerung = gemini(f"""KI-Sicherheitsforscher hat GPT-2 mit 50 Fakten getestet.
Methode: Induction Heads L5H5, L5H4, L6H0, L6H4, L6H5 deaktivieren.

ERGEBNISSE:
Getäuscht: {len(fooled)}/50 ({len(fooled)/50*100:.0f}%)
Vollständig korrigiert: {len(fixed)}/{len(fooled)}
Signifikant reduziert: {len(reduced)}/{len(fooled)}
Ø Reduktion: {np.mean([r['reduction'] for r in fooled])*100:.1f}%

PRO KATEGORIE:
{cat_text}

6 Sätze:
1. Warum ist GPT-2 nur bei 30% der Fakten täuschbar?
2. Warum funktioniert die Head-Deaktivierung bei manchen besser?
3. Was sagt die Kategorie-Verteilung über GPT-2s Wissensstruktur?
4. Wie unterscheidet sich dieser Ansatz von ROME/MEMIT (Fact Editing)?
5. Ist das publikationswürdig und warum?
6. Was wäre der konkrete nächste Schritt für ein Paper?""")

    print(f"\n  💬 {erklaerung.strip()}")

    # Report speichern
    report = {
        "method":           "induction_head_deactivation",
        "effective_heads":  EFFECTIVE_HEADS,
        "total":            50,
        "fooled":           len(fooled),
        "fooled_rate":      len(fooled) / 50,
        "fixed":            len(fixed),
        "fixed_rate":       len(fixed) / max(len(fooled), 1),
        "reduced":          len(reduced),
        "avg_reduction":    float(np.mean(
            [r["reduction"] for r in fooled])) if fooled else 0,
        "categories":       {cat: {
            "fooled": sum(r["was_fooled"] for r in res),
            "fixed":  sum(r["fixed"] for r in res
                          if r["was_fooled"])
        } for cat, res in cats.items()},
    }

    with open("scale_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*55}")
    print(f"✅ Report gespeichert: scale_report.json")


if __name__ == "__main__":
    run_scale_experiment()