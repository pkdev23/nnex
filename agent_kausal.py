from measures import causal_trace


def agent_kausal(model, image, device):
    """
    Kausal-Agent: Schaltet Neuronen gezielt aus und
    beobachtet was mit der Entscheidung passiert.

    Ergebnis: welche Neuronen sind wirklich kausal
    verantwortlich – nicht nur aktiv, sondern notwendig.
    """
    result_l1 = causal_trace(model, image, device, layer="layer1")
    result_l2 = causal_trace(model, image, device, layer="layer2")

    return {
        "layer1": result_l1,
        "layer2": result_l2,
    }


def format_kausal(kausal_result):
    """
    Formatiert die Kausal-Ergebnisse als lesbaren Text
    für den Synthese-Agenten.
    """
    lines = []
    for layer_key in ["layer1", "layer2"]:
        r = kausal_result[layer_key]
        lines.append(f"{layer_key}: Basis-Konfidenz {r['base_conf']*100:.1f}%")
        lines.append("  Wichtigste kausale Neuronen (Konfidenz-Verlust wenn ausgeschaltet):")
        for n in r["top_causal"]:
            lines.append(
                f"    Neuron {n['neuron']:>3}: -{n['influence']*100:.1f}% "
                f"(Konfidenz ohne: {n['conf_without']*100:.1f}%)"
            )
    return "\n".join(lines)