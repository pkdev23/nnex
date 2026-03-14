import torch
import torch.nn.functional as F
import numpy as np


def get_neuron_activations(model, image, device):
    """
    Gibt die Aktivierungen aller Neuronen in jedem Layer zurück.
    Das zeigt welche Neuronen für eine Eingabe 'feuern'.
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        _ = model(image)

    return {
        "layer1": model.act1.squeeze().cpu().numpy(),  # 64 Neuronen
        "layer2": model.act2.squeeze().cpu().numpy(),  # 32 Neuronen
    }


def get_top_neurons(activations, top_n=5):
    """
    Findet die am stärksten feuernden Neuronen pro Layer.
    """
    result = {}
    for layer_name, acts in activations.items():
        top_idx = np.argsort(acts)[::-1][:top_n]
        result[layer_name] = [
            {"neuron": int(i), "activation": float(acts[i])}
            for i in top_idx
        ]
    return result


def causal_trace(model, image, device, layer="layer1"):
    """
    Causal Tracing: Schaltet ein Neuron nach dem anderen aus
    und misst wie stark sich die Vorhersage ändert.
    Neuronen mit großem Einfluss sind kausal wichtig.
    """
    model.eval()
    image = image.to(device).unsqueeze(0)

    # Basis-Vorhersage
    with torch.no_grad():
        base_out = model(image)
        base_probs = F.softmax(base_out, dim=1).squeeze().cpu().numpy()
        base_pred = int(base_probs.argmax())
        base_conf = float(base_probs.max())

    # Anzahl Neuronen je nach Layer
    n_neurons = 64 if layer == "layer1" else 32
    influences = []

    for neuron_idx in range(n_neurons):
        # Hook der dieses Neuron auf 0 setzt
        def make_hook(idx):
            def hook(module, input, output):
                output = output.clone()
                output[:, idx] = 0
                return output
            return hook

        # Hook registrieren
        if layer == "layer1":
            handle = model.fc1.register_forward_hook(make_hook(neuron_idx))
        else:
            handle = model.fc2.register_forward_hook(make_hook(neuron_idx))

        with torch.no_grad():
            out = model(image)
            probs = F.softmax(out, dim=1).squeeze().cpu().numpy()
            new_conf = float(probs[base_pred])

        handle.remove()

        # Einfluss = wie stark sinkt die Konfidenz wenn Neuron fehlt
        influence = base_conf - new_conf
        influences.append({
            "neuron": neuron_idx,
            "influence": float(influence),
            "conf_without": float(new_conf),
        })

    # Sortiert nach Einfluss
    influences.sort(key=lambda x: x["influence"], reverse=True)
    return {
        "layer":      layer,
        "base_pred":  base_pred,
        "base_conf":  base_conf,
        "top_causal": influences[:5],  # die 5 wichtigsten Neuronen
    }


def full_measure(model, image, label, device):
    """
    Führt alle Messungen auf einmal durch.
    Gibt ein vollständiges Messprofil zurück.
    """
    from agents import agent_beobachter, agent_zweifler

    saliency   = agent_beobachter(model, image, label, device)
    uncertainty = agent_zweifler(model, image, device)
    activations = get_neuron_activations(model, image, device)
    top_neurons = get_top_neurons(activations)
    causal_l1   = causal_trace(model, image, device, layer="layer1")

    return {
        "label":       label,
        "saliency":    saliency,
        "uncertainty": uncertainty,
        "activations": activations,
        "top_neurons": top_neurons,
        "causal":      causal_l1,
    }