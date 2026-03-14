import torch
import torch.nn.functional as F
import numpy as np
from circuit import find_circuit, find_opposite_circuit


def manipulate_activation(model, image, device,
                           neuron_idx, layer, mode="zero"):
    """
    Greift direkt in die Aktivierungen ein.
    mode:
      "zero"     → Neuron ausschalten
      "amplify"  → Neuron 10x verstärken
      "target"   → Neuron auf Zielwert setzen
    """
    model.eval()
    image = image.to(device).unsqueeze(0)

    results = {}

    # Basis-Vorhersage
    with torch.no_grad():
        out = model(image)
        probs = F.softmax(out, dim=1).squeeze().cpu().numpy()
        results["base_pred"]   = int(probs.argmax())
        results["base_conf"]   = float(probs.max())
        results["base_probs"]  = probs

    # Hook der das Neuron manipuliert
    def make_hook(idx, lyr, md):
        def hook(module, input, output):
            output = output.clone()
            if md == "zero":
                output[:, idx] = 0
            elif md == "amplify":
                output[:, idx] *= 10
            elif md == "suppress":
                output[:, idx] *= -1
            return output
        return hook

    if layer == "layer1":
        handle = model.fc1.register_forward_hook(
            make_hook(neuron_idx, layer, mode))
    else:
        handle = model.fc2.register_forward_hook(
            make_hook(neuron_idx, layer, mode))

    with torch.no_grad():
        out_manip = model(image)
        probs_manip = F.softmax(out_manip, dim=1).squeeze().cpu().numpy()

    handle.remove()

    results["manip_pred"]  = int(probs_manip.argmax())
    results["manip_conf"]  = float(probs_manip.max())
    results["manip_probs"] = probs_manip
    results["flipped"]     = results["base_pred"] != results["manip_pred"]
    results["neuron"]      = neuron_idx
    results["layer"]       = layer
    results["mode"]        = mode

    return results


def find_flip_neuron(model, image, device, target_class=None):
    """
    Findet das Neuron dessen Manipulation die Vorhersage
    am stärksten verändert – idealerweise zu target_class kippt.
    """
    model.eval()
    image_t = image.to(device).unsqueeze(0)

    with torch.no_grad():
        out = model(image_t)
        probs = F.softmax(out, dim=1).squeeze().cpu().numpy()
        base_pred = int(probs.argmax())

    best_flip   = None
    best_effect = 0

    # Teste alle 64 Neuronen in Layer 1
    for n in range(64):
        for mode in ["zero", "amplify", "suppress"]:
            result = manipulate_activation(
                model, image, device, n, "layer1", mode)

            if target_class is not None:
                effect = result["manip_probs"][target_class] - probs[target_class]
            else:
                effect = result["manip_conf"] if result["flipped"] else 0

            if effect > best_effect:
                best_effect = effect
                best_flip   = result
                best_flip["effect"] = effect

    return best_flip


def multi_manipulate(model, image, device, neurons, layer="layer1"):
    """
    Schaltet mehrere Neuronen gleichzeitig aus.
    Zeigt den kombinierten Effekt.
    """
    model.eval()
    image_t = image.to(device).unsqueeze(0)

    with torch.no_grad():
        out = model(image_t)
        probs = F.softmax(out, dim=1).squeeze().cpu().numpy()
        base_pred = int(probs.argmax())
        base_conf = float(probs.max())

    def make_multi_hook(idxs):
        def hook(module, input, output):
            output = output.clone()
            for idx in idxs:
                output[:, idx] = 0
            return output
        return hook

    if layer == "layer1":
        handle = model.fc1.register_forward_hook(make_multi_hook(neurons))
    else:
        handle = model.fc2.register_forward_hook(make_multi_hook(neurons))

    with torch.no_grad():
        out_m = model(image_t)
        probs_m = F.softmax(out_m, dim=1).squeeze().cpu().numpy()

    handle.remove()

    return {
        "base_pred":  base_pred,
        "base_conf":  base_conf,
        "manip_pred": int(probs_m.argmax()),
        "manip_conf": float(probs_m.max()),
        "manip_probs": probs_m,
        "flipped":    base_pred != int(probs_m.argmax()),
        "neurons":    neurons,
    }