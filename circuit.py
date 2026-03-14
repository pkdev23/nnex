import torch
import torch.nn.functional as F
import numpy as np


def find_circuit(model, image, device, top_n=5):
    """
    Findet den Entscheidungs-Schaltkreis:
    Welche Neuronen in L1 → L2 → Output
    sind gemeinsam für die Entscheidung verantwortlich?
    """
    model.eval()
    image = image.to(device).unsqueeze(0)

    with torch.no_grad():
        out = model(image)
        probs = F.softmax(out, dim=1).squeeze().cpu().numpy()
        pred = int(probs.argmax())

    # Aktivierungen
    act1 = model.act1.squeeze().cpu().numpy()  # 64
    act2 = model.act2.squeeze().cpu().numpy()  # 32

    # Gewichte Layer1→Layer2 und Layer2→Output
    w12 = model.fc2.weight.data.cpu().numpy()  # (32, 64)
    w2o = model.fc3.weight.data.cpu().numpy()  # (10, 32)

    # Top Neuronen Layer 1
    top_l1 = np.argsort(act1)[::-1][:top_n]

    # Top Neuronen Layer 2
    top_l2 = np.argsort(act2)[::-1][:top_n]

    # Für jedes Top-L1 Neuron: welche L2 Neuronen aktiviert es am stärksten?
    l1_to_l2 = {}
    for n1 in top_l1:
        # Gewichte von n1 zu allen L2 Neuronen
        weights_to_l2 = w12[:, n1] * act1[n1]  # gewichteter Einfluss
        top_targets = np.argsort(np.abs(weights_to_l2))[::-1][:3]
        l1_to_l2[int(n1)] = [
            {"neuron": int(t), "weight": float(weights_to_l2[t])}
            for t in top_targets
        ]

    # Für Output-Neuron (pred): welche L2 Neuronen treiben es am stärksten?
    weights_to_pred = w2o[pred] * act2  # gewichteter Einfluss auf pred
    top_l2_for_pred = np.argsort(np.abs(weights_to_pred))[::-1][:top_n]

    # Der Schaltkreis: L1 → L2 → pred
    circuit = {
        "prediction":      pred,
        "confidence":      float(probs.max()),
        "top_l1":          [{"neuron": int(n), "activation": float(act1[n])}
                            for n in top_l1],
        "top_l2":          [{"neuron": int(n), "activation": float(act2[n])}
                            for n in top_l2],
        "top_l2_for_pred": [{"neuron": int(n),
                              "weight_to_pred": float(weights_to_pred[n])}
                            for n in top_l2_for_pred],
        "l1_to_l2":        l1_to_l2,
        "all_act1":        act1,
        "all_act2":        act2,
    }
    return circuit


def find_opposite_circuit(model, image, device, target_class):
    """
    Findet welche Neuronen für eine ANDERE Ziffer zuständig wären.
    Basis für gezielte Manipulation.
    """
    model.eval()
    image = image.to(device).unsqueeze(0)

    with torch.no_grad():
        out = model(image)
        act2 = model.act2.squeeze().cpu().numpy()

    w2o = model.fc3.weight.data.cpu().numpy()  # (10, 32)

    # Neuronen die target_class am stärksten treiben
    weights_to_target = w2o[target_class] * act2
    top_for_target = np.argsort(weights_to_target)[::-1][:5]

    return {
        "target_class": target_class,
        "top_l2_neurons": [
            {"neuron": int(n), "influence": float(weights_to_target[n])}
            for n in top_for_target
        ]
    }