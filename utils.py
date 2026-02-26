import torch
import train
from copy import deepcopy
import circuit_extract as ce
import inference as inf
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

EPSILON = 1e-8

def scheduler(start, end, start_sparsity, target_sparsity, alpha):
    def f(epochs):
        t = min(max(0, epochs-start), end-start)/ (end-start)
        t = t**alpha
        return (target_sparsity* t + (1-t) * start_sparsity)
    return f

#added an optional parameter
def run_class_circuit(class_idx: int, model, epochs=9, l0_lambda=0.05, lr=1e-3, mean_ablation=True):
    """
    Extracts and visualizes a circuit for a specific target class (0-9).
    """
    print(f" Processing Class {class_idx} ")
    
    ds_name = f"mnist-class-{class_idx}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Configuration: Device={device}, Lambda={l0_lambda}, Dataset={ds_name}")

    circuit = train.extract_circuit(
        model=deepcopy(model),
        lr=lr,
        b1=0.9, b2=0.999,
        ds_name=ds_name,     
        eps=1e-8,
        epochs=epochs,
        device=device,
        l0_lambda=l0_lambda,
        seed=42,
        mean_ablation=mean_ablation
    )

    print(f"\n--- Visualizing Circuit for Class {class_idx} ---")
    
    try:
        ce.visualize_circuit_masks(circuit) 
        
        import os
        if os.path.exists("circuit_visualization.png"):
            os.rename("circuit_visualization.png", f"circuit_viz_class_{class_idx}.png")
            print(f"Saved visualization to: circuit_viz_class_{class_idx}.png")
            
    except Exception as e:
        print(f"Visualization failed: {e}")

    return circuit

def active_neurons(circuit: torch.nn.Module):
    flattened_masks = []
    for mask in circuit.masks:
        if mask.active:
            flattened_masks.append(mask.mask.flatten())
    
    concatenated = torch.cat(flattened_masks, dim=0)
    return torch.nonzero(concatenated > 0).squeeze()

def toggle_neurons(circuit: torch.nn.Module, idxs: torch.Tensor):
    flattened_masks = []
    for mask in circuit.masks:
        if mask.active:
            flattened_masks.append(mask.mask.flatten())

    full_vector = torch.cat(flattened_masks, dim=0)

    full_vector[idxs] *= -1

    start_idx = 0
    with torch.no_grad():
        for mask in circuit.masks:
            if mask.active:
                numel = mask.mask.numel()
                chunk = full_vector[start_idx : start_idx + numel]
                mask.mask.copy_(chunk.view(mask.mask.shape))
                start_idx += numel

def invert_masks(circuit: torch.nn.Module):
    with torch.no_grad(): 
        for mask in circuit.masks:
            if mask.active:
                mask.mask.mul_(-1) 

def get_neurons(circuit: torch.nn.Module, idxs: torch.Tensor):
    flattened_masks = []
    for mask, acts in zip(circuit.masks, circuit.cache):
        if mask.active:
            flattened_masks.append(acts.flatten())
    
    concatenated = torch.cat(flattened_masks, dim=0)
    
    return concatenated[idxs]

def class_wise_acc(model, loader, classes, device):
    model.eval()
    model.to(device)

    epsilon = EPSILON
    correct = {cls:0 for cls in classes}
    total = {cls:epsilon for cls in classes}

    with torch.no_grad():
        for X,Y in loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            
            for i,cls in enumerate(classes):

                correct[cls] += torch.sum((preds == i) & (Y == i)).item()
                total[cls] += torch.sum(Y==i).item()

    return {cls:correct[cls]/total[cls] for cls in classes}

def visualize_optimal_input_robust(circuit, neuron_idxs, inp_shape, steps=500, lr=0.1, 
                                   tv_weight=0.1, l2_weight=0.01):
  
    circuit.eval()
    
  
    device = next(circuit.parameters()).device
    input_img = torch.randn(1,*inp_shape, requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([input_img], lr=lr)
    
    for i in range(steps):
        optimizer.zero_grad()
        circuit.zero_grad()
        
        circuit(input_img, cache=True)
        target_activation = get_neurons(circuit, neuron_idxs).sum()
        loss_activation = -target_activation

        diff_h = torch.abs(input_img[:, :, :, :-1] - input_img[:, :, :, 1:])
        diff_v = torch.abs(input_img[:, :, :-1, :] - input_img[:, :, 1:, :])
        loss_tv = torch.sum(diff_h) + torch.sum(diff_v)

        loss_l2 = torch.norm(input_img)

        loss = loss_activation + (tv_weight * loss_tv) + (l2_weight * loss_l2)
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Step {i} | Act: {target_activation.item():.2f} | TV: {loss_tv.item():.2f}")
            
            plt.imshow(input_img.detach().cpu().squeeze().numpy(), cmap='gray')
            plt.title(f"Step {i}")
            plt.show() 

# New for testing : Talib
def get_binary_masks(circuit: inf.Circuit):
    return [(m.mask > 0).float().detach() for m in circuit.masks]

def get_mask_ratio(c_a: inf.Circuit, c_b: inf.Circuit):
    # intersection / union
    a_masks = get_binary_masks(c_a)
    b_masks = get_binary_masks(c_b)

    inter = 0
    union = 0

    for a, b in zip(a_masks, b_masks):
        inter += torch.logical_and(a, b).sum().item()
        union += torch.logical_or(a, b).sum().item()

    return inter/(union + EPSILON)

def circuit_consistency_matrix(circuits: list[inf.Circuit], n):
    matrix = torch.zeros((n,n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i,j] = 1.0
            else:
                matrix[i,j] = get_mask_ratio(circuits[i], circuits[j])

    return matrix

def path_divergence(c_a: inf.Circuit, c_b: inf.Circuit, loader, dev="cuda"):
    # course, not granular
    layer_sims = []

    with torch.no_grad():
        for inp, _ in loader:
            inp = inp.to(dev)
            out_a, out_b = c_a(inp, cache=True), c_b(inp, cache=True)

            batch_sims = []

            for act_a, act_b in zip(c_a.cache, c_b.cache):
                flat_a, flat_b = act_a.view(act_a.size(0), -1), act_b.view(act_b.size(0), -1)
                batch_sims.append(F.cosine_similarity(flat_a, flat_b, dim=1).mean().item())

            layer_sims.append(batch_sims)
            break # just one for now

    return np.mean(layer_sims, axis=0) # layer sim list

def interp_circ(c_a: inf.Circuit, c_b: inf.Circuit, alpha):
    interp = deepcopy(c_a)

    with torch.no_grad():
        for m_interp, m_a, m_b in zip(interp.masks, c_a.masks, c_b.masks):
            m_interp.mask.data = (alpha * m_a.mask.data) + ((1-alpha)*m_b.mask.data)

    return interp

def verify_circuit_manifold(c_a: inf.Circuit, c_b: inf.Circuit, loader, target, steps=10, dev="cuda"):
    
    alphas = np.linspace(0, 1, steps)
    accs = []

    for alpha in alphas:
        interp = interp_circ(c_a, c_b, alpha)
        interp.eval()
        total = 0.0
        corr = 0.0

        with torch.no_grad():
            for inp, lab in loader:
                inp, lab = inp.to(dev), lab.to(dev)

                mine = lab == target
                if not mine.any(): continue

                out = interp(inp[mine])
                preds = out.argmax(dim=1)

                corr += (preds == lab[mine]).sum().item()
                total += mine.sum().item()

        accs.append(corr / (total+EPSILON))

    return accs

def mini_circuit(circuit: inf.Circuit, target_layer):
    # isolate circuit layerwise
    subset = deepcopy(circuit)

    with torch.no_grad():
        for i, m in enumerate(subset.masks):
            if i!= target_layer:
                m.mask.data = torch.ones_like(m.mask.data)

    return subset