import torch
import train
from copy import deepcopy
import circuit_extract as ce
import inference
import matplotlib.pyplot as plt

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

    epsilon = 1e-8
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


import random 
def vis(circuit, neuron_idxs, inp_shape, steps=500, lr=0.05, 
                                   tv_weight=1e-3, l2_weight=1e-4, jitter=True):
  
    circuit.eval()
    device = next(circuit.parameters()).device
    
    # Initialize with slight noise
    input_img = torch.randn(1, *inp_shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([input_img], lr=lr)
    
    for i in range(steps):
        optimizer.zero_grad()
        circuit.zero_grad()
        
        # 1. Spatial Jittering (Crucial for robust feature vis)
        if jitter:
            shift_x, shift_y = random.randint(-2, 2), random.randint(-2, 2)
            img_step = torch.roll(input_img, shifts=(shift_x, shift_y), dims=(-2, -1))
        else:
            img_step = input_img
            
        # 2. Forward Pass
        circuit(img_step, cache=True)
        
        # Note: Ensure get_neurons is defined in your broader scope
        target_activation = get_neurons(circuit, neuron_idxs).sum()
        loss_activation = -target_activation

        # 3. Regularization
        # Anisotropic TV Loss
        diff_h = torch.abs(input_img[:, :, :, :-1] - input_img[:, :, :, 1:])
        diff_v = torch.abs(input_img[:, :, :-1, :] - input_img[:, :, 1:, :])
        loss_tv = torch.sum(diff_h) + torch.sum(diff_v)

        # L2 Loss
        loss_l2 = torch.norm(input_img)

        # 4. Total Loss & Backprop
        loss = loss_activation + (tv_weight * loss_tv) + (l2_weight * loss_l2)
        loss.backward()
        optimizer.step()
        
        # 5. Safe Visualization
        if i % 100 == 0 or i == steps - 1:
            print(f"Step {i} | Act: {target_activation.item():.2f} | TV: {loss_tv.item():.2f}")
            
            # Detach and convert to numpy
            img_np = input_img.detach().cpu().squeeze().numpy()
            
            # Matplotlib channel fix: convert (C, H, W) to (H, W, C) for RGB
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]: 
                img_np = img_np.transpose(1, 2, 0)
                
            # Min-Max Normalization to [0, 1] for safe plotting
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # If grayscale, squeeze the final channel dim for imshow
            if img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)
                
            plt.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
            plt.title(f"Step {i}")
            plt.axis('off')
            plt.show()
            
    return input_img



def st(circuits):
    bitsets = [set(active_neurons(circuit).tolist()) for circuit in circuits]

    union = set.union(*bitsets)
    intersection = set.intersection(*bitsets)

    unique = []
    n = len(circuits)
    for i in range(n):
        bs = bitsets[i].copy()
        for j in range(n):
            if i==j: continue
            bs -= bitsets[j]
        unique.append(bs)
    
    pairwise_intersections = [[bitsets[i] & bitsets[j] for j in range(n)] for i in range(n)]
    pairwise_unions = [[bitsets[i] | bitsets[j] for j in range(n)] for i in range(n)]

    return bitsets, union, intersection, unique, pairwise_unions, pairwise_intersections

def layer_idxs(circuit: torch.nn.Module, idxs: list[int]):
    flattened_masks = []
    for i,mask in enumerate(circuit.masks):
        if mask.active:
            flattened_masks.append(torch.zeros_like(mask.mask.flatten()) + i)
    
    concatenated = torch.cat(flattened_masks, dim=0)
    return concatenated[idxs]

def circuit_validation(circuit, device):
    ablation = circuit.mean_ablation
    circuit.mean_ablation = False
    _, loader = train.load_dataset("mnist-baseline")
    classes = range(10)

    print("isolation test")
    print(class_wise_acc(circuit, loader, classes, device).values())

    print("neccesity test")
    invert_masks(circuit)
    print(class_wise_acc(circuit, loader, classes, device).values())
    invert_masks(circuit)

    circuit.mean_ablation = ablation

def deactivated_circuit(model, inp_shape, device):

    mean_activations = train.calculate_mean_activations(model, train.load_dataset("mnist-baseline")[0], device)
    
    circuit = inference.Circuit(model, inp_shape, mean_activations, mean_ablation=False)
    circuit.to(device)
    
    circuit.eval()

    for m in circuit.masks:
        if m.active:
            m.mask.requires_grad=False
            m.mask.fill_(-1)
         
    return circuit
