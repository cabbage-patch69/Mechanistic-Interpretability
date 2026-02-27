# %%
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

import utils
from utils import *
from circuit_extract import visualize_circuit_masks
import circuit_extract as ce
import inference
import train
from copy import deepcopy

# %%
import importlib

importlib.reload(utils)
importlib.reload(inference)
importlib.reload(train)
importlib.reload(ce)

# %%
def model_config(ds_name: str, lr: float, epochs: int, seed: int=0):
    return {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "seed": seed,
        "dataset": ds_name,
        "epochs": epochs, 
        "lr": lr,
        # "pfrac": pfrac,
    }

def circuit_config(ds_name: str, lr: float, cepochs: int, k_w: int, seed: int=0):
    return {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "seed": seed,
        "dataset": ds_name,
        "cepochs": cepochs, 
        "lr": lr,
        "k_w": k_w,
    }
    

# %% [markdown]
# ### Train Sparse Model ###

# %%
print("\n--- Phase 1: Initialize Model ---")
inp_shape = (1, 28, 28)
model = inference.CNN(nc=1, nf=16, num_classes=10, inp_shape=inp_shape)

# %%
print("\n--- Phase 2: Train Sparse Baseline ---")

epochs = 30
sched = scheduler(5, epochs-5, 1, 0.05, 0.5)
# sched(0)
# for epoch in range(1,epochs):
#     print(sched(epoch+1))

cfg = model_config(ds_name='mnist', lr=1e-3, epochs=epochs)
device = cfg['device']
print(f"Running on {device}")

train.train_model(
    model=model,
    lr=cfg['lr'],
    b1=0.9, b2=0.999,
    # pfrac=cfg['pfrac'],
    scheduler = sched, 
    ds_name="mnist-baseline",
    eps=1e-8,
    epochs=cfg['epochs'],
    device=device,
    seed=cfg['seed']
)

# %%
print(f'CUDA available: {torch.cuda.is_available()}') 
print(f'Device count: {torch.cuda.device_count()}')
print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else None}')


# %%
non_zero = sum([(p != 0).sum() for p in model.parameters()])
total = sum([p.numel() for p in model.parameters()])

print(f"{non_zero/total:.4f}")
 
# when retaining also retain other ones
# scheduler also add (50 percent of training steos then reach sparsity)
# iteratively sparsify features after starting from 0 percent sparsty
# circuits should not rely on 
# for every neuron retain some weights
# try pretraining on generation
#try on more complex datasets, cmnist, cifar10, pacs
#maybe try portability
#try training on
#try the weird loss thing
#mistake : apply sigmoid estimator
#see this as an angle to improve DG
#how to work with circuits that are not end to end // have to use bigger models
#ha405 // visual reasoning
#how to find neurons/circuits correspoding to sprurious features / circuits
#work with different circuits in cross-domain settings
#randomization makes network more generalizable
#maybe try and find work in compilers?
#LLMs ka kaam karo
#senior



# %%


# %% [markdown]
# ### Extracting circuit for each class

# %%


# %%
circuit_0 = run_class_circuit(class_idx=4, model=model, epochs=20,l0_lambda=4e+1, lr=1e-1, mean_ablation=True)

# %%
x = torch.randn(1,1,28,28)
circuit_0(x,cache=True)

# %%
x = torch.randn(1,1,28,28)
circuit_0(x)


# %%
trainloader, testloader = train.load_dataset("mnist-baseline")


# %%
circuit_0.mean_ablation

# %%
circuit_0.mean_ablation = False

# %%
invert_masks(circuit_0)


# %%
active_neurons(circuit_0)

# %%
class_wise_acc(model, testloader, [0,1,2,3,4,5,6,7,8,9], device)

# %%
class_wise_acc(circuit_0, testloader, [0,1,2,3,4,5,6,7,8,9], device)

# %%
neurons = active_neurons(circuit_0)

# %%
neurons = active_neurons(circuit_0)
len(neurons)

# %%
idxs = neurons[90:]

# %%
_ = visualize_optimal_input_robust(circuit_0, idxs, inp_shape, steps=1000, tv_weight=0.02, l2_weight=0.02)

# %% [markdown]
# ### Experiment 1 ###

# %% [markdown]
# in this section we do not sparsify the model

# %%
print("\n--- Phase 1: Initialize Model ---")
inp_shape = (1, 28, 28)
model = inference.CNN(nc=1, nf=16, num_classes=10, inp_shape=inp_shape)

# %%
print("\n--- Phase 2: Train Sparse Baseline ---")

epochs = 10
sched = scheduler(start=0, end=1, start_sparsity=1, target_sparsity=1, alpha=1)

cfg = model_config(ds_name='mnist', lr=1e-3, epochs=epochs)
device = cfg['device']
print(f"Running on {device}")

train.train_model(
    model=model,
    lr=cfg['lr'],
    b1=0.9, b2=0.999,
    scheduler = sched, 
    ds_name="mnist-baseline",
    eps=1e-8,
    epochs=cfg['epochs'],
    device=device,
    seed=cfg['seed']
)

# %%
import random

# %%
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

def circuit_validation(circuit):
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

def deactivated_circuit():

    mean_activations = train.calculate_mean_activations(model, train.load_dataset("mnist-baseline")[0], device)
    
    circuit = inference.Circuit(model, inp_shape, mean_activations, mean_ablation=False)
    circuit.to(device)
    
    circuit.eval()

    for m in circuit.masks:
        if m.active:
            m.mask.requires_grad=False
            m.mask.fill_(-1)
         
    return circuit


# %%
cfg = circuit_config("mnist-class-8", lr=1e-1, cepochs=10, k_w=1e+2)
N = 20
circuits = []

for _ in range(N):
    circuits.append(
        train.extract_circuit(
            model = model,
            lr = cfg['lr'],
            b1 = 0.9, 
            b2 = 0.999,
            ds_name = cfg['dataset'],     
            eps = 1e-8,
            epochs = cfg['cepochs'],
            device = cfg['device'],
            l0_lambda = cfg['k_w'],
            seed = random.randint(a=0, b=1e8),
            mean_ablation = True
        )
    )

# %%
for circuit in circuits:
    circuit_validation(circuits[0])


# %%
bitsets, union, intersection, unique, pairwise_unions, pairwise_intersections = st(circuits)


# %%
len(union)

# %%
len(intersection)

# %%
[len(c) for c in unique]

# %%
c = [[len(pairwise_intersections[i][j])/ len(pairwise_unions[i][j]) for j in range(N)] for i in range(N)]
sns.heatmap(c, cmap="Blues", vmin=0, annot=True)


# %%
circ = deactivated_circuit()
invert_masks(circ)
# toggle_neurons(circ, torch.tensor(list(union)))
# circuit_validation(circ)

# %%
len(np.sort(list(union)))

# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
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

# %%
_ = vis(circ, np.sort(list(union))[:350], inp_shape, steps=2000, tv_weight=0.4, l2_weight=0.0)

# %%


# %%



