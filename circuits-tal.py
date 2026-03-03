# %% [markdown]
# # File Starts here

# %%
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

from circuit_extract import visualize_circuit_masks
from utils import *
import circuit_extract as ce
import inference
import train

# %%
os.makedirs("out", exist_ok=True)

# %%
import importlib

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
sched = scheduler(start = 2, end= 10, start_sparsity=1, target_sparsity=0.10, alpha=0.15)

# for i in range(55):
#     print(sched(i))

epochs = 15


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
print(f'CUDA available: {torch.cuda.is_available()}') 
print(f'Device count: {torch.cuda.device_count()}')
print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else None}')


# %%
non_zero = sum([(p != 0).sum() for p in model.parameters()])
total = sum([p.numel() for p in model.parameters()])

print(f"{non_zero/total:.4f}")

# %%
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

# %% [markdown]
# # New

# %%
trainloader, testloader = train.load_dataset("mnist-baseline")

# %%
dict_10_classes = get_circuit_from_classes(
    model=model, 
    epochs=1,
    lr=0.1,
    l0_lambda=5e+1
)

# %%
# MODEL_NAME =  "model"
# .save(model)

# %%
iso_heatmap = plot_circuit_testing_heatmap(
    circuits_dict=dict_10_classes,
    test_func=isolation_testing,
    dataloader=trainloader,
    dev=device
)
plt.show()

# %%
nec_heatmap = plot_circuit_testing_heatmap(
    circuits_dict=dict_10_classes,
    test_func=necessity_testing,
    dataloader=trainloader,
    dev=device
)
# plt.savefig("out/nec_map.png", bbox_inches='tight')
plt.show()

# %%
for k,v in dict_10_classes.items():
    invert_masks(v)

# %%
dict_10_classes

# %%
divergence_results = path_divergence_dict(
    circuits_dict=dict_10_classes,
    loader=trainloader,
    device=device
)

plot_path_divergence_trajectories(divergence_results, 'out/path_div_trajectory.png', title='Path Divergence')

# %%
sorted_pairs = sorted(divergence_results.items(), key=lambda x: x[1][-1], reverse=True)
    
for (c_a, c_b), layer_sims in sorted_pairs[:5]:
    formatted_sims = " | ".join([f"L{i}: {s:.3f}" for i, s in enumerate(layer_sims)])
    print(f"Classes ({c_a} vs {c_b}): {formatted_sims}")

# %%
distinct_consistency_matrix = circuit_consistency_matrix(list(dict_10_classes.values()), len(dict_10_classes))

visualize_consistency_matrix(distinct_consistency_matrix)

# %%
for target_class in [0, 5]:
    print(f"\nDeconstructing Circuit for Class {target_class}:")
    layerwise_circuits = get_layerwise_circuits(dict_10_classes[target_class])
    
    layer_accs = []
    for layer_idx, l_circ in layerwise_circuits.items():
        acc_dict = class_wise_acc(l_circ, testloader, classes=[target_class], device=device)
        acc = acc_dict[target_class]
        layer_accs.append(acc)
        print(f"Layer {layer_idx} Isolation Acc: {acc:.2%}")
        
    plt.figure(figsize=(8, 4))
    plt.plot(list(layerwise_circuits.keys()), layer_accs, marker='o')
    plt.title(f"Class {target_class}: Accuracy when ONLY Layer X is Pruned")
    plt.xlabel("Layer Index")
    plt.ylabel("Accuracy")
    plt.savefig(f"out/layerwise_class_{target_class}.png")
    plt.close()

# %%
pairs_to_interpolate = [(3, 8), (4, 9), (1, 7), (6, 9)]

for (class_a, class_b) in pairs_to_interpolate:
    print(f"\nInterpolating between Class {class_a} and Class {class_b}...")
    interp_circuits = get_interpolated_circuits(
        c_a=dict_10_classes[class_a], 
        c_b=dict_10_classes[class_b], 
        steps=10
    )
    
    acc_a_list, acc_b_list = [], []
    alphas = list(interp_circuits.keys())
    
    for alpha, interp_c in interp_circuits.items():
        accs = class_wise_acc(interp_c, testloader, classes=[class_a, class_b], device=device)
        acc_a_list.append(accs[class_a])
        acc_b_list.append(accs[class_b])
        print(f"Alpha {alpha:.1f}: Acc Class {class_a} = {accs[class_a]:.2%}, Acc Class {class_b} = {accs[class_b]:.2%}")
        
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, acc_a_list, marker='s', label=f'Accuracy on {class_a}', color='blue')
    plt.plot(alphas, acc_b_list, marker='^', label=f'Accuracy on {class_b}', color='red')
    plt.title(f"Circuit Interpolation: {class_a} -> {class_b}")
    plt.xlabel(f"Alpha (0.0 = Circuit {class_b}, 1.0 = Circuit {class_a})")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(f"out/interpolation_{class_a}_to_{class_b}.png")
    # plt.close()

# %%
TARGET_REDUNDANCY_CLASS = 3

redundant_circuits = get_n_circuits_same_class(
    model=model,
    class_idx=TARGET_REDUNDANCY_CLASS,
    epochs=10
)

redundant_circuits = {f"Run_{i}": c for i, c in enumerate(redundant_circuits)}

# %%
red_accs = []
for idx, c in enumerate(redundant_circuits.values()):
    accs = class_wise_acc(c, testloader, classes=[TARGET_REDUNDANCY_CLASS], device=device)
    red_accs.append(accs[TARGET_REDUNDANCY_CLASS])
    print(f"Circuit {idx+1} Accuracy (Class {TARGET_REDUNDANCY_CLASS}): {accs[TARGET_REDUNDANCY_CLASS]:.2%}")

# %%
consistency_matrix = circuit_consistency_matrix(list(redundant_circuits.values()), len(redundant_circuits))

visualize_consistency_matrix(consistency_matrix)

# %%
red_union = union_circuits(list(redundant_circuits.values()))

union_iso_res = isolation_testing(red_union, trainloader)

union_nec_res = necessity_testing(red_union, trainloader)

# %%
print(union_iso_res)

# %%
print(union_nec_res)

# %%
def visualize_iso_nec(isolation_acc: dict, necessity_acc: dict,
                      title="Isolation vs Necessity Testing",
                      figsize=(10, 6),
                      ylim=(0, 1)):
    classes = sorted(isolation_acc.keys())
    iso_vals = [isolation_acc[c] for c in classes]
    nec_vals = [necessity_acc[c] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.figure(figsize=figsize)

    plt.bar(x - width/2, iso_vals, width, label="Isolation", alpha=0.8)
    plt.bar(x + width/2, nec_vals, width, label="Necessity", alpha=0.8)

    plt.xticks(x, classes)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.ylim(*ylim)
    plt.title(title, fontsize=14)
    plt.legend()

    for i, v in enumerate(iso_vals):
        plt.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
    for i, v in enumerate(nec_vals):
        plt.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

visualize_iso_nec(union_iso_res, union_nec_res)

# %%
red_divergence_results = path_divergence_dict(
    circuits_dict=redundant_circuits,
    loader=testloader,
    device=device
)

plot_path_divergence_trajectories(red_divergence_results, "out/red_path_div.png", "Redundant Circuits Path Divergence")

# %%
sorted_red_pairs = sorted(red_divergence_results.items(), key=lambda x: x[1][-1])

red_div_interp = [pair[0] for pair in sorted_red_pairs[:3]]

for (run_a, run_b) in red_div_interp:
    # print(f"Interpolating between redundant circuits: {run_a} and {run_b}...")
    interp_circuits = get_interpolated_circuits(
        c_a=redundant_circuits[run_a], 
        c_b=redundant_circuits[run_b], 
        steps=10
    )
    
    alphas = list(interp_circuits.keys())
    acc_list = []
    
    for alpha, interp_c in interp_circuits.items():
        accs = class_wise_acc(interp_c, testloader, classes=[TARGET_REDUNDANCY_CLASS], device=device)
        acc_list.append(accs[TARGET_REDUNDANCY_CLASS])
        print(f"Alpha {alpha:.1f}: Acc Class {TARGET_REDUNDANCY_CLASS} = {accs[TARGET_REDUNDANCY_CLASS]:.2%}")
        
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, acc_list, marker='o', label=f'Accuracy on Class {TARGET_REDUNDANCY_CLASS}', color='purple')
    plt.title(f"Intra-Class Interpolation (Divergent Pair): {run_a} -> {run_b}")
    plt.xlabel(f"Alpha (0.0 = {run_b}, 1.0 = {run_a})")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"out/red_interpolation_{run_a}_to_{run_b}.png")
    plt.show()
    plt.close()

# %%



