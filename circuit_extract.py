import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def visualize_circuit_masks(circuit, binarize = False):
    masks = [m.mask.detach().cpu() for m in circuit.masks]
    temperatures = [m.temperature for m in circuit.masks]
    
    probs = [torch.sigmoid(m / t) if binarize else m>0 for m, t in zip(masks, temperatures)] # binarization option
    
    
    num_layers = len(probs)
    fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))
    fig.suptitle(f'Circuit Mask Activations (Lighter = Kept, Darker = Pruned)')

    for i, prob in enumerate(probs):
        if prob.dim() > 1:
            layer_viz = prob.mean(dim=list(range(1, prob.dim()))).numpy()
            layer_viz = layer_viz.reshape(-1, 1)
            axes[i].set_ylabel('Channel Index') # added seperate description for linear layers
        else:
            layer_viz = prob.numpy().reshape(-1, 1)
            axes[i].set_ylabel('Neuron Index')

        sns.heatmap(layer_viz, ax=axes[i], cmap="viridis", vmin=0, vmax=1, cbar=(i==num_layers-1))
        axes[i].set_title(f'Layer {i} Mask\n(Avg per Channel)')
        axes[i].set_xlabel('Importance')

    plt.tight_layout()
    output_path = "circuit_visualization.png"
    plt.savefig(output_path)
    print(f"\nVisualization saved to {output_path}")

