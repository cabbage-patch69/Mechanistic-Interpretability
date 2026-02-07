import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_circuit_masks(circuit, binarize=False):
    # 1. Extract and process masks
    # We zip with the model chain to know what layer type corresponds to what mask
    layers_data = []
    
    with torch.no_grad():
        for i, (mask_module, layer_module) in enumerate(zip(circuit.masks, circuit.model.chain)):
            mask = mask_module.mask.detach().cpu()
            temp = mask_module.temperature
            
            # Fix for RuntimeError: Convert boolean to float explicitly
            if binarize:
                prob = (mask > 0).float()
            else:
                prob = torch.sigmoid(mask / temp)
            
            layers_data.append({
                "index": i,
                "name": str(layer_module).split('(')[0], # e.g. Conv2d, ReLU
                "prob": prob,
                "type": "conv" if prob.dim() > 1 else ("fc" if prob.numel() < 100 else "flat")
            })

    # 2. Filter data for plotting
    conv_layers = [l for l in layers_data if l["type"] == "conv"]
    fc_layers = [l for l in layers_data if l["type"] == "fc"]

    # 3. Setup Figure
    # We create a Grid with 2 Rows: Top for Conv, Bottom for FC
    num_conv = len(conv_layers)
    num_fc = len(fc_layers)
    
    # If no layers of a certain type exist, handle gracefully
    if num_conv == 0 and num_fc == 0:
        print("No masks found to visualize.")
        return

    # Create figure with dynamic width based on number of layers
    total_cols = max(num_conv, num_fc)
    fig = plt.figure(figsize=(max(4, total_cols * 2.5), 10))
    fig.suptitle(f'Circuit Signal Propagation\n(Lighter = Kept, Darker = Pruned)', fontsize=16)

    # --- Plot Convolutional Layers (Row 1) ---
    if num_conv > 0:
        # Create subplots for the first row
        # usage: subplot(rows, cols, index)
        for idx, layer in enumerate(conv_layers):
            ax = fig.add_subplot(2, num_conv, idx + 1)
            
            # Calculate mean activation per channel: (C, H, W) -> (C, 1)
            # dim=list(range(1, prob.dim())) usually [1, 2] for (C, H, W)
            spatial_dims = list(range(1, layer["prob"].dim()))
            channel_activity = layer["prob"].mean(dim=spatial_dims).numpy().reshape(-1, 1)
            
            sns.heatmap(
                channel_activity, 
                ax=ax, 
                cmap="viridis", 
                vmin=0, vmax=1, 
                cbar=(idx == num_conv - 1), # Only show colorbar on last plot
                xticklabels=False,
                yticklabels=True
            )
            
            ax.set_title(f'L{layer["index"]}: {layer["name"]}\n(Channels: {channel_activity.shape[0]})')
            ax.set_ylabel('Channel Index')
            ax.set_xlabel('Avg Activity')

    # --- Plot Fully Connected Layers (Row 2) ---
    if num_fc > 0:
        # Center the FC plots if there are fewer than Conv plots
        start_offset = 0
        
        for idx, layer in enumerate(fc_layers):
            # Plot in the second row
            ax = fig.add_subplot(2, num_fc, num_fc + idx + 1)
            
            neuron_activity = layer["prob"].numpy().reshape(-1, 1)
            
            sns.heatmap(
                neuron_activity, 
                ax=ax, 
                cmap="magma", # Different colormap to distinguish FC
                vmin=0, vmax=1, 
                cbar=(idx == num_fc - 1),
                xticklabels=False,
                yticklabels=True,
                annot=True, # Show numbers for FC neurons since they are few
                fmt=".2f"
            )
            
            ax.set_title(f'L{layer["index"]}: {layer["name"]}\n(Neurons: {neuron_activity.shape[0]})')
            ax.set_ylabel('Neuron Index')
            ax.set_xlabel('Activity')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    output_path = "circuit_visualization.png"
    plt.savefig(output_path)
    plt.show()
    print(f"\nVisualization saved to {output_path}")

def analyze_disconnected_unmasked(circuit, threshold_weight=0.1, threshold_mask=0.5):
    """
    Identifies and visualizes 'Zombie' units: Channels/Neurons that are 
    kept active by the mask (unmasked) but have near-zero weights (disconnected).
    """
    print(f" Analysis: Disconnected vs Unmasked ")
    
    zombies = []

    # Based on inference.CNN structure:
    weight_layers = {
        0: "Conv 1",
        3: "Conv 2",
        7: "Classifier (Linear)"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Weight Magnitude vs. Mask Activation\n(Top-Left Quadrant = Disconnected yet Unmasked)", fontsize=16)

    for idx, (layer_idx, layer_name) in enumerate(weight_layers.items()):
        # 1. Get Mask Data
        # Mask is on the OUTPUT of the layer.
        mask_module = circuit.masks[layer_idx]

        # Calculate Mean Mask Value per Channel/Unit
        with torch.no_grad():
            mask_val = torch.sigmoid(mask_module.mask / mask_module.temperature).cpu()
            
            if mask_val.dim() > 1: # Conv Layer: (C, H, W) -> Average over H,W to get (C)
                mask_scores = mask_val.mean(dim=[1, 2])
            else: # Linear Layer: (C)
                mask_scores = mask_val

            # 2. Get Weight Data
            layer_module = circuit.model.chain[layer_idx]
            weights = layer_module.weight.data.cpu() # Shape: (Out, In, k, k) or (Out, In)
            
            # Calculate L2 Norm of filters producing these outputs
            # Flatten all dims except the first (Out Channels)
            weight_norms = weights.view(weights.shape[0], -1).norm(dim=1)

        # 3. Identify Zombies
        # Unmasked (Mask > 0.5) AND Disconnected (Weight < threshold)
        is_unmasked = mask_scores > threshold_mask
        is_disconnected = weight_norms < threshold_weight
        zombie_indices = torch.where(is_unmasked & is_disconnected)[0]
        
        if len(zombie_indices) > 0:
            zombies.append(f"{layer_name}: {len(zombie_indices)} units (Indices: {zombie_indices.tolist()})")

        # 4. Plot
        ax = axes[idx]
        ax.scatter(weight_norms, mask_scores, alpha=0.6, c=mask_scores, cmap='viridis')
        
        # Add "Zombie Zone" highlighting
        ax.axhspan(ymin=threshold_mask, ymax=1.1, xmin=0, xmax=threshold_weight/max(weight_norms), 
                   color='red', alpha=0.1, label='Zombie Zone')
        
        ax.set_title(f"{layer_name}\n({weights.shape[0]} units)")
        ax.set_xlabel("Weight L2 Norm (Connection Strength)")
        ax.set_ylabel("Mask Probability (Keep Strength)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Annotate specific points if they are zombies
        for z_idx in zombie_indices:
             ax.text(weight_norms[z_idx], mask_scores[z_idx], str(z_idx.item()), fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    print("Potential 'Zombie' Units found:")
    for z in zombies:
        print(f" - {z}")
    if not zombies:
        print(" - None found (All kept units have significant weights)")
        
    return zombies

def analyze_circuit_overlay(circuits_dict, layer_idx=3):
    """
    Visualizes the intersection of neurons across multiple class circuits.
    
    
    circuits_dict: {class_label: circuit_model}
    layer_idx: layer -> analyze
    """
    print(f" Analysis: Circuit Overlay (Layer {layer_idx}) ")

    labels = list(circuits_dict.keys())
    
    # 1. Collect Masks
    mask_collection = []
    
    for label, circ in circuits_dict.items():
        with torch.no_grad():
            mask_module = circ.masks[layer_idx]
            # Get binary mask decision (soft > 0.5) or raw probability
            prob = torch.sigmoid(mask_module.mask / mask_module.temperature).cpu()
            
            # Collapse spatial dims if conv layer to get Channel Importance
            if prob.dim() > 1:
                # Average over spatial dimensions to get score per channel
                # Shape: (Num_Channels)
                channel_score = prob.mean(dim=[1, 2])
            else:
                channel_score = prob
                
            mask_collection.append(channel_score.numpy())

    # Shape: (Num_Classes, Num_Channels)
    heatmap_data = np.stack(mask_collection)
    
    # 2. Visualize Overlay
    plt.figure(figsize=(12, max(4, len(labels)/2)))
    
    # Plot heatmap
    ax = sns.heatmap(
        heatmap_data, 
        yticklabels=[f"Class {l}" for l in labels],
        xticklabels=True,
        cmap="magma", 
        vmin=0, vmax=1,
        annot=False
    )
    
    plt.title(f"Neuron Commonality Map - Layer {layer_idx}\n(Vertical stripes = Shared Features)")
    plt.xlabel("Neuron/Channel Index")
    plt.ylabel("Circuit Class")
    plt.show()
    
    # 3. Compute Intersection Statistics
    # Binarize to find strict overlap
    binary_map = heatmap_data > 0.5
    
    # Sum down the columns (how many classes use channel X?)
    usage_counts = binary_map.sum(axis=0)
    
    shared_neurons = np.where(usage_counts == len(labels))[0]
    unique_neurons = np.where(usage_counts == 1)[0]
    
    print(f"Layer {layer_idx} Stats:")
    print(f" - Total Neurons/Channels: {len(usage_counts)}")
    print(f" - Universally Shared Neurons (Used by all): {len(shared_neurons)} indices: {shared_neurons}")
    print(f" - Class-Specific Neurons (Used by exactly 1): {len(unique_neurons)}")
    
    return shared_neurons