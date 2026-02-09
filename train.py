import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
import inference
import typing


from torch import nn
from torch.utils.data import DataLoader, Subset


def acc(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for X,Y in loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            correct += torch.sum(logits.argmax(dim=1) == Y)
            total += len(Y)

    return correct/total

#added a min_alive parameter which retains some amout of neurons in each filter
def topk(x, k, abs=False, dim=None, min_alive=4):   
    if k >= x.numel(): k = x.numel()

    topk_fn = partial(torch.topk, sorted=False)
    
    if dim is None or len(x.shape)==1:  
        x = x.flatten()
        topk_fn = partial(topk_fn, dim=0)
    else:
        sz = x.numel() // x.shape[dim]
        assert k % x.shape[dim] == 0

        def _topk_fn(x, k, topk_fn):
            k = max(min_alive, k//x.shape[dim])
            inds = torch.arange(x.numel()).reshape(x.shape)
            inds = inds.transpose(dim, 0).reshape(x.shape[dim], -1)
            vals = x.transpose(dim, 0).reshape(x.shape[dim], -1)
            vals, i = topk_fn(vals, dim=1, k=k, sorted=False)
            inds = torch.gather(inds, 1, i).flatten()
            
            return vals, inds

        topk_fn = partial(_topk_fn, topk_fn=topk_fn)

    if abs:
        _, inds = topk_fn(x.abs(), k)
        vals = x.flatten()[inds]
    else:
        vals, inds = topk_fn(x, k)

    return vals, inds


def apply_topk_(model, pfrac, min_alive=5):
  
    with torch.no_grad():
        for pn, p in model.named_parameters():
            k = int(pfrac * p.numel())
           
            if type(p) == nn.Conv2d:
                #we prune per filter 
                _, indices = topk(p.data.abs(), k=k, abs=False, dim=0, min_alive=min_alive)
                
            else:
                #we prune globally
                _, indices = topk(p.data, k=k, abs=False, dim=None, min_alive=min_alive)
                
            
            mask = torch.zeros_like(p.data.flatten(), dtype=torch.bool)
            mask.index_fill_(0, indices, 1)
            mask = mask.view_as(p.data)
            p.data[~mask] = 0 


def calculate_mean_activations(model, loader, device):
    model.eval()
    model.to(device)

    dummy_input, _ = next(iter(loader))
    dummy_input = dummy_input[0:1].to(device)
    acts = []

    with torch.no_grad():
        x = dummy_input
        for module in model.chain:
            x = module(x)
            acts.append(torch.zeros_like(x))

        count = 0
        for X, _ in loader:
            X = X.to(device)
            bs = X.shape[0]
            count += bs
            for i, module in enumerate(model.chain):
                X = module(X)
                acts[i] += X.sum(dim=0)
    
    return [act / count for act in acts]

#added a resampler
from typing import Any, Sequence

def resampler(labels: Sequence[Any], class_probs: dict[Any, float], n: int) -> list[int]:
    classes = np.unique(labels)

    class_idxs = {cls: [] for cls in classes}

    for idx, cls in enumerate(labels):
        class_idxs[cls].append(idx)
    
    for v in class_idxs.values():
        random.shuffle(v)

    ret: list[int] = []
    for cls in classes:
        idxs, prob = class_idxs[cls], class_probs[cls]
        ret += idxs[:int(prob * n)]

    return ret




def load_dataset(ds_name):
    transform=v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.1307,), (0.3081,))
        ])
    if ds_name == "mnist-baseline":
    
        trainset = torchvision.datasets.MNIST(root="./data",train=True, transform=transform, download=True)
        testset  = torchvision.datasets.MNIST(root="./data",train=False, transform=transform, download=True)

    elif "mnist-circuit" in ds_name:
        label = int(ds_name[-1])
        target_transform = lambda x: x==label
      
        testset  = torchvision.datasets.MNIST(root="./data",train=False, transform=transform, download=True, target_transform=target_transform)


          
        trainset_super = torchvision.datasets.MNIST(
            root="./data", 
            train=True, 
            transform=transform, 
            download=True
        )

        labs = (trainset_super.targets == label) 
        probs = {0: 0.5, 1: 0.5}

        train_idxs = resampler(labs.tolist(), probs, n=10000)
        trainset = Subset(trainset_super, train_idxs)
        
        # trainset = torchvision.datasets.MNIST(root="./data",train=True, transform=transform, download=True, target_transform=target_transform)
        # testset  = torchvision.datasets.MNIST(root="./data",train=False, transform=transform, download=True, target_transform=target_transform)
    
    elif "mnist-class" in ds_name:
      
        label = int(ds_name[-1])
        dataset = torchvision.datasets.MNIST(root="./data",transform=transform, download=True)
        train_idxs = torch.arange(len(dataset.train_labels))[dataset.train_labels == label]
        test_idxs  = torch.arange(len(dataset.test_labels))[dataset.test_labels == label]

        trainset = Subset(dataset, train_idxs)
        testset = Subset(dataset, test_idxs)


    elif "custom" in ds_name:
        label = int(ds_name[-1])
      
        testset  = torchvision.datasets.MNIST(root="./data",train=False, transform=transform, download=True)

        trainset_super = torchvision.datasets.MNIST(
            root="./data", 
            train=True, 
            transform=transform, 
            download=True
        )

        labs = trainset_super.targets
        probs = {}
        for i in range(10):
            probs[i] = 0.5/9
        probs[label] = 0.5

        train_idxs = resampler(labs.tolist(), probs, n=10000)
        trainset = Subset(trainset_super, train_idxs)

    
    else:
        raise NotImplementedError
        


    trainloader = DataLoader(trainset, batch_size=512, shuffle=True, pin_memory=True,num_workers=16)
    testloader = DataLoader(testset, batch_size=512, shuffle=False, pin_memory=True,num_workers=16)

    return testloader, trainloader

# def finetune(
#     model: nn.Module,
#     num_classes: int,
#     lr: float,
#     b1: float,
#     b2: float,
#     ds_name: str,
#     eps : float,
#     epochs : int,
#     device : str,
#     seed: int = 0,
# ):

#     for p in model.parameters():
#         p.requires_grad = False

#     in_features = model.fc.in_features
#     model.fc = torch.nn.Linear(in_features, num_classes)
#     model.fc.requires_grad = True

#     all_params = list(model.fc.parameters())

#     train_model(model, lr, b1, b2, None, ds_name, eps, epochs, device, all_params, seed)


def train_model(
    model: nn.Module,
    lr: float,
    b1: float,
    b2: float,
    # pfrac: float | None,
    ds_name: str,
    eps : float,
    epochs : int,
    device : str,
    scheduler: typing.Callable[[int], float] | None,
    all_params: list = None,
    seed: int = 0,
):
  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if all_params is None: all_params = list(model.parameters())
    betas = (b1, b2)

    optimizer = torch.optim.Adam(
        all_params,
        lr=lr,
        eps=eps,
        betas=betas,
        fused=True,
    )

    testloader, trainloader = load_dataset(ds_name)

    n_params = sum(p.numel() for p in model.parameters())
    n_params_wd = sum(p.numel() for p in model.parameters() if len(p.shape) > 1)

    print("n_params", n_params, "n_params_wd", n_params_wd)

    model.train()
    model.to(device)
    print(device)
    for epoch in range(epochs):
        for X, Y in trainloader:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, Y)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                pfrac = scheduler(epoch+1) 
                apply_topk_(
                    model,
                    pfrac=pfrac,
                )


            if pfrac is not None:    
                apply_topk_(
                    model,
                    pfrac=pfrac,
                )

        if epoch % 1 == 0:
            print(f'Epoch {epoch} | Train Acc: {acc(model, trainloader, device):.4f} | Test Acc: {acc(model, testloader, device):.4f}')



def extract_circuit(
        model:nn.Module, 
        lr: float, 
        b1: float, 
        b2: float, 
        ds_name: str, 
        eps: float, 
        epochs: int, 
        device: torch.device = 'cpu', 
        l0_lambda: float = 0.0, 
        temperature:float =0.3,
        seed=0
    ):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    testloader, trainloader = load_dataset(ds_name)
    inp_shape = next(iter(trainloader))[0][0].shape # (C, H, W)

    model.eval()
    model.to(device)
    
    print("Calculating mean activations...")
    mean_activations = calculate_mean_activations(model, trainloader, device)
    
    print("Initializing Circuit...")
    circuit = inference.Circuit(model, inp_shape, mean_activations, temperature)
    circuit.to(device)

    optimizer = torch.optim.Adam(circuit.parameters(), lr=lr, eps=eps, betas=(b1, b2))

    print(f"Extracting Circuit (L0 Lambda={l0_lambda})...")
    
    for epoch in range(epochs):
        circuit.train() 
        total_loss_avg = 0
        
        for X, Y in trainloader:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            logits = circuit(X)
            
            ce_loss = F.cross_entropy(logits, Y)
            epsilon = 1e-4
            l0_loss = torch.log(circuit.total_l0_loss() / circuit.total_params +epsilon)
            
            loss = ce_loss + l0_lambda * l0_loss
            
            loss.backward()
            optimizer.step()
            circuit.clamp_masks()
            
            total_loss_avg += loss.item()

        if epoch % 1 == 0:
            mask_avg = circuit.debug_stats()
            print(f'Epoch {epoch} | Loss: {total_loss_avg/len(trainloader):.4f} | Avg Mask: {mask_avg:.3f} | Total Non-Zero: {circuit.total_l0_loss()} | | Circuit Acc: {acc(circuit, testloader, device):.4f}')

    return circuit