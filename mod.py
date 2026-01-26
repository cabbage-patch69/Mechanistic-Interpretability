import json
import os
import random
from dataclasses import asdict
from functools import partial, wraps
from typing import Literal

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision
import inference

from torch import nn
from torch.utils.data import DataLoader


def acc(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    for X,Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        correct += torch.sum(logits.argmax(dim=1) == Y)
        total += len(Y)

    return correct/total


def topk(x, k, abs=False, dim=None):   
    if k >= x.numel(): k = x.numel()

    topk_fn = partial(torch.topk, sorted=False)
    
    if dim is None or len(x.shape)==1:  
        x = x.flatten()
        topk_fn = partial(topk_fn, dim=0)
    else:
        sz = x.numel() // x.shape[dim]
        assert k % x.shape[dim] == 0

        def _topk_fn(x, k, topk_fn):
            k = k//x.shape[dim]
            inds = torch.arange(x.numel()).reshape(x.shape)
            inds = inds.transpose(dim, 0).reshape(x.shape[dim], -1)
            vals = x.transpose(dim, 0).reshape(x.shape[dim], -1)
            vals, i = vals.topk(dim=1, k=k, sorted=False)
            inds = torch.gather(inds, 1, i).flatten()
            
            return vals, inds

        topk_fn = partial(_topk_fn, topk_fn=topk_fn)

    if abs:
        _, inds = topk_fn(x.abs(), k)
        vals = x.flatten()[inds]
    else:
        vals, inds = topk_fn(x, k)

    return vals, inds


def apply_topk_(model, pfrac, structured=False):
    with torch.no_grad():
        for pn, p in model.named_parameters():
            k = int(pfrac * p.numel())

            if (not "bias" in pn) and structured:
                _, indices = topk(
                    p.data,
                    k=k,
                    abs=False,
                    dim=0,
                )
            else:
                _, indices = topk(p.data.abs(), k=k, abs=False, dim=None)

            
            mask = torch.ones_like(p.data.flatten(), dtype=torch.bool)
            mask.index_fill_(0, indices, 1)
            mask = mask.view_as(p.data)
            p.data[~mask] = 0 

def calculate_mean_activations(model, loader, device):
    model.to(device)

    dummy_input, _ = next(iter(loader))
    dummy_input = dummy_input[0].to(device)
    acts = []

    with torch.no_grad():
        for module in model.chain():
            dummy_input = module(dummy_input)
            acts.append(torch.zeros_like(dummy_input))

        for X,_ in loader:
            X = X.to(device)
            for i, module in enumerate(model.chain()):
                X = module(X)
                acts[i] += X.sum(dim=0) 
    
    return [act/len(loader.dataset) for act in acts]

def load_dataset(ds_name):
    if ds_name is "mnist":
        transform=torch.transforms.Compose([
            torch.transforms.ToTensor(),
            torch.transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = torchvision.datasets.MNIST(root ="./data",train=True, transform=transform)
        testset  = torchvision.datasets.MNIST(root="./data",train=False, transform=transform)

    else:
        raise NotImplementedError
    
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, pin_memory=True,)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, pin_memory=True,)

    return testloader, trainloader

def train_model(
    model: nn.Module,
    lr: float,
    b1: float,
    b2: float,
    pfrac: float | None,
    ds_name: str,
    eps : float,
    epochs : int,
    device : str,
    seed: int = 0,
):
  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    all_params = list(model.parameters())
    betas = (b1, b2)

    optimizer = torch.optim.Adam(
        all_params,
        lr=lr,
        eps=eps,
        betas=betas,
        fused=True,
    )

    trainloader, testloader = load_dataset(ds_name)

    n_params = sum(p.numel() for p in model.parameters())
    n_params_wd = sum(p.numel() for p in model.parameters() if len(p.shape) > 1)

    print("n_params", n_params, "n_params_wd", n_params_wd)

    model.train()
    for epoch in range(epochs):
        for X, Y in trainloader:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, Y,)
            loss.backward()
            optimizer.step()

            if pfrac is not None:    
                apply_topk_(
                    model,
                    pfrac=pfrac,
                    structured=False,
                )

        if epoch%5:
            print(f'Train Accuracy : {acc(model, trainloader, device)}')
            print(f'Test  Accuracy : {acc(model, testloader, device)}')


            


def extract_circuit(model,
    lr: float,
    b1: float,
    b2: float,
    ds_name: str,
    eps : float,
    epochs : int,
    device : str,
    seed: int = 0,
):
  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
    trainloader, testloader = load_dataset(ds_name)
    inp_shape, _ = next(iter(trainloader))
    inp_shape = inp_shape[0]

    for p in model.parameters():
        p.requires_grad = False
    
    mean_activations = calculate_mean_activations(model, trainloader, device)
    circuit = inference.Circuit(model, inp_shape, mean_activations)

    all_params = [p for p in circuit.parameters() if p.requires_grad]
    betas = (b1, b2)

    optimizer = torch.optim.Adam(
        all_params,
        lr=lr,
        eps=eps,
        betas=betas,
        fused=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    
    print("n_params", n_params)

    model.train()
    for epoch in range(epochs):
        for X, Y in trainloader:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()
            logits = circuit(X)
            loss = F.cross_entropy(logits, Y,) + circuit.nonzero_params()
            loss.backward()
            optimizer.step()
            circuit.clamp_masks()

           
        if epoch%5:
            print(f'Train Accuracy : {acc(model, trainloader, device)}')
            print(f'Test  Accuracy : {acc(model, testloader, device)}')






