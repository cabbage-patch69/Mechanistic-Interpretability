import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, nc: int, nf: int, num_classes: int, inp_shape: list[int]):
        super(CNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=nf, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=nf, out_channels=2*nf, kernel_size=3, padding=1, bias=False),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
     
        dummy_input = torch.randn(1,*inp_shape)
        with torch.no_grad():
            inp_flat = self.encoder(dummy_input).numel()

        self.fc = nn.Linear(inp_flat, num_classes, bias=False)
        self.chain = nn.ModuleList([m for m in self.encoder.children()])
        self.chain.append(self.fc)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

class Prune(torch.autograd.Function):   
    @staticmethod
    def forward(ctx, mask_param, temperature):
        ctx.save_for_backward(mask_param)
        ctx.temperature = temperature
        return (mask_param > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        mask_param, = ctx.saved_tensors
        temp = ctx.temperature
        
        sig = torch.sigmoid(mask_param / temp)
        grad_sigmoid = (1 / temp) * sig * (1 - sig)
        return grad_output * grad_sigmoid, None

class Mask(nn.Module):
    def __init__(self, inp_shape: list[int], temperature: float, mean_act: torch.Tensor):
        super().__init__()
        self.mask = nn.Parameter(torch.normal(1, 0.1, size=inp_shape), requires_grad=True)
        self.temperature = temperature
        self.register_buffer('mean_act', mean_act.detach())

    def forward(self, x):
        mask = Prune.apply(self.mask, self.temperature)
        return x * mask #+ (1 - mask) * self.mean_act
    
    def clamp(self):
        with torch.no_grad():
            #the paper clamps between [-1,1]
            self.mask.clamp_(min=-1, max=1)
    
    def l0_loss(self):
        return torch.sum(Prune.apply(self.mask, self.temperature))

class Circuit(nn.Module):
    def __init__(self, model: nn.Module, inp_shape: list[int], mean_activations: list[torch.Tensor], temperature:float=0.3):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.mean_activations = mean_activations
       
        dummy_input = torch.randn(1, *inp_shape).to(next(model.parameters()).device)
        
        self.masks = nn.ModuleList([])

        self.temperature = temperature

        assert len(mean_activations) == len(model.chain)

        with torch.no_grad():
            for mean_act, module in zip(mean_activations, model.chain):
                dummy_input = module(dummy_input)
                self.masks.append(Mask(dummy_input.shape[1:], temperature=temperature, mean_act=mean_act))
 
    def forward(self, x):
        for mask, module in zip(self.masks, self.model.chain):
            x = mask(module(x))
        return x
    
    def clamp_masks(self):
        for mask in self.masks:
            mask.clamp()
    
    def total_l0_loss(self):
        return sum(m.l0_loss() for m in self.masks)
    
    def debug_stats(self):
        with torch.no_grad():
            avg_mask = torch.mean(torch.stack([m.mask.mean() for m in self.masks]))
            return avg_mask.item()

