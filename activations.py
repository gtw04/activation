from torch import Tensor
from torch.nn.modules import Module

class TanhExp(Module):
    """
    Xinyu Liu, Xiaoguang Di
    TanhExp: A Smooth Activation Function
    with High Convergence Speed for
    Lightweight Neural Networks
    https://arxiv.org/pdf/2003.09855v1.pdf
    """
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(torch.exp(x))