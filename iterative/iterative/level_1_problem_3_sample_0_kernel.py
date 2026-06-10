import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class BatchMatMulFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return torch.bmm(A, B)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = torch.bmm(grad_output, B.transpose(1, 2))
        grad_B = torch.bmm(A.transpose(1, 2), grad_output)
        return grad_A, grad_B

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return BatchMatMulFunction.apply(A, B)

# Example usage
batch_size = 128
m = 128 * 4
k = 256 * 4
n = 512 * 4

def get_inputs():
    A = torch.rand(batch_size, m, k)
    B = torch.rand(batch_size, k, n)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed