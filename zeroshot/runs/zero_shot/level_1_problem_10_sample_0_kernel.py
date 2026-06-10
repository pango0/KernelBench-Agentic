import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class CustomMatMulFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return torch.matmul(A, B)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = torch.matmul(grad_output, B.transpose(-2, -1))
        grad_B = torch.matmul(A.transpose(-2, -1), grad_output)
        return grad_A, grad_B

class ModelNew(nn.Module):
    """
    Optimized version using custom CUDA operators for 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication using custom CUDA operators.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return CustomMatMulFunction.apply(A, B)

# Example usage
N = 16
M = 1024
K = 2048
L = 768

def get_inputs():
    A = torch.rand(N, M, K)
    B = torch.rand(K, L)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed