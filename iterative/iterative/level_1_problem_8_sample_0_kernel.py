import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class MatMulFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        C = torch.matmul(A, B)
        return C

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = torch.matmul(grad_output, B.t())
        grad_B = torch.matmul(A.t(), grad_output)
        return grad_A, grad_B

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operators for matrix multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B using custom CUDA operators.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        return MatMulFunction.apply(A, B)

# Example usage
M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed