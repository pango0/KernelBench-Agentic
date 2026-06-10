import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class MatrixScalarMulFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, A, s):
        ctx.save_for_backward(A, s)
        return A * s

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        A, s = ctx.saved_tensors
        grad_A = grad_output * s
        grad_s = grad_output.sum()
        return grad_A, grad_s

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operators for matrix-scalar multiplication.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication using custom CUDA operator.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return MatrixScalarMulFunction.apply(A, s)

# Example usage
M = 16384 * 4
N = 4096 * 4

def get_inputs():
    A = torch.rand(M, N)
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []  # No special initialization inputs needed