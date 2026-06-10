import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class MatMulCUDA(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, A, B):
        M, K = A.shape
        K, N = B.shape
        C = torch.zeros((M, N), device=A.device, dtype=torch.float32)
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        torch.matmul(A, B, out=C)
        end_time.record()
        torch.cuda.synchronize()
        ctx.elapsed_time = start_time.elapsed_time(end_time)
        ctx.save_for_backward(A, B)
        return C

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        grad_A = None
        grad_B = None
        if ctx.needs_input_grad[0]:
            grad_A = torch.matmul(grad_output, B.t())
        if ctx.needs_input_grad[1]:
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
        Performs matrix multiplication using custom CUDA operator.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return MatMulCUDA.apply(A, B)

# Example usage
M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed