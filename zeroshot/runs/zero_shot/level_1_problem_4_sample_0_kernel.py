import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

class MatMulBwd(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, grad_output, input1, input2):
        ctx.save_for_backward(input1, input2)
        return torch.matmul(grad_output, input2.t())

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = torch.matmul(grad_grad_output, input2)
        grad_input2 = torch.matmul(input1.t(), grad_grad_output)
        return grad_input1, grad_input2, None

class CustomMatMul(nn.Module):
    def __init__(self):
        super(CustomMatMul, self).__init__()

    def forward(self, A, B):
        return MatMulBwd.apply(B, A, B)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operators.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_matmul = CustomMatMul()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return self.custom_matmul(A, B)

M = 256 * 8  # 2048
K = 131072 * 8  # 1048576

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, 1)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed