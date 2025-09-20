from torch import nn, Tensor


class SSIM(nn.SSIMLoss):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target)
