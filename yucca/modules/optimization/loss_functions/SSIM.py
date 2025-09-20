from torch import Tensor
from kornia.losses import SSIMLoss


class SSIM(SSIMLoss):
    def __init__(self):
        super().__init__(window_size=11)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target)
