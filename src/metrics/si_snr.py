import torch

from src.metrics.base_metric import BaseMetric


class SI_SNR(BaseMetric):
    def __init__(self, reduction: str = "mean", *args, **kwargs) -> None:
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            reduction (str | None): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                * 'none': no reduction will be applied,
                * 'mean': mean of the output is taken,
                * 'sum': the output will be summed.
        """
        super().__init__(*args, **kwargs)
        self.reduction = reduction

    def __call__(self, output: torch.Tensor, target: torch.Tensor, **kwargs):
        """
        SI-SNR calculation logic.

        Args:
            output (Tensor): model output predictions. Shape: (batch, time)
            target (Tensor): ground-truth audio. Shape: (batch, time)
        Returns:
            metric (float): calculated SI-SNR.
        """
        output = output - output.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        coef = (output * target).sum(dim=-1, keepdim=True) / (target**2).sum(
            dim=-1, keepdim=True
        )
        lg = ((coef * target) ** 2).sum(dim=-1, keepdim=True) / (
            (output - target) ** 2
        ).sum(dim=-1, keepdim=True)
        return 10 * torch.log10(lg)
