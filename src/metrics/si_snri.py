import torch
from torchmetrics.functional import scale_invariant_signal_noise_ratio

from src.metrics.base_metric import BaseMetric


class SI_SNRi(BaseMetric):
    def __init__(
        self, name: str | None = None, reduction: str = "mean", *args, **kwargs
    ) -> None:
        """
        SI-SNR metric class.

        Args:
            name (str | None): metric name to use in logger and writer.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                * 'none': no reduction will be applied
                * 'mean': the weighted mean of the output is taken
                * 'sum': the output will be summed.
        """
        super().__init__(name, *args, **kwargs)
        self.reduction = reduction

    def __call__(
        self, input: torch.Tensor, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        SI-SNR calculation logic.

        Args:
            input (Tensor): model audio inputs (unseparated audio). Shape: (..., time)
            output (Tensor): model output predictions (separated audio). Shape: (..., time)
            target (Tensor): ground-truth audio. Shape: (..., time)
        Returns:
            metric (Tensor): calculated SI-SNR.
        """
        sisnr_input = scale_invariant_signal_noise_ratio(input, target)
        sisnr_output = scale_invariant_signal_noise_ratio(output, target)
        result = sisnr_output - sisnr_input

        if self.reduction == "mean":
            return result.mean()
        if self.reduction == "sum":
            return result.sum()
        if self.reduction == "none":
            return result
