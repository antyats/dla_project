import torch
from torchmetrics.functional import signal_distortion_ratio

from src.metrics.base_metric import BaseMetric


class SDRi(BaseMetric):
    def __init__(
        self, name: str | None = None, reduction: str = "mean", *args, **kwargs
    ) -> None:
        """
        SDRi metric class.

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
        SDRi calculation logic.

        Args:
            input (Tensor): model audio inputs (unseparated audio). Shape: (..., time)
            output (Tensor): model output predictions (separated audio). Shape: (..., time)
            target (Tensor): ground-truth audio. Shape: (..., time)
        Returns:
            metric (Tensor): calculated SDRi.
        """
        sdr_input = signal_distortion_ratio(input, target)
        sdr_output = signal_distortion_ratio(output, target)
        result = sdr_output - sdr_input

        if self.reduction == "mean":
            return result.mean()
        if self.reduction == "sum":
            return result.sum()
        if self.reduction == "none":
            return result
