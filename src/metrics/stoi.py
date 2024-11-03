import torch
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    def __init__(
        self,
        sample_rate: int,
        name: str | None = None,
        reduction: str = "mean",
        keep_same_device: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        STOI metric class.

        Args:
            sample_rate (int): sample rate of input and target audiofiles
            name (str | None): metric name to use in logger and writer.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'
                * 'none': no reduction will be applied
                * 'mean': the weighted mean of the output is taken
                * 'sum': the output will be summed.
            keep_same_device (bool): whether to move metric results to the device of input tensors (metric is computed on CPU)
        """
        super().__init__(name, *args, **kwargs)
        self.reduction = reduction
        self.sample_rate = sample_rate
        self.keep_same_device = keep_same_device

    def __call__(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        STOI calculation logic.

        Args:
            output (Tensor): model output predictions (separated audio). Shape: (..., time)
            target (Tensor): ground-truth audio. Shape: (..., time)
        Returns:
            metric (Tensor): calculated STOI.
        """
        stoi = short_time_objective_intelligibility(
            output,
            target,
            fs=self.sample_rate,
            keep_same_device=self.keep_same_device,
        )

        if self.reduction == "mean":
            return stoi.mean()
        if self.reduction == "sum":
            return stoi.sum()
        if self.reduction == "none":
            return stoi
