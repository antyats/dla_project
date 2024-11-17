import torch
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

from src.metrics.base_metric import BaseMetric


class PESQ(BaseMetric):
    def __init__(
        self,
        sample_rate: int,
        name: str | None = None,
        reduction: str = "mean",
        mode: str = "wb",
        keep_same_device: bool = False,
        n_processes: int = 1,
        *args,
        **kwargs,
    ) -> None:
        """
        PESQ metric class.

        Args:
            sample_rate (int): sample rate of input and target audiofiles
            name (str | None): metric name to use in logger and writer.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                * 'none': no reduction will be applied
                * 'mean': the weighted mean of the output is taken
                * 'sum': the output will be summed.
            mode (str): 'wb' | 'nb'. Whether to use wb (wideband, 50-7000 Hz) or nb (narrowband,  300-3400 Hz) mode.
            keep_same_device (bool): whether to move metric results to the device of input tensors (metric is computed on CPU)
            n_processes (int): number of processes to use for metric calculations
        """
        super().__init__(name, *args, **kwargs)
        self.reduction = reduction
        self.sample_rate = sample_rate
        self.mode = mode
        self.keep_same_device = keep_same_device
        self.n_processes = n_processes

    def __call__(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        PESQ calculation logic.

        Args:
            output (Tensor): model output predictions (separated audio). Shape: (..., time)
            target (Tensor): ground-truth audio. Shape: (..., time)
        Returns:
            metric (Tensor): calculated SDRi.
        """
        pesq = perceptual_evaluation_speech_quality(
            output,
            target,
            fs=self.sample_rate,
            mode=self.mode,
            keep_same_device=self.keep_same_device,
            n_processes=self.n_processes,
        )

        if self.reduction == "mean":
            return pesq.mean()
        if self.reduction == "sum":
            return pesq.sum()
        if self.reduction == "none":
            return pesq
