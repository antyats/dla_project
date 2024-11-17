import torch
from torch import amp

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, batch_idx, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()
        with amp.autocast(
            device_type=self.device,
            dtype=self.amp_float_type,
            enabled=self.amp_float_type is not None,
        ):
            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            if batch_idx % self.n_grad_accum_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    if (
                        type(self.lr_scheduler)
                        == torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.lr_scheduler.step(metrics=batch["loss"].item())
                    else:
                        self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            # self.log_spectrogram(**batch)
            # self.log_audio(**batch)
            pass
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_audio(**batch)

    def log_spectrogram(self, mix_spectrogram, target_spectrogram, **batch):
        spectrogram_for_plot = mix_spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("mix_spectrogram", image)

        spectrogram_for_plot = target_spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("target_spectrogram_1", image)

        spectrogram_for_plot = (
            target_spectrogram[self.config.dataloader.batch_size].detach().cpu()
        )
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("target_spectrogram_2", image)

    def log_audio(self, mix_audio, target_audio, output_audio, **batch):
        def _normalize_audio(audio: torch.Tensor):
            audio /= torch.max(torch.abs(audio))
            return audio.detach().cpu()

        audio = _normalize_audio(mix_audio[0])
        self.writer.add_audio(
            "mix_audio", audio.float(), sample_rate=self.config.writer.audio_sample_rate
        )

        audio = _normalize_audio(target_audio[0])
        self.writer.add_audio(
            "target_audio_1",
            audio.float(),
            sample_rate=self.config.writer.audio_sample_rate,
        )

        audio = _normalize_audio(target_audio[self.config.dataloader.batch_size])
        self.writer.add_audio(
            "target_audio_2",
            audio.float(),
            sample_rate=self.config.writer.audio_sample_rate,
        )

        audio = _normalize_audio(output_audio[0])
        self.writer.add_audio(
            "output_audio_1",
            audio.float(),
            sample_rate=self.config.writer.audio_sample_rate,
        )

        audio = _normalize_audio(output_audio[self.config.dataloader.batch_size])
        self.writer.add_audio(
            "output_audio_2",
            audio.float(),
            sample_rate=self.config.writer.audio_sample_rate,
        )
