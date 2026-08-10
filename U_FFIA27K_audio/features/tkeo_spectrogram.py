import torch
import torch.nn as nn
import torch.nn.functional as F


class TKEOAdaptiveSpectrogram(nn.Module):
    """
    STFT frontend with TKEO adaptive pre-emphasis applied after framing
    and before windowing, matching the TKEO.py processing order.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window: str = "hann",
        center: bool = True,
        pad_mode: str = "reflect",
        alpha_max: float = 0.99,
        beta: float = 0.8,
    ) -> None:
        super().__init__()
        if window != "hann":
            raise ValueError("TKEOAdaptiveSpectrogram currently supports only the hann window.")
        if win_length > n_fft:
            raise ValueError("win_length must be less than or equal to n_fft.")

        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.center = bool(center)
        self.pad_mode = pad_mode
        self.alpha_max = float(alpha_max)
        self.beta = float(beta)

        fft_window = torch.hann_window(self.win_length, periodic=True)
        if self.win_length < self.n_fft:
            left = (self.n_fft - self.win_length) // 2
            right = self.n_fft - self.win_length - left
            fft_window = F.pad(fft_window, (left, right))
        self.register_buffer("fft_window", fft_window)

    def _tkeo_adaptive_preemphasis(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [batch, frames, frame_length]
        batch_size, num_frames, frame_length = frames.shape
        filtered_frames = torch.empty_like(frames)
        alpha_prev = torch.zeros(batch_size, dtype=frames.dtype, device=frames.device)

        for frame_idx in range(num_frames):
            frame = frames[:, frame_idx, :]

            psi = torch.zeros_like(frame)
            if frame_length > 2:
                psi[:, 1:-1] = frame[:, 1:-1].pow(2) - frame[:, :-2] * frame[:, 2:]
                psi[:, 0] = psi[:, 1]
                psi[:, frame_length - 1] = psi[:, frame_length - 2]

            mean_psi = psi.abs().mean(dim=1)
            mean_energy = frame.pow(2).mean(dim=1)
            ctrl_signal = torch.where(
                mean_energy < 1e-10,
                torch.zeros_like(mean_energy),
                mean_psi / mean_energy,
            )

            alpha_raw = self.alpha_max * (1.0 - torch.exp(-ctrl_signal))
            alpha = self.beta * alpha_prev + (1.0 - self.beta) * alpha_raw
            alpha = torch.clamp(alpha, min=0.1, max=self.alpha_max)
            alpha_prev = alpha

            filtered = torch.empty_like(frame)
            filtered[:, 0] = frame[:, 0]
            filtered[:, 1:] = frame[:, 1:] - alpha[:, None] * frame[:, :-1]
            filtered_frames[:, frame_idx, :] = filtered

        return filtered_frames

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim != 2:
            raise ValueError(f"Expected raw waveform tensor [batch, samples], got {tuple(input_tensor.shape)}")

        x = input_tensor
        if self.center:
            pad = self.n_fft // 2
            x = F.pad(x, (pad, pad), mode=self.pad_mode)

        frames = x.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)
        frames = self._tkeo_adaptive_preemphasis(frames)

        window = self.fft_window.to(dtype=frames.dtype, device=frames.device)
        windowed_frames = frames * window
        stft_matrix = torch.fft.rfft(windowed_frames, n=self.n_fft, dim=-1)
        power_spectrogram = stft_matrix.real.pow(2) + stft_matrix.imag.pow(2)

        return power_spectrogram.unsqueeze(1)
