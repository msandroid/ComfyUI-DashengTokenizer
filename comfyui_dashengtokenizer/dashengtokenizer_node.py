"""
ComfyUI nodes for mispeech/dashengtokenizer (Hugging Face).
Audio encode/decode via transformers AutoModel; no direct edit of third-party code.
"""

from __future__ import annotations

import torch
from transformers import AutoModel

try:
    import librosa
    import numpy as np
except ImportError as e:
    raise ImportError(
        "comfyui_dashengtokenizer requires librosa and numpy. Install with: pip install -r requirements.txt"
    ) from e

DASHENG_SAMPLE_RATE = 16000
DEFAULT_MODEL_ID = "mispeech/dashengtokenizer"


class DashengModelHolder:
    """Holds the loaded DashengTokenizer model for use by Encode/Decode nodes."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device


def _audio_to_16k_mono(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Convert ComfyUI AUDIO waveform [B, C, T] to (B, T) mono at 16kHz on CPU."""
    if waveform.dim() == 3:
        waveform = waveform.mean(dim=1, keepdim=False)  # (B, T)
    elif waveform.dim() == 2:
        if waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)
        else:
            waveform = waveform.unsqueeze(0)  # (1, T)
    else:
        waveform = waveform.unsqueeze(0)
    if sample_rate != DASHENG_SAMPLE_RATE:
        wav_np = waveform.cpu().float().numpy()
        if wav_np.ndim == 1:
            wav_16k = librosa.resample(
                wav_np, orig_sr=sample_rate, target_sr=DASHENG_SAMPLE_RATE, res_type="kaiser_fast"
            )
            waveform = torch.from_numpy(wav_16k).float()
        else:
            rows = [
                librosa.resample(
                    wav_np[i], orig_sr=sample_rate, target_sr=DASHENG_SAMPLE_RATE, res_type="kaiser_fast"
                )
                for i in range(wav_np.shape[0])
            ]
            waveform = torch.from_numpy(np.stack(rows)).float()
    return waveform


class DashengTokenizerLoadModel:
    """Load DashengTokenizer model from Hugging Face. Output connects to Encode/Decode nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "model_id": ("STRING", {"default": DEFAULT_MODEL_ID}),
            },
        }

    RETURN_TYPES = ("DASHENG_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/dashengtokenizer"

    def load_model(self, model_id: str = DEFAULT_MODEL_ID):
        model_id = (model_id or DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model = model.to(device)
        model.eval()
        return (DashengModelHolder(model, device),)


class DashengTokenizerEncode:
    """Encode audio to DashengTokenizer embeddings. Input audio is resampled to 16kHz mono."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DASHENG_MODEL",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("DASHENG_EMBEDDINGS",)
    RETURN_NAMES = ("embeddings",)
    FUNCTION = "encode"
    CATEGORY = "audio/dashengtokenizer"

    def encode(self, model: DashengModelHolder, audio: dict):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if waveform.dim() == 3:
            w = waveform  # [B, C, T]
        else:
            w = waveform.unsqueeze(0).unsqueeze(0) if waveform.dim() == 1 else waveform.unsqueeze(0)
        audio_16k = _audio_to_16k_mono(w, sample_rate)  # (B, T)
        audio_16k = audio_16k.to(model.device)
        with torch.no_grad(), torch.autocast(device_type="cuda" if model.device.type == "cuda" else "cpu"):
            embeddings = model.model.encode(audio_16k)
        return ({"embeddings": embeddings.cpu(), "sample_rate": DASHENG_SAMPLE_RATE},)


class DashengTokenizerDecode:
    """Decode DashengTokenizer embeddings back to audio (16kHz mono)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DASHENG_MODEL",),
                "embeddings": ("DASHENG_EMBEDDINGS",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "audio/dashengtokenizer"

    def decode(self, model: DashengModelHolder, embeddings: dict):
        emb = embeddings["embeddings"].to(model.device)
        with torch.no_grad(), torch.autocast(device_type="cuda" if model.device.type == "cuda" else "cpu"):
            reconstructed = model.model.decode(emb)
        if reconstructed.dim() == 1:
            reconstructed = reconstructed.unsqueeze(0)
        if reconstructed.dim() == 2:
            reconstructed = reconstructed.unsqueeze(1)  # (B, 1, T)
        waveform = reconstructed.cpu().float()
        return ({"waveform": waveform, "sample_rate": DASHENG_SAMPLE_RATE},)
