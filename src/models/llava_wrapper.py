# models/llava_wrapper.py
import torch
import torch.nn as nn
from typing import Optional, List

class LLaVAWrapper(nn.Module):
    """
    Minimal wrapper to produce per-frame multimodal embeddings from a LLaVA model.
    Expected to return tensor (B, T, D_llava).

    Usage:
        w = LLaVAWrapper.from_pretrained(llava_path, out_dim=1024, freeze=True, device=device)
        feats = w(images, texts)  # images: (B, T, C, H, W)
    """

    def __init__(self, llava_model, out_dim: int = 1024, freeze: bool = True, device: Optional[torch.device] = None):
        super().__init__()
        self.llava = llava_model
        self.out_dim = out_dim
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        if freeze:
            for p in self.llava.parameters():
                p.requires_grad = False

    @classmethod
    def from_pretrained(cls, llava_path: str, out_dim: int = 1024, freeze: bool = True, device: Optional[torch.device] = None):
        """
        Try a couple of common ways to load LLaVA. If none succeed, raise a descriptive error.
        Replace the loading code below with your project's loader if needed.
        """
        load_errors = []
        try:
            # If your local llava repo exposes a load_model function
            import llava
            model = llava.load_model(llava_path)  # adjust if API differs
            return cls(model, out_dim=out_dim, freeze=freeze, device=device)
        except Exception as e:
            load_errors.append(str(e))
        try:
            # Fallback: try HF AutoModel (only works if checkpoint is HF-compatible)
            from transformers import AutoModel
            model = AutoModel.from_pretrained(llava_path, local_files_only=True)
            return cls(model, out_dim=out_dim, freeze=freeze, device=device)
        except Exception as e:
            load_errors.append(str(e))

        # If we reach here, we couldn't load LLaVA
        raise RuntimeError(
            "Could not load LLaVA from path '{}'.\n"
            "Tried local llava.load_model and transformers.AutoModel but failed.\n"
            "Please ensure you have a local llava loader or a HF compatible checkpoint.\n"
            "Errors: {}".format(llava_path, load_errors)
        )

    def forward(self, images: torch.Tensor, texts: Optional[List[str]] = None) -> torch.Tensor:
        """
        images: (B, T, C, H, W) or (B, C, H, W) — will be normalized/reshaped as needed.
        texts: Optional list[str] length B, passed to LLaVA if its API supports text conditioning.
        Returns: (B, T, D_llava)
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)
        B, T, C, H, W = images.shape
        images = images.to(self.device)

        # If llava model exposes a multimodal embedding function, use it.
        # The actual function name may differ. Replace these calls according to your local LLaVA.
        # We attempt several likely attributes; adjust if your LLaVA uses different names.
        if hasattr(self.llava, "get_multimodal_embeddings"):
            out = self.llava.get_multimodal_embeddings(images, texts)  # user replacement likely needed
            if out is None:
                raise RuntimeError("llava.get_multimodal_embeddings returned None")
            if out.dim() == 3 and out.shape[0] == B:
                return out.to(self.device)
            else:
                raise RuntimeError(f"Unexpected shape from get_multimodal_embeddings: {out.shape}")

        # HF-like AutoModel: use vision tower (if exists) and pool per-frame features
        if hasattr(self.llava, "vision_tower") or hasattr(self.llava, "vision_encoder") or hasattr(self.llava, "vision_model"):
            # Try to find a vision encoder attribute
            enc = None
            for name in ("vision_tower", "vision_encoder", "vision_model"):
                if hasattr(self.llava, name):
                    enc = getattr(self.llava, name)
                    break
            if enc is None:
                raise RuntimeError("Found llava object but no vision encoder attribute. Please adapt wrapper.")
            # flatten frames into batch
            flat = images.view(B * T, C, H, W)
            feats = enc(flat)  # shape depends on encoder
            # pool spatial dims if present
            if feats.dim() == 4:
                feats = feats.mean(dim=[2, 3])  # (B*T, Dfeat)
            feats = feats.view(B, T, -1)
            # optionally project to out_dim if different
            if feats.shape[-1] != self.out_dim:
                # lightweight linear projection
                proj = nn.Linear(feats.shape[-1], self.out_dim).to(self.device)
                feats = proj(feats)
            return feats.to(self.device)

        raise RuntimeError("LLaVA model present but wrapper cannot find a known embedding method. "
                           "Please modify models/llava_wrapper.py to call your local LLaVA API.")
