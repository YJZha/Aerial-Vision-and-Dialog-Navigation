# models/llava_wrapper.py
import os
import torch
import torch.nn as nn
from transformers import CLIPModel, AutoTokenizer, AutoModel, AutoConfig

class LLaVAWrapper(nn.Module):
    """
    Minimal LLaVA-like wrapper:
    - Loads a CLIPModel for vision features (vision_model)
    - Loads a text model (AutoModel) for text embeddings
    - fuse(images_tensor, input_ids, attention_mask) -> fused vector (batch, hidden_dim)
    NOTE: expects local files under llava_dir for model names you pass; set local_files_only=True
    """
    def __init__(self, clip_model_name_or_path, text_model_name_or_path, device='cpu'):
        super().__init__()
        # CLIP model for vision
        self.device = torch.device(device)
        # load CLIP
        self.clip = CLIPModel.from_pretrained(clip_model_name_or_path, local_files_only=True).to(self.device)
        # text encoder (AutoModel) - uses pooler_output if available, else last_hidden_state[:,0,:]
        self.text_model = AutoModel.from_pretrained(text_model_name_or_path, local_files_only=True).to(self.device)
        # text tokenizer should be loaded by caller if needed (we don't wrap tokenizer here)
        self.text_hidden_size = self.text_model.config.hidden_size if hasattr(self.text_model, "config") else 768
        # clip vision feature size
        # CLIPModel has vision_model.config.hidden_size
        self.vision_hidden_size = self.clip.config.vision_config.hidden_size if hasattr(self.clip.config, "vision_config") else getattr(self.clip.config, "hidden_size", 768)

    @classmethod
    def from_pretrained(cls, llava_dir, clip_name=None, text_name=None, device='cpu'):
        """
        llava_dir: root path you provided (contains subfolders or model names)
        clip_name/text_name: if None, defaults to typical dir names inside llava_dir
        """
        if clip_name is None:
            clip_name = os.path.join(llava_dir, "clip-vit-large-patch14-336")
        if text_name is None:
            text_name = os.path.join(llava_dir, "vicuna-7b-v1.5")
        return cls(clip_name, text_name, device=device)

    def forward(self, images_tensor, input_ids=None, attention_mask=None):
        """
        images_tensor: float tensor (B, C, H, W) expected in pixel values (0..255 or preprocessed)
        input_ids, attention_mask: optionally provide text tokens (already tokenized by proper tokenizer)
        Returns: fused tensor (B, hidden_dim) where hidden_dim = vision_hidden + text_hidden
        """
        device = self.device
        images_tensor = images_tensor.to(device)
        # CLIP vision forward
        with torch.no_grad():
            clip_out = self.clip.vision_model(pixel_values=images_tensor, return_dict=True)
            # many CLIP variants: try pooled output if exists, else mean pool last_hidden_state
            if hasattr(clip_out, "pooler_output") and clip_out.pooler_output is not None:
                vis_feat = clip_out.pooler_output  # (B, vision_hidden)
            else:
                # last_hidden_state: (B, seq_len, dim) -> mean pool
                vis_feat = clip_out.last_hidden_state.mean(dim=1)
        # Text forward
        if input_ids is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
            with torch.no_grad():
                txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                if hasattr(txt_out, "pooler_output") and txt_out.pooler_output is not None:
                    txt_feat = txt_out.pooler_output
                else:
                    txt_feat = txt_out.last_hidden_state[:, 0, :]
        else:
            # if no text provided, zero vector
            txt_feat = torch.zeros((vis_feat.size(0), self.text_hidden_size), device=device)

        fused = torch.cat([vis_feat, txt_feat], dim=1)  # (B, vision_hidden + text_hidden)
        return fused
