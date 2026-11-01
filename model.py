import torch, torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from data import FEAT_DIM

KP_DIM = FEAT_DIM  # keypoint features (576) + optional RGB embeddings


class SLT(nn.Module):
    """keypoints -> conv projector (4x temporal downsample) -> T5 encoder-decoder -> English text.

    Same idea as DeepMind's 'Scaling Sign Language Translation': a pretrained text encoder-decoder
    reads a sequence of skeleton features and generates the spoken-language sentence.
    """

    def __init__(self, name="t5-small", kp_dim=KP_DIM, random_init_cfg=None):
        super().__init__()
        if random_init_cfg is not None:  # only for smoke tests
            self.t5 = T5ForConditionalGeneration(random_init_cfg)
        else:
            self.t5 = T5ForConditionalGeneration.from_pretrained(name)
        d = self.t5.config.d_model
        self.proj_in = nn.Sequential(nn.LayerNorm(kp_dim), nn.Linear(kp_dim, d))
        self.conv = nn.Sequential(
            nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(d, d, 5, stride=2, padding=2), nn.GELU(),
        )
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(0.1)

    def projector_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("t5.")]

    def encode_inputs(self, kp, mask):
        x = self.proj_in(kp)                       # (B,T,d)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.drop(self.norm(x))
        m = mask[:, ::2][:, ::2]                   # matches conv length ceil(ceil(T/2)/2)
        return x, m[:, : x.shape[1]]

    def forward(self, kp, mask, labels=None):
        x, m = self.encode_inputs(kp, mask)
        return self.t5(inputs_embeds=x, attention_mask=m, labels=labels)

    @torch.no_grad()
    def translate(self, kp, mask, tok, num_beams=4, max_new_tokens=48):
        x, m = self.encode_inputs(kp, mask)
        ids = self.t5.generate(inputs_embeds=x, attention_mask=m,
                               num_beams=num_beams, max_new_tokens=max_new_tokens)
        return tok.batch_decode(ids, skip_special_tokens=True)