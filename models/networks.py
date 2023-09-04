from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.core.clip import build_model, load_clip, tokenize

# positional embedding with sin/cos
class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out

def get_embedder(multires, input_dim=3):
    if multires < 0:
        return nn.Identity(), input_dim

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim

    
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, embed_dim, max_len=80):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        po = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.) / embed_dim))
        pe[:, 0::2] = torch.sin(po * div_term)
        pe[:, 1::2] = torch.cos(po * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        l, _, _ = x.shape
        pos = self.pe[:l, :].unsqueeze(1)
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, embed_dim, max_len=80, padding_idx=0):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, embed_dim, padding_idx)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, x):
        l, _, _ = x.shape
        idx = torch.arange(l, device=x.device)
        pos = self.pos_embed(idx).unsqueeze(1)
        return pos

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CrossResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = CrossModalAttention(embed_dim=d_model, num_heads=n_head, output_dim=d_model)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        attn_output, attn_weights = self.attn(q=q, k=k, v=v, attn_mask=self.attn_mask)
        return attn_output, attn_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        attn_output, attn_weights = self.attention(self.ln_1(q), self.ln_1(k), self.ln_1(v))
        q = q + attn_output
        q = q + self.mlp(self.ln_2(q))
        return q, attn_weights

# multi layer
class CrossTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([CrossResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        for i, _ in enumerate(self.resblocks):
            q, attn_weights = self.resblocks[i](q, k, v)

        q = q.permute(1, 0, 2) # L'ND -> NL'D
        return q, attn_weights

# one layer without shortcut: naivest cross attention
class CrossModalAttention(nn.Module):
    """ Cross-Modal Attention. Adapted from: https://github.com/openai/CLIP/blob/main/clip/model.py#L56 """

    def __init__(self, embed_dim=1024, num_heads=32, output_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, q, k, v, attn_mask=None):
        x, attn_weights = F.multi_head_attention_forward(
            query=q, key=k, value=v,
            embed_dim_to_check=v.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            need_weights=True,
            attn_mask=attn_mask
        )
        
        return x, attn_weights
        
class CLIPGraspFusion(nn.Module):
    def __init__(self, grasp_dim, width, layers, heads, device):
        super().__init__()
        
        self.device = device
        self._load_clip()
        
        self.cross_attn = CrossTransformer(width=width, layers=layers, heads=heads)

        self.grasp_embbedding = nn.Sequential(
                                nn.Linear(grasp_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                )

        self.pos_projection, pos_proj_dim = get_embedder(multires=5, input_dim=3)

        self.bbox_pos_embbedding = nn.Sequential(
                                nn.Linear(pos_proj_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, width),
                                nn.ReLU(),
                                nn.Linear(width, width)
                                )
                            

    def _load_clip(self): # patch_size = 32, vision_width = 768, vision_layers = 12, vision_heads = 12, output_dim = 512, vocab_size = 49408
        model, _ = load_clip("ViT-B/32", device=self.device)
        self.clip = build_model(model.state_dict()).to(self.device)
        del model

    def encode_bbox(self, x):
        with torch.no_grad():
            if x.shape[0] == 1:
                bboxs = x[0]
                padding = nn.ZeroPad2d(bboxs[0].shape[1] * 3)
                bboxs = padding(bboxs)
                bbox_feat = self.clip.encode_image(bboxs.to(self.device))
                bbox_feat = bbox_feat.unsqueeze(0)
        return bbox_feat
    
    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self.device)
            text_feat = self.clip.encode_text(tokens)
            text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_mask

    def encode_grasp(self, x):
        grasp_emb = self.grasp_embbedding(x.to(self.device)) # shape = [N, L', D]
        return grasp_emb

    def encode_bbox_pos(self, x):
        bbox_pos_emb = self.bbox_pos_embbedding(x.to(self.device)) # shape = [N, L', D]
        return bbox_pos_emb
        
    def mult_fusion(self, bbox_feat, text_feat):
        text_feat = text_feat.unsqueeze(-2)
        text_feat = text_feat.repeat(1, bbox_feat.shape[1], 1)
        fusion_feat = bbox_feat * text_feat
        return fusion_feat
    
    def forward(self, bboxes, pos_bboxes, text, actions):

        # encode bbox
        bbox_feat = self.encode_bbox(bboxes) # shape = [N, L, D] D=512
        # encode text
        text_feat, _ = self.encode_text(text) # shape = [N, D]
        # fusion
        fusion_feat = self.mult_fusion(bbox_feat, text_feat) # shape = [N, L, D]
        
        # normalized features   
        bbox_feat_normlized = bbox_feat / bbox_feat.norm(dim=-1, keepdim=True)
        text_feat_normlized = text_feat / text_feat.norm(dim=-1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * bbox_feat_normlized @ text_feat_normlized.t()

        # logits_per_text = logit_scale * text_feat_normlized @ bbox_feat_normlized.t()
        probs = logits_per_image.softmax(dim=-2).reshape(logits_per_image.shape[0], -1)
        
        # encode grasp
        grasp_feat = self.encode_grasp(actions) # shape = [N, L', D]
        grasp_feat = grasp_feat.permute(1, 0, 2)  # NL'D -> L'ND
        fusion_feat = fusion_feat.float().permute(1, 0, 2)  # NLD -> LND

        # encode bbox positions
        pos_bboxes = self.pos_projection(pos_bboxes)
        bbox_pos_feat = self.encode_bbox_pos(pos_bboxes) # shape = [N, L, D]
        # bbox_pos_feat = bbox_pos_feat.permute(1, 0, 2) # NLD -> LND

        # add fusion
        bbox_compound_feat = bbox_pos_feat + bbox_feat
        # concat fusion
        # bbox_compound_feat = torch.cat((bbox_feat, bbox_pos_feat), dim=-1)
        # bbox_compound_feat = self.pos_vision_fusion(bbox_compound_feat)
        bbox_compound_feat = bbox_compound_feat.permute(1, 0, 2)

        # cross attention
        cross_feat, attn_weights = self.cross_attn(q=grasp_feat, k=bbox_compound_feat, v=fusion_feat) # shape = [N, L', D]
        
        return cross_feat, probs, attn_weights


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# Q Networks for RL algorithms
class QNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, sa):
        
        x1 = F.relu(self.linear1(sa))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(sa))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

# Policy Network for RL algorithms
class Policy(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x).squeeze()
        return logits
