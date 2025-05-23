import torch
import torch.nn as nn
from typing import Optional
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F

class LoRALayer(nn.Module):
    """低秩适配层（A/B矩阵）"""
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=False)
        
        # 初始化：A用高斯分布，B初始为0
        nn.init.normal_(self.down.weight, std=1/torch.sqrt(torch.tensor(rank)))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)
class MyLoraAttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self,hidden_size,rank=4,scale=0.2):
        super().__init__()
        
        self.scale = scale
        self.rank = rank
        self.to_q_lora = LoRALayer(hidden_size, hidden_size, self.rank)
        self.to_k_lora = LoRALayer(hidden_size, hidden_size, self.rank)
        self.to_v_lora = LoRALayer(hidden_size, hidden_size, self.rank)
        self.to_out_lora = LoRALayer(hidden_size, hidden_size, self.rank)
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        # 首次运行时初始化LoRA层
        # if self.to_q_lora is None:
        #     self._init_lora(hidden_states.shape[-1])
        
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + self.scale * self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + self.scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + self.scale * self.to_v_lora(encoder_hidden_states)

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + self.scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



class LoRAAttnProcessor2_0(nn.Module):
    """
    替换原始AttnProcessor2_0，注入LoRA到Q/K/V投影层
    保持与原始处理器相同的接口，兼容Diffusers库
    """
    def __init__(
        self, 
        original_processor: nn.Module, 
        rank: int = 8, 
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        self.original_processor = original_processor  # 原始处理器（冻结）
        self.hidden_size = None  # 延迟初始化
        
        # LoRA配置
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scale = lora_alpha / rank
        
        # 初始化LoRA层（延迟到第一次forward时确定维度）
        self.lora_q = None
        self.lora_k = None
        self.lora_v = None
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # 冻结原始处理器
        for param in original_processor.parameters():
            param.requires_grad_(False)

    def _init_lora(self, hidden_size: int):
        """延迟初始化LoRA层（因hidden_size在forward时才能确定）"""
        self.lora_q = LoRALayer(hidden_size, hidden_size, self.rank, self.scale)
        self.lora_k = LoRALayer(hidden_size, hidden_size, self.rank, self.scale)
        self.lora_v = LoRALayer(hidden_size, hidden_size, self.rank, self.scale)
        
    def __call__(
        self, 
        attn,  # Attention模块
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        # 首次运行时初始化LoRA层
        if self.lora_q is None:
            self._init_lora(hidden_states.shape[-1])
        
        # 获取原始Q/K/V计算
        batch_size, sequence_length, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states if encoder_hidden_states is None else encoder_hidden_states)
        value = attn.to_v(hidden_states if encoder_hidden_states is None else encoder_hidden_states)
        
        # 注入LoRA
        query = query + self.lora_dropout(self.lora_q(hidden_states))
        key = key + self.lora_dropout(self.lora_k(hidden_states))
        value = value + self.lora_dropout(self.lora_v(hidden_states))
        
        # 剩余处理逻辑与原始AttnProcessor2_0一致
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        return hidden_states

