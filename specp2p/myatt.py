import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import gl

# 与masa文件夹下的不同：这是只存针对第二张图的
# for unet
class AttControl:
    def __init__(self,num_steps=50,num_att_layer=16):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_steps = num_steps  
        self.num_att_layers = num_att_layer    # 多少组 [self, cross]

    def __call__(self, model,hidden_states, encoder_hidden_states, attention_mask=None, **kwargs):
        out = self.forward(model,hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        if True:
            self.cur_att_layer += 1
            if self.cur_att_layer == self.num_att_layers:
                self.cur_att_layer = 0
                self.cur_step += 1
                if self.cur_step == self.num_steps:         # my add
                    self.cur_step = 0           
        return out
        
# down,mid,up 层数:  6 1 9   (cross 也是)
class SelfAttControl(AttControl):
    def __init__(self,num_steps=50,num_att_layer=16):
        super().__init__(num_steps=num_steps,num_att_layer=num_att_layer)
        self.mode = None # or "edit" None  (None 表示普通的 不存也不用已经存的)
        self.kvs = [[None for j in range(self.num_att_layers)] for i in range(self.num_steps)]      # invert 时候的
        
    def forward(self, model,hidden_states, encoder_hidden_states, attention_mask=None, **kwargs):
        context = hidden_states
        q = model.to_q(context)
        # specp2p 只用invert模式
        if self.mode == "edit" and self.cur_step in self.step_idx and self.cur_att_layer in self.layer_idx:
            k,v = self.kvs[-1-self.cur_step][self.cur_att_layer]  # 记得改回来
            if k.shape[0] < q.shape[0]:  # invert 的时候没有uncond
                k,v = torch.cat([model.to_k(context[0:1]),k]), torch.cat([model.to_v(context[0:1]),v])
        else:
            k,v = model.to_k(context), model.to_v(context)
            if self.mode == "invert":       
                if False and self.cur_att_layer < 8:
                    pass
                else:
                    # self.kvs[self.cur_step][self.cur_att_layer] = (k.clone().detach(),v.clone().detach())
                    # 只存第二张图片的
                    if gl.invert_use_uncond:
                        k1 = torch.cat([k[1:2],k[3:4]],dim=0)   # uncond  cond 拼接
                        v1 = torch.cat([v[1:2],v[3:4]],dim=0)
                    else:
                        k1 = k[1:2]
                        v1 = v[1:2]
                    self.kvs[self.cur_step][self.cur_att_layer] = (k1.clone().detach(),v1.clone().detach())
                # pass
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=model.heads), (q, k, v))
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * model.scale 
        attn = attention_scores.softmax(dim=-1)
        hidden_states = torch.matmul(attn, v)
        hidden_states = rearrange(hidden_states, '(b h) n d -> b n (h d)', h=model.heads)   # my modify
        hidden_states = model.to_out[0](hidden_states)      # # linear proj
        hidden_states = model.to_out[1](hidden_states)      # dropout
        return hidden_states
    
    def set_mode(self,mode):
        assert mode is None or mode == "invert" or mode == "edit" 
        self.mode = mode
        
    def set_use_idx(self, step_idx = [], layer_idx = []):
        self.step_idx = step_idx
        self.layer_idx = layer_idx
    
    # 直接存到gl 多省事
    def to_gl(self):
        gl.speckvs = self.kvs

def _inj_forward_attention(self, editor):
    def forward(hidden_states, encoder_hidden_states, attention_mask=None):
        out = editor(self,hidden_states, encoder_hidden_states, attention_mask)
        return out        
    return forward
    
def register_attention(unet, editor, isself = True):
    for _name, _module in unet.named_modules():
        if _module.__class__.__name__ == "Attention":   # 0.8.0的命名是CrossAttention
            if (isself and 'attn1' in _name) or (not isself and 'attn2' in _name):
                # attn1: selfAttention  att2: crossAttention
                _module.forward = _inj_forward_attention(_module,editor) 