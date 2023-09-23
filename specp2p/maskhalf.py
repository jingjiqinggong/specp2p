import os
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torchvision.utils import save_image
import gl
from PIL import Image
from p2p.ptp_utils import get_word_inds

# p2p self attmap replace 图内相对结构的保留    cross attmap replace 是非编辑区域结构的保留  不信试试猫照镜子
# 用这个forward
# 智能抠图 去背景 再得到mask
# 前景只跟前景算相似度、output  背景只跟背景算相似度、output
# sim_mask  用mask_s  
# output
class AttHalfMutualMask():
    def __init__(self,thres=0.1, mask_s=None, target_token_idx=[1], mask_save_dir=None,speckvs=None,useuncond=True,
                 polatemode="nearest",
                 source_token_idx = None):
        """
        Args:
            thres: the thereshold for mask thresholding
            mask_s: source mask with shape (h, w)
            target_token_idx: the token index list for cross-attention map aggregation
            mask_save_dir: the path to save the mask image
            speckvs:  对象原图 inversion时候存下来的self kv
        """
        self.thres = thres
        self.mask_s = mask_s  # source mask with shape (h, w)
        self.target_token_idx = target_token_idx
        self.source_token_idx = source_token_idx
        
        self.cross_attns = []

        self.mask_save_dir = mask_save_dir
        if self.mask_save_dir is not None:
            os.makedirs(self.mask_save_dir, exist_ok=True)
        self.speckvs = speckvs
        self.useuncond = useuncond
        self.polatemode = polatemode
            
    def aggregate_cross_attn_map(self, idx):
        # self.cross_attns[0].shape[0]  [4,256,77]
        # eg stack: [25,4, 256, 77]
        attn_map = torch.stack(self.cross_attns, dim=0).mean(0)  # (layers,B, N_spatial, N_sentence) -> (B, N_spatial, N_sentence)
        B = attn_map.shape[0]
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]  #[4,16,16,len(idx)]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image

    # mask [H,W]  插值为 
    def deal_mask(self,mask,H,W,imgname=None): 
        # mode: nearest(default)  linear  bilinear(ok) bicubic trilinear area
        # mode用 bilinear 或者 bicubic吧, 或者像localblend那样先池化    别用nearest！
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (H, W),mode=self.polatemode)
        if self.mask_save_dir is not None and imgname is not None:
            mask_image1 = mask.clone()  #.reshape(1,1,H, W)
        mask[mask >= self.thres] = 1
        mask[mask < self.thres] = 0
        if self.mask_save_dir is not None and imgname is not None:
            mask_image2 = mask  #.reshape(1,1,H, W)
            save_image(torch.cat([mask_image1,mask_image2]), os.path.join(self.mask_save_dir,imgname))
        
        # 出去记得reshape呀
        return mask
            
    def __call__(self, q, k, v, heads,scale,cur_step, cur_layer,**kwargs):
        """
        Args:
            q,k,v: target的
        k1, v1: 另一张原图的k,v, inversion 时候存下来了   对应mask_s
        """
        h = heads
        # 1.两路的  重建 + 编辑
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=heads), (q, k, v))
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)    # [32,]
        
        if self.useuncond:
            q_t,out_t = map(lambda t : torch.cat([t[h:h*2],t[h*3:]],dim=0), (q,out))   # 两路: 重建、编辑  这里取出编辑的uncond,cond
            k1,v1 = self.speckvs[-1-cur_step][cur_layer] 
        else:
            q_t,out_t = map(lambda t : t[h*3:], (q,out))
            k1,v1 = self.speckvs[-1-cur_step][cur_layer]
            if gl.invert_use_uncond:
                k1,v1 = k1[1:],v1[1:]
                
        # 2. 原图的前景
        k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=heads), (k1, v1))
        sim1 = torch.einsum("b i d, b j d -> b i j", q_t, k1) * scale   # [bs,HW,HW]
        H = W = int(np.sqrt(q_t.shape[1]))
        # mask 掉原图的 ===========看啊===========
        # sim: [h, HW,HW]   mask: [1,1,HW]   求相似度矩阵 用原图的mask  
        # 整张结果图只对原图的前景进行注意力计算    遮住原图的背景  只要原图的前景:   # 使用mask_s
        imgname = f"mask_s.png"  if cur_step == 49 and H == 64 else None
        if gl.spec_mask_type in [0,1]:   # 0,1: 使用mask_s
            mask_s = self.deal_mask(self.mask_s,H,W,imgname)
        else:
            mask_s = torch.ones(H,W).to(sim1.device)
        # mask = self.aggregate_cross_attn_map(idx=self.source_token_idx)  # (4, H, W)
        # mask_s0 = mask[-2]  # (H, W)  mask_target condition的 因为uncond没用词
        # if gl.spec_mask_type in [0,1]:   # 0,1: 使用mask_s
        #     mask_s0 = self.deal_mask(mask_s0,H,W,imgname)
        # else:
        #     mask_s0 = torch.ones(H,W).to(sim1.device)
        # mask_s = mask_s0.reshape(H,W) * mask_s.reshape(H,W)  # 看啊
        mask_s = mask_s.reshape(1,1,H*W)
        sim1 = sim1 + mask_s.masked_fill(mask_s == 0, torch.finfo(sim.dtype).min) #前景  阻止背景像素 与 前景算相似度 mask = 1 代表前景  0 背景
        attn1= sim1.softmax(dim=-1)
        out1 = torch.einsum("b i j, b j d -> b i d", attn1, v1)
        
        # 3. 结合两个output   output:[bs,HW,dim]   前景用原图的    背景用结果图的
        mask = self.aggregate_cross_attn_map(idx=self.target_token_idx)  # (4, H, W)
        mask_t = mask[-1]  # (H, W)  mask_target condition的 因为uncond没用词
        imgname = f"mask_t_{cur_step}.png"  if cur_step % 10 == 0 and H == 64 else None
        if gl.spec_mask_type in [0,2]:  # 使用mask_t
            mask_t = self.deal_mask(mask_t,H,W,imgname)
        else:
            mask_t = torch.ones(H,W).to(mask.device)    # 应该全0还是全1 ?? 
        mask_t = mask_t.reshape(-1,H *W, 1)
    
        out_t = out1 * mask_t + out_t * (1 - mask_t)    # 前景用对象原图   背景用结果图

        if self.useuncond:
            out[h:h*2] = out_t[0:h] # 编辑图 uncond
            out[h*3:] = out_t[h:]   # 编辑图 cond
        else:
            out[h*3:] = out_t
            
        out = rearrange(out, '(b h) n d -> b n (h d)', h=heads)
        # 最后还需要to_out呀
        return out,attn     # 这个attn给p2p的controller用吧  其实
    
