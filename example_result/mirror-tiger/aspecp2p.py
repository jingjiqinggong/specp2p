import torch
from diffusers import DDIMScheduler
import gl
from specp2p.pipeline import Pipeline
from specp2p.myatt import register_attention, SelfAttControl
from specp2p.maskhalf import AttHalfMutualMask
from torchvision.io import read_image
from torchvision.utils import save_image
import torch.nn.functional as F

num_inference_steps = 50
invert_guidance_scale = 1
guidance_scale = 7.5
mdict = {
    "": "/home/jchuang/huggingface_cache/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819/",
    "anything-v3.0": "/home/jchuang/huggingface_cache/models--Linaqruf--anything-v3.0/snapshots/8323d54dcf89c90c39995b04ae43166520e8992a/",
    "anything-v4.0": "/home/jchuang/huggingface_cache/andite--anything-v4.0/models--andite--anything-v4.0/snapshots/1f509626c469baf6fd3ab37939158b4944f90005/",
    "stable-diffusion-v1-4":"/home/jchuang/huggingface_cache/models--CompVis--stable-diffusion-v1-4/snapshots/249dd2d739844dea6a0bc7fc27b3c1d014720b28/",
    "stable-diffusion-v1-5": "/home/jchuang/huggingface_cache/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819/",
}
model_path = mdict["stable-diffusion-v1-4"]
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,set_alpha_to_one=True,steps_offset=1)   #my
model = Pipeline.from_pretrained(model_path,scheduler=scheduler,low_cpu_mem_usage=False).to(device) #torch_dtype=torch.float64

def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))   # 默认是nearest   Image.resize 默认是bicubic
    image = image.to(device)
    return image

def set_gl():
    gl.NUM_DDIM_STEPS = num_inference_steps
    gl.GUIDANCE_SCALE = guidance_scale
    gl.device = device
    gl.tokenizer = model.tokenizer
    gl.ldm_stable = model
    gl.isuncond = False
    gl.opt = 2  #0: 没有   1:null-text  2:ref   3:uncond noise   (4: 一步加噪)
    gl.invert_use_uncond = False
    
set_gl()
from p2p.p2p import *
image_path1 =  "./gradio_app/gnochi_mirror_1.jpeg"
image_path2 = "./gradio_app/tiger.jpg"  # 这里最好用原图吧
image_cut = "./gradio_app/tiger_mask.jpg"    # imga2 用AI抠图 去掉背景的  为了获取mask  https://photokit.com/editor/?lang=zh https://tool.lu/cutout/
# image_mask = "./gradio_app/tiger_mask.jpg"
source_image1= load_image(image_path1, device)    # [1,3,512,512]
source_image2= load_image(image_path2, device)    # [1,3,512,512]   取没有背景的 （背景黑色的）
# mask_image = load_image(image_mask,device)        # 也可以的啦
mask_image = (load_image(image_cut,device) >-1).max(dim=1)[0].float().expand(1,3,-1,-1)
## c = b[0].permute(1,2,0).cpu().numpy()
## c = (c * 255).astype('uint8')
## Image.fromarray(c).save('./gradio_app/tiger_mask.jpg')
# save_image(b,"./gradio_app/tiger_mask.jpg")

source_image = torch.cat([source_image1,source_image2],dim=0)
save_image(source_image*0.5+0.5,"ap2p1.jpg")
prompts = ["a cat sitting next to a mirror",
           "",		# 后面的有“有话” “无话” 指这里
           "a tiger sitting next to a mirror",]
target_words = ["tiger"]       # 记得设置成和和local blend的相同哦

target_token_idx = []
for word in target_words:
    id=ptp_utils.get_word_inds(prompts[2], word,gl.tokenizer,print_tokens=True).tolist()
    target_token_idx.extend(id)
print("target token idx: ", target_token_idx)


editor = SelfAttControl(num_inference_steps,16)
register_attention(model.unet, editor, isself=True)
editor.set_mode("invert")
start_code, x0t_list,noise = model.invert(source_image,
                                         prompts[0:2],
                                        guidance_scale=invert_guidance_scale,
                                        num_inference_steps=num_inference_steps,
                                        return_intermediates=2)
x0t_list = [x[0:1] for x in x0t_list]
# null_inversion = NullInversion(ldm_stable)
# (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path1, prompts[0], offsets=(0,0,0,0), verbose=True,
#                                             controller = None, use_opt=gl.opt)
# x0t_list = [x[0:1] for x in x_t]
x_t = x0t_list[-1] 
gl.x0t_list = x0t_list 


editor.to_gl()  # 存到  speckvs  用 k,v = gl.speckvs[-1-self.cur_step][self.cur_att_layer]  # 注释掉 就是原版prompt-to-prompt
gl.speckv_step = (4,50)  # 优先级比 self_replace_steps 高
gl.speckv_layer = (10,16)   # <= <
gl.spec_useuncond = (False & gl.invert_use_uncond) # # ptp_utils.py register里   替换self kv是否替换uncond self attention kv
gl.spec_mask_type = 0   # 0: 两个mask都用   1: 只用mask_s   2: 只用mask_t   其他值: 都不用mask

# thres 老虎 replace 0.5  refine 0.3
if hasattr(gl,'speckvs'):
    editor_half = AttHalfMutualMask(thres=0.5, mask_s=mask_image[0,0,:,:], 
                            target_token_idx=target_token_idx, 
                            mask_save_dir="./maskimg",
                            speckvs=gl.speckvs,useuncond=gl.spec_useuncond,
                            polatemode="bilinear")  # 对cross attmap的上采样插值方式   nearest,bilinear,bicubic
    gl.editor_half_mutual = editor_half     # 

prompts = [prompts[0], prompts[2]]
cross_replace_steps = {'default_': 0.8}
self_replace_steps = 0.5 # 哈士奇的图 用0.6吧
blend_word = None
blend_word = (('cat',), ("tiger",)) # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
# substruct_words = (("human","toy",),("human","toy",))
substruct_words = None
eq_params = None
# eq_params = {"words": ("bird",), "values": (2,)} 
controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps, blend_word, eq_params,
                             substruct_words,
                             th=(0.3,0.3)) # False:Refinement   th0: words的  th1: substruct word 的
# controller = AttentionStore()
# 这里又register了controller
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=None,
                            outimgname="ap2p2.jpg")

# Image.fromarray(np.concatenate(images,axis=1)).save("shan.jpg")

for i in range(len(prompts)):
    show_cross_attention(prompts,controller, 16, ["up", "down"],i)
    # show_cross_attention([prompt,target_prompt],controller, 8, ["mid"],i)
    # show_cross_attention([prompt,target_prompt],controller, 32, ["up", "down"],i)  
print('hhh')