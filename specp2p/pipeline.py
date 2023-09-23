"""
Util functions based on Diffuser framework.
"""
import os
import torch
import cv2
import numpy as np

import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torchvision.io import read_image

from diffusers import StableDiffusionPipeline
from torch.optim.adam import Adam
from pytorch_lightning import seed_everything
from typing import List,Tuple
import gl

class Pipeline(StableDiffusionPipeline):
    # 从 x_t 变成 x_t+1
    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        timestep, next_step = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod

        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]    
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next, pred_x0
    # x_t 变成 x_t-1
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        # 不管把DDIM 的steps_offset设成多少，timesteps 都是[...., 1]
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        # 源码：scheduler.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod 
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev, pred_x0

    
    @torch.no_grad()
    def image2latent(self, image):
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        # input image density range [-1, 1]
        latents = self.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='pt'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    @torch.no_grad()
    def invert(
        self,
        image: torch.Tensor,
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        eta=0.0,
        return_intermediates=False,
        **kwds):
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        IG = 1.0 if hasattr(gl,"invert_use_uncond") and not gl.invert_use_uncond else -1.0  # my add
        DEVICE = self.device
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        # text embedding
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        )
        text_embedding = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        print("input text embedding :", text_embedding.shape)
        # define initial latents
        latents = self.image2latent(image)
        start_latents = latents
        # unconditional embedding for classifier free guidance
        if guidance_scale > IG:
            max_length = text_input.input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=77,
                return_tensors="pt"
            )
            unconditional_embedding = self.text_encoder(unconditional_input.input_ids.to(DEVICE))[0]
            text_embedding = torch.cat([unconditional_embedding, text_embedding], dim=0)
            self.context = text_embedding

        print("latents shape: ", latents.shape)
        # interative sampling
        self.scheduler.set_timesteps(num_inference_steps)   #参数：timestep的步数 = num_inference : 生成的时候用的步数   
        print("Valid timesteps: ", reversed(self.scheduler.timesteps))
        # print("attributes: ", self.scheduler.__dict__)
        latents_list = [latents] #x_t,z_t
        pred_x0_list = [latents] #pred z_0
        noise_pred_list = []
        # loop:  inversion 过程   (reversed 从)
        timesteps = reversed(self.scheduler.timesteps)
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Inversion")):
            if guidance_scale > IG:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            noise_pred = self.unet(model_inputs, t, encoder_hidden_states=text_embedding).sample
            if guidance_scale > IG:
                # image * 2  
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)
            latents, pred_x0 = self.next_step(noise_pred, t, latents) #  加噪
            # pred_x0; 在括号里的latente下，完全去噪的图 （可以一步去噪一步加噪，但是一步去噪效果很差）
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)
            noise_pred_list.append(noise_pred)
        if return_intermediates == 1:
            # return the intermediate laters during inversion
            # pred_x0_list = [self.latent2image(img, return_type="pt") for img in pred_x0_list]
            return latents, latents_list   # [1,4,64,64]  len() = 51  [0] == start_latents
        elif return_intermediates == 2:
            return latents, latents_list,noise_pred_list 
        return latents, start_latents   # [1,4,64,64]  [1,4,64,64]
           