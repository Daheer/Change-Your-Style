from base64 import b64encode

import numpy
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel

from IPython.display import HTML
from matplotlib import pyplot as plt
from PIL import Image
from torch import autocast
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging

import gdown
import os

torch.manual_seed(1)
logging.set_verbosity_error()

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists('models/vae.pt'): vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
if not os.path.exists('models/unet.pt'): unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")
if not os.path.exists('models/scheduler.pt'): scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
if not os.path.exists('models/tokenizer.pt'): tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
if not os.path.exists('models/text_encoder.pt'): text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

def download_models():
    if not os.path.exists('models/vae.pt'): gdown.download(url = '', output = 'vae.pt')
    if not os.path.exists('models/unet.pt'): gdown.download(url = '', output = 'unet.pt')
    if not os.path.exists('models/scheduler.pt'): gdown.download(url = '', output = 'scheduler.pt')
    if not os.path.exists('models/tokenizer.pt'): gdown.download(url = '', output = 'tokenizer.pt')
    if not os.path.exists('models/text_encoder.pt'): gdown.download(url = '', output = 'text_encoder.pt')   

def pil_to_latent(input_im):
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) 
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def get_style(style):
    learned_emebeds_map = {
        'Ghibli': ['<ghibli-face>', 'ghibli'],
        'Manga': ['<manga>', 'manga'],
        'GTA 5': ['<gta5-artwork>', 'gta'],
        'Sims': ['<sims2-portrait>', 'sims'],
        'Kaya Ghost Assasin': ['<kaya-ghost-assasin>', 'kaya'],
        'Uzumaki': ['<NARUTO>', 'uzumaki'],
        'Arcane': ['<arcane-style-jv>', 'arcane']
    }
    return learned_emebeds_map[style]

def change_style(image, style, inf_steps, guidance, str_step):

    input_image = Image.fromarray(image).resize((512, 512))
    encoded = pil_to_latent(input_image)
    learned_emebed = torch.load('learned_embeds/{}_learned_embeds.bin'.format(get_style(style)[1]))
    prompt = 'portrait of a person in the style of temp'

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    input_ids = text_input.input_ids.to(torch_device)
    position_ids = text_encoder.text_model.embeddings.position_ids[:, :77]

    token_emb_layer = text_encoder.text_model.embeddings.token_embedding
    pos_emb_layer = text_encoder.text_model.embeddings.position_embedding

    position_embeddings = pos_emb_layer(position_ids)
    token_embeddings = token_emb_layer(input_ids)

    replacement_token_embedding = learned_emebed[get_style(style)[0]].to(torch_device)

    token_embeddings[0, torch.where(input_ids[0]==11097)] = replacement_token_embedding.to(torch_device)

    input_embeddings = token_embeddings + position_embeddings

    bsz, seq_len = input_embeddings.shape[:2]
    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len, dtype=input_embeddings.dtype)

    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=input_embeddings,
        attention_mask=None, 
        causal_attention_mask=causal_attention_mask.to(torch_device),
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
    )
    modified_output_embeddings = encoder_outputs[0]

    modified_output_embeddings = text_encoder.text_model.final_layer_norm(modified_output_embeddings)

    height = 512                        
    width = 512                         
    num_inference_steps = inf_steps            
    guidance_scale = guidance             
    generator = torch.manual_seed(32)   
    batch_size = 1

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 
    text_embeddings = torch.cat([uncond_embeddings, modified_output_embeddings])

    scheduler.set_timesteps(num_inference_steps)
    start_step = str_step
    start_sigma = scheduler.sigmas[start_step]
    noise = torch.randn_like(encoded)

    latents = scheduler.add_noise(encoded, noise, timesteps=torch.tensor([scheduler.timesteps[start_step]]))
    latents = latents.to(torch_device).float()

    for i, t in tqdm(enumerate(scheduler.timesteps)):
        if i >= start_step:
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            torch.cuda.empty_cache()

            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

    return(latents_to_pil(latents)[0])

    
