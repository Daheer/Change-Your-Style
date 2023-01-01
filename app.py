import gradio as gr
from utils import change_style

def generate(Image, Style, Inference_Steps, Guidance, Start_Step):
    return change_style(Image, Style, Inference_Steps, Guidance, Start_Step)
    
style = gr.Radio(['GTA 5', 'Manga', 'Ghibli', 'Sims', 'Kaya Ghost Assasin', 'Arcane', 'Uzumaki'])
inf_steps = gr.Slider(minimum = 10, maximum = 100, value = 50, step = 1)
guidance = gr.Slider(minimum = 5, maximum = 50, value = 10, step = 1)
str_step = gr.Slider(minimum = 10, maximum = 100, value = 25, step = 1)

io = gr.Interface(generate, ["image", style, inf_steps, guidance, str_step], gr.Image())

io.launc()

