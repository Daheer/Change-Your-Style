from fastapi import FastAPI
import gradio as gr
from utils import change_style

CUSTOM_PATH = "/change-your-style"

app = FastAPI()


@app.get("/")
def read_main():
    return {"Status": "Working"}

def generate(Image, Style, Inference_Steps, Guidance, Start_Step):
    return change_style(Image, Style, Inference_Steps, Guidance, Start_Step)
    
style = gr.Radio(['GTA 5', 'Manga', 'Ghibli', 'Sims', 'Kaya Ghost Assasin', 'Arcane', 'Uzumaki'])
inf_steps = gr.Slider(minimum = 10, maximum = 100, value = 50, step = 1)
guidance = gr.Slider(minimum = 5, maximum = 50, value = 10, step = 1)
str_step = gr.Slider(minimum = 10, maximum = 100, value = 25, step = 1)

io = gr.Interface(generate, ["image", style, inf_steps, guidance, str_step], gr.Image())
app = gr.mount_gradio_app(app, io, path=CUSTOM_PATH)
io.launch()

