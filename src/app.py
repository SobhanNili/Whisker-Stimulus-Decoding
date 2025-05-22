import gradio as gr
from .tabs.temp_matching.interface import app as temp_matching_app
from .tabs.threshold_optim.interface import app as thresh_optim_app
from .tabs.continuous_vis.interface import app as cont_vis_app

with gr.Blocks() as multi_tab_app:
    with gr.Tab("Template Construction"):
        temp_matching_app.render()  # Render the imported interface
    with gr.Tab("Threshold Optimization"):
        thresh_optim_app.render()  # Render the example tab
    with gr.Tab("Session Perception Visulization"):
        cont_vis_app.render()  # Render the example tab

multi_tab_app.launch()