import gradio as gr
from .callbacks import *

DEFAULT_SLIDING_WINDOW_LENGTH = 60 
DEFAULT_SLIDING_WINDOW_SEP = 10

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown(f"### Threshold Optimization")

    file_picker = gr.File(label="Select Session Data File", file_types=[".npz"])

    with gr.Column(visible=False) as inputs_outputs:
        with gr.Row():
            sliding_window_length = gr.Slider(label="Sliding Window Length (sec)", minimum=20, maximum=120, step=10, value=DEFAULT_SLIDING_WINDOW_LENGTH)
            sliding_window_sep = gr.Slider(label="Window Stride (sec)", minimum=5, maximum=20, step=5, value=DEFAULT_SLIDING_WINDOW_SEP)
            run_analysis_btn = gr.Button('Run Analysis')
        threshold_evolution_plot = gr.Plot(label="Optimized Threshold")
        with gr.Row():
            gen_output = gr.Button('Save Results')
            download_output = gr.File(label="Save Session with Optimized Thresholds")

    file_picker.change(
        load_file,
        inputs=[file_picker],
        outputs=[inputs_outputs]
    )
    run_analysis_btn.click(run_analysis,inputs=[sliding_window_length,sliding_window_sep],outputs=threshold_evolution_plot)
    gen_output.click(save_thresholds2file,inputs=[],outputs=[download_output])

if __name__ == "__main__":
    app.launch()