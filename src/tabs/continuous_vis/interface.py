RESPONSE_WINDOW = 1
FIXED_THRESHOLD = 0.8

import gradio as gr
from .callbacks import *

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown(f"### Session Perception Plot")

    file_picker = gr.File(label="Select Session Data File", file_types=[".npz"])

    with gr.Column(visible=False) as inputs_outputs:
        with gr.Row():
            start_time_slider = gr.Slider(0, 0, value=0, step=1, label="Start Time (seconds)")
            interval_slider = gr.Slider(0, 100, value=10, step=1, label="Interval (seconds)")

        with gr.Row():
            show_fr_checkbox = gr.Checkbox(label="Show False Reaction", value=True)
            seek2stim_button = gr.Button("Seek to Next Stim Time")
        
        plot_button = gr.Button("Plot")
        plot_output = gr.Plot(label="Perception Trajectory Plot")

    # Update button appearance when inputs change
    start_time_slider.change(lambda: gr.update(value="Replot", variant="primary"), inputs=[], outputs=plot_button)
    interval_slider.change(lambda: gr.update(value="Replot", variant="primary"), inputs=[], outputs=plot_button)
    show_fr_checkbox.change(lambda: gr.update(value="Replot", variant="primary"), inputs=[], outputs=plot_button)

    file_picker.change(
        load_file,
        inputs=[file_picker],
        outputs=[inputs_outputs,start_time_slider]
    )

    seek2stim_button.click(get_next_stim_time, inputs=[start_time_slider,interval_slider], outputs=start_time_slider)

    plot_button.click(
        plot_data,
        inputs=[start_time_slider, interval_slider, show_fr_checkbox],
        outputs=[plot_output, plot_button]  # Reset button appearance after plotting
    )