import gradio as gr
from .callbacks import *

DEFAULT_SLIDING_WINDOW_LENGTH = 60 
DEFAULT_SLIDING_WINDOW_SEP = 30

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown(f"### Threshold Optimization")

    file_picker = gr.File(label="Select Session Data File", file_types=[".npz"])

    with gr.Column(visible=False) as inputs_outputs:
        with gr.Column():
            with gr.Row():
                sliding_window_length = gr.Slider(label="Sliding Window Length (sec)", minimum=20, maximum=120, step=10, value=DEFAULT_SLIDING_WINDOW_LENGTH)
                sliding_window_sep = gr.Slider(label="Window Stride (sec)", minimum=10, maximum=120, step=10, value=DEFAULT_SLIDING_WINDOW_SEP)
            with gr.Row():
                decoding_beta = gr.Number(minimum = 0,value=1,label='Penalizing Missing Stimulus Over Hallucinating It')
                behavior_beta = gr.Number(minimum = 0,value=1,label='Penalizing Not Predicting Licks Over Hallucinating Its Presence')
            run_analysis_btn = gr.Button('Run Analysis')
        threshold_evolution_plot = gr.Plot(label="Optimized Threshold")
        decoding_behavior_vs_plot = gr.Plot(label='Decoder over Behavior Performance')
        with gr.Row():
            with gr.Column():
                gr.Markdown('## Decoder Optimized')
                decoder_true_low_perception_rate = gr.Textbox(label="Rate of Correctly Not Anticipating Lick", interactive=False,value='')
                decoder_false_low_perception_rate = gr.Textbox(label="Rate of Wrongly Not Anticipating Lick", interactive=False,value='')
                decoder_true_high_perception_rate = gr.Textbox(label="Rate of Correctly Anticipating Lick", interactive=False,value='')
                decoder_false_high_perception_rate = gr.Textbox(label="Rate of Wrongly Anticipating Lick", interactive=False,value='')
            with gr.Column():
                gr.Markdown('## Bahavior Optimized')
                behavior_true_low_perception_rate = gr.Textbox(label="Rate of Correctly Not Anticipating Lick", interactive=False,value='')
                behavior_false_low_perception_rate = gr.Textbox(label="Rate of Wrongly Not Anticipating Lick", interactive=False,value='')
                behavior_true_high_perception_rate = gr.Textbox(label="Rate of Correctly Anticipating Lick", interactive=False,value='')
                behavior_false_high_perception_rate = gr.Textbox(label="Rate of Wrongly Anticipating Lick", interactive=False,value='')
        with gr.Column():
            gen_output = gr.Button('Save Results',interactive=False)
            download_output = gr.File(label="Save Session with Optimized Thresholds")

    file_picker.change(
        load_file,
        inputs=[file_picker],
        outputs=[inputs_outputs]
    )
    decoder_metrics = [decoder_true_low_perception_rate,decoder_false_low_perception_rate,decoder_true_high_perception_rate,decoder_false_high_perception_rate]
    behavior_metrics = [behavior_true_low_perception_rate,behavior_false_low_perception_rate,behavior_true_high_perception_rate,behavior_false_high_perception_rate]
    run_analysis_btn.click(run_analysis,inputs=[sliding_window_length,sliding_window_sep,decoding_beta,behavior_beta],outputs=[threshold_evolution_plot,decoding_behavior_vs_plot,gen_output,*decoder_metrics,*behavior_metrics])
    gen_output.click(save_results2file,inputs=[],outputs=[download_output])

if __name__ == "__main__":
    app.launch()