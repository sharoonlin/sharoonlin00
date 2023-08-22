import gradio as gr
import numpy as np
from prediction import *

with gr.Blocks() as demo:
    gr.Markdown("利用LSTM預測台灣加權指數未來5個交易日的開盤價")
    with gr.Tab("預測"):
        with gr.Row():
            with gr.Column():
                hidden_dim_dd = gr.Dropdown(["256", "512"], label="Hidden Size", info="LSTM中每一層的神經元數目")
                num_layers_dd = gr.Dropdown(["1", "2"], label="Layer", info="模型中LSTM使用的層數")
                epoch_dd = gr.Dropdown(["100", "1000"], label="Epoch", info="模型訓練次數")
                input_list = [hidden_dim_dd, epoch_dd, num_layers_dd]
                predict_button = gr.Button("開始預測")
            with gr.Column():
                model_test = gr.Plot(label="模型測試結果")
                predict_output = gr.Plot(label="模型預測結果")
                output_list = [model_test, predict_output]
    # with gr.Tab("Flip Image"):
    #     with gr.Row():
    #         image_input = gr.Image()
    #         image_output = gr.Image()
    #     image_button = gr.Button("Flip")

    predict_button.click(predict, inputs=input_list, outputs=output_list)
    # image_button.click(flip_img, inputs=image_input, outputs=image_output)

demo.launch(share=True)  # share=True
