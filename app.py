import os
#env_value = os.getenv({``env_value``})

import gradio as gr

# 输入name字符串，输出Hello {name}!字符串
def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(lines=2, placeholder="Name Here..."),
    outputs="text",
    allow_flagging="never",
)
import torch
import requests
from torchvision import transforms

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


# def predict(inp):
#     inp = transforms.ToTensor()(inp).unsqueeze(0)
#     with torch.no_grad():
#         prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
#         confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
#     return confidences
#
#
# demo1 = gr.Interface(fn=predict,
#                     inputs=gr.inputs.Image(type="pil"),
#                     outputs=gr.outputs.Label(num_top_classes=3),
#                     examples=[["cheetah.jpg"]],
#                     )

if __name__ == "__main__":
    demo.launch()
