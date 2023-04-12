```python
import jieba
jieba.initialize()
import gradio as gr


def image_classifier(text):
    res = jieba.lcut(text, cut_all=False)
    return res


demo = gr.Interface(fn=image_classifier,
                    inputs=[
                        gr.components.Textbox(
                            lines=10, label="Input", placeholder="请输入文本"
                        ),
                    ],
                    outputs=[
                        gr.components.JSON(
                            label="Output"
                        ),
                    ],
                    title="Chinese-Vicuna 中文小羊驼",
                    )
demo.launch()
```

![image](https://user-images.githubusercontent.com/27845149/231429340-0180bd85-822d-4b72-956d-a40e6e78b85a.png)
