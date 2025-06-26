## Description
In this project I created a LLM (Large Language Model) from scratch, without the aid of libraries such as PyTorch or TensorFlow. Taking on such a challenges is important to my personal development as it allows me to understand LLM's in a practical, encompassing manner, in all of it's complexity which modern libraries often take away. While getting the basic structure of the LLM was the most important task, this project allowed me to implement advanced methods such as LoRA (Low-rank Adaptation), and learning decay (cosine, linear). 

## Model structure
While my model doesn't rely on libraries to take on the complexity of modern LLM logic, I was still able to implement a state of the art structure in my code, particularily the self-attention head. Outlined in Google's famous paper "Attention is all you need" (2017), self-attention is arguable the most important breakthrough in modern machine learning. I was able to succesfully implement a multi-headed self attention structure, where each transformer had 8 self attention heads. Along with this, each transformer had a simple hidden layer with non-linear activation (ReLU). Finally, I implemented used 2 LayerNorms and a residual connection step in each transformer logic to supplement my attention based training. Using 6 of these transformers in my model's training, along with the embedding and positional matrices, makes up my from-scratch LLM.

## Hugging Face Repository
https://huggingface.co/spaces/kalandjl/leanai-gradio/tree/main
