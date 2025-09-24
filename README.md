# 🧠💬 QnA Model  

A Question Answering (QnA) system built with **PyTorch Lightning**, **Hugging Face Transformers**, and **Gradio**.  
The project fine-tunes the pretrained **RoBERTa (SQuAD2)** model on custom QnA data, with MLflow tracking, checkpointing, and early stopping.  

🌐 **Live Demo:** [Try it on Hugging Face Spaces 🚀](https://huggingface.co/spaces/Samin7479/QnAModel)

---

## ✨ Features
- 🔥 Pretrained transformer backbone: `deepset/roberta-base-squad2`  
- ⚡ Fine-tuning with PyTorch Lightning  
- 📂 Custom dataset loading with `QnADataset`  
- 📊 Logging & experiment tracking using **MLflow**  
- 💾 Checkpointing & Early Stopping  
- 🎨 Interactive Gradio UI + Deployed on Hugging Face Spaces  

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/Yasar8936/QnA-Model.git
cd QnA-Model
pip install -r requirements.txt
