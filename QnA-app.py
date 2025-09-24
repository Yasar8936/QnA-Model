import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# copied: device + model name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "deepset/roberta-base-squad2"  # replace with your fine-tuned path if you have one

# copied: load tokenizer/model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
model.eval()

def answer_question(context, question, max_length=384, doc_stride=128):
    if not context or not question:
        return ""
    enc = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_offsets_mapping=True,
        padding="max_length",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0]  # keep on CPU

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        start = int(torch.argmax(out.start_logits[0]))
        end   = int(torch.argmax(out.end_logits[0]))
    if end < start:
        end = start

    start_char, end_char = 0, 0
    if 0 <= start < len(offsets) and 0 <= end < len(offsets):
        start_char = offsets[start][0]
        end_char   = offsets[end][1]
    return context[start_char:end_char].strip()

# simple UI
with gr.Blocks(title="QnA (Roberta SQuAD2)") as demo:
    gr.Markdown("### Question Answering\nEnter a context and a question.")
    ctx = gr.Textbox(label="Context", lines=8, placeholder="Paste passage here...")
    q   = gr.Textbox(label="Question", lines=2, placeholder="Ask something about the passage...")
    max_len = gr.Slider(128, 512, value=384, step=16, label="max_length")
    stride  = gr.Slider(32, 256, value=128, step=16, label="doc_stride")
    ans = gr.Textbox(label="Answer", lines=2)
    btn = gr.Button("Answer")
    btn.click(answer_question, inputs=[ctx, q, max_len, stride], outputs=ans)

# For Hugging Face Spaces (Gradio SDK): just call launch().
demo.launch()
