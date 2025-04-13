import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Example device logic
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("medical_chatbot_model").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("medical_chatbot_tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

model, tokenizer = load_model()

def prepare_input(tokenizer, question):
    prompt = f"Question: {question} Answer:"
    encoded_input = tokenizer.encode(prompt, return_tensors='pt').to(device)
    return encoded_input

def generate_text(model, tokenizer, encoded_input):
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            encoded_input,
            max_length=512,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

st.title("Medical Chatbot (Fine tuned with the given mle_screening_dataset)")

question = st.text_input("Ask a medical question:")
if st.button("Send"):
    if question.strip():
        encoded_input = prepare_input(tokenizer, question)
        generated_text = generate_text(model, tokenizer, encoded_input)
        
        if "Answer:" in generated_text:
            answer_only = generated_text.split("Answer:")[1].strip()
            st.write("**Answer:**", answer_only)
        else:
            st.write("Sorry, I couldn't parse the answer from the model.")
    else:
        st.warning("Please enter a valid question.")