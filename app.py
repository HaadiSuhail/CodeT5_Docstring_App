import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer once (cache them)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "./model"
    )
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("CodeT5 Python Docstring Generator")
st.write("Paste your Python function below:")

code_input = st.text_area("Python Function", height=300)

if st.button("Generate Docstring"):
    if not code_input.strip():
        st.warning("Please enter some code.")
    else:
        # Add prefix to tell model what to do
        prefixed_input = "generate docstring: " + code_input

        # Encode the input
        inputs = tokenizer.encode(
            prefixed_input,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Generate output IDs
        output_ids = model.generate(
            inputs,
            max_length=64,
            num_beams=5,
            no_repeat_ngram_size=3
        )
        # Decode generated text
        generated_docstring = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        # Display result
        st.subheader("Generated Docstring:")
        st.code(generated_docstring, language="markdown")