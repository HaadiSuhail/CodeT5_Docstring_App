# CodeT5 Python Docstring Generator App

This is a simple Streamlit web app that uses the [Salesforce CodeT5-small](https://huggingface.co/Salesforce/codet5-small) transformer model to generate function-level docstrings for Python code.

**Features:**
- Paste Python function code in a text area
- Generate a docstring automatically
- Built with Streamlit and Hugging Face Transformers

**Note:**
This app uses the pretrained CodeT5-small model. For higher-quality docstrings, you should fine-tune the model on the CodeSearchNet dataset.

## How to Run

1. Clone the repository:
git clone https://github.com/HaadiSuhail/CodeT5_Docstring_App.git
cd CodeT5_Docstring_App


2. Install dependencies:
pip install -r requirements.txt


3. Run the app:
streamlit run app.py


## Example

Input:
```python
def add(a, b):
 return a + b

Output (pretrained model):
add

(Fine-tuning recommended for better results)

License
MIT License