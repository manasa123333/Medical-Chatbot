
# PDF-Based QA Chatbot

This repository contains a Python-based PDF question-answering (QA) chatbot application. The chatbot utilizes LangChain, FAISS, and the Llama-2 language model to provide intelligent responses based on the content of a PDF document.






## Features

1. PDF Document Processing: Extracts and processes text from PDF files.
2. Text Splitting: Segments text into manageable chunks for efficient processing.
3. Vector Embeddings: Uses Hugging Face's embedding models to convert text chunks into vectors.
4. FAISS Vector Store: Stores and retrieves vector embeddings efficiently.
5. Language Model Integration: Uses the Llama-2 model to generate contextual answers.
6. Custom Prompting: Implements a custom prompt template for structured responses.
7. Interactive Chatbot: A Gradio-based web interface for user interaction.


## Installation

Requirements

Python 3.8 or later

langchain

langchain_huggingface

langchain_community

gradio

huggingface_hub

faiss-cpu (for FAISS vector store)

transformers (for Llama-2 model)


    
## Run Locally

Clone the project

```bash
git clone https://github.com/manasa123333/Medical-Chatbot.git
cd Medical-Chatbot

```



Install dependencies

```bash
  python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt

```

Run the Vector Database Creation Script:

```bash
  python create_vector_db.py

```

Run the Chatbot:

```bash
  python chatbot.py


```

The Gradio interface will launch, and you can interact with the chatbot via your web browser.

## Usage

### Input: 
Type your questions related to the content of the PDF in the chat interface.
### Output: 
The chatbot provides answers based on the PDF content. Sources are included if available.

## Customization

### Change PDF File: 
Update the DATA_PATH variable in create_vector_db.py to point to a different PDF file.
### Model and Embeddings: 
Adjust the model_name and embedding model settings as needed.
### Prompt Template: 
Modify the custom_prompt_template in chatbot.py to change how the chatbot formulates responses.
## Troubleshooting

### Memory Issues: 
Ensure you have sufficient memory if processing large PDFs. Consider chunking the PDF into smaller sections if necessary.
### Model Download Issues: 
Verify the model name and path. Make sure you have the necessary permissions to download and use the model.
## Acknowledgements

 - [Langchain](https://www.langchain.com/)
 - [FAISS](https://github.com/facebookresearch/faiss)
 - [Gradio](https://www.gradio.app/)
  - [Hugging Faces Transformers](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin)


## Demo

[Medical Chatbot](https://huggingface.co/spaces/Manasa1/medicalbot)


## Screenshots

![Screenshot (13)](https://github.com/user-attachments/assets/1bf2a679-d281-4550-a140-9a0a59306a5f)

![Screenshot (14)](https://github.com/user-attachments/assets/bcd5b60f-1a95-48fb-81c3-0b5ffc898d4c)

