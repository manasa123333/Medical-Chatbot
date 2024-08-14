from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import gradio as gr
from huggingface_hub import hf_hub_download

DB_FAISS_PATH = "vectorstores/db_faiss"

def load_llm():
    model_name = 'TheBloke/Llama-2-7B-Chat-GGML'  # Correct model repository
    model_path = hf_hub_download(repo_id=model_name, filename='llama-2-7b-chat.ggmlv3.q8_0.bin', cache_dir='./models')
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_QA_chain(llm, prompt, db):
    qachain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qachain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-miniLM-L6-V2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_QA_chain(llm, qa_prompt, db)
    return qa

bot = qa_bot()

def chatbot_response(message, history):
    try:
        response = bot({'query': message})
        answer = response["result"]
        sources = response["source_documents"]
        if sources:
            answer += f"\nSources:" + str(sources)
        else:
            answer += "\nNo sources found"
        history.append((message, answer))
    except Exception as e:
        history.append((message, f"An error occurred: {str(e)}"))
    return history, history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox(show_label=False, placeholder="Enter your question...")
        submit = gr.Button("Send")
    submit.click(chatbot_response, [msg, chatbot], [chatbot, chatbot])
    msg.submit(chatbot_response, [msg, chatbot], [chatbot, chatbot])

demo.launch()


