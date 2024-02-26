#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from agents.SQLagent import build_sql_agent
from agents.csv_chat import build_csv_agent
from utils import utility as ut
import streamlit as st

# app.py
from typing import List, Union, Optional
# from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import io
from langchain_google_genai import GoogleGenerativeAIEmbeddings


st.session_state.csv_file_paths = []


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

PROMPT_TEMPLATE = """
Use the following pieces of context enclosed by triple backquotes to answer the question at the end.
\n\n
Context:
```
{context}
```
\n\n
Question: [][][][]{question}[][][][]
\n
Answer:"""



def open_ai_key():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key or Google Gemini", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
    os.environ["OPENAI_API_KEY"] = openai_api_key
        
@st.cache_data
def dbActive():
    os.environ['DB_ACTIVE'] = 'false'


def init_page() -> None:
    st.set_page_config(
    )
    st.sidebar.title("Options")
    icon, title = st.columns([3, 20])
    with icon:
        st.image('./img/image.png')
    with title:
        st.title('Finance Chatbot')
    st.session_state['db_active'] = False

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?", "img_path": None}]
        st.session_state.costs = []


def init_memory():
    return ConversationBufferMemory(
        llm=ChatOpenAI(temperature=0.1),
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)

def get_csv_file() -> Optional[str]:
    """
    Function to load PDF text and split it into chunks.
    """
    import tempfile
    
    st.header("Upload Document or Connect to a Database")
    
    uploaded_files = st.file_uploader(
        label="Here, upload your documents you want AskMAY to use to answer",
        type= ["csv", 'xlsx', 'pdf','docx'],
        accept_multiple_files= True
    )

    if uploaded_files:
        all_docs = []
        csv_paths = []
        all_files = []
        for file in uploaded_files:
            
            Loader = None
            if file.type == "text/plain":
                Loader = TextLoader
            elif file.type == "application/pdf":
                Loader = PyPDFLoader
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                Loader = Docx2txtLoader

            elif file.type == "text/csv":
                csv_paths.append(file)

            elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                loader = ut.ExcelLoader(file)
                paths = loader.load()
                
                csv_paths.extend(paths)

            else:
                file.type
                raise ValueError('File type is not supported')

            if Loader:
                file_buffer = io.BytesIO(file.getvalue())
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file_buffer.getvalue())
                    loader = Loader(temp_file.name)
                    docs = loader.load()
                    all_docs.extend(docs)

        if all_docs:
            documents = text_splitter.split_documents(all_docs)
            all_files.append(('docs', documents))
        if csv_paths:
            all_files.append(('csv', csv_paths))
        all_files = tuple(all_files)

        return all_files
    else:
        return None
    
def get_db_credentials(model_name, temperature, chain_mode='Database'):
    """
    creates a form for the user to input database login credentials
    """

    # Check if the form has already been submitted
    
    db_active = os.environ['DB_ACTIVE']
    if db_active == "true":
        print(db_active)

        return st.session_state['models']
        
    else:
        username = None
        host = None
        port = None
        db = None
        password = None
        import time
        pholder = st.empty()
        
        with pholder.form('Database_Login'):
            st.write("Enter Database Credentials ")
            username = st.text_input('Username').strip()
            password = st.text_input('Password', type='password',).strip()
            rdbs = st.selectbox('Select RDBS:',
                                ("Postgres",
                                'MS SQL Server/Azure SQL',
                                "MySQL",
                                "Oracle")
                            )
            port = st.number_input('Port')
            host = st.text_input('Hostname').strip()
            db = st.text_input('Database name').strip()

            submitted = st.form_submit_button('Submit')

        if submitted:
            with st.spinner("Logging into database..."):
                
                llm_chain, llm = init_agent(model_name=model_name,
                                    temperature=temperature,
                                    rdbs = rdbs,
                                    username=username,
                                    password=password,
                                    port=port,
                                    host=host,
                                    database=db,
                                    chain_mode = chain_mode)
            st.session_state['models'] = (llm_chain, llm)
            st.success("Login Success")
            os.environ['DB_ACTIVE'] = "true"
            db_active = os.environ['DB_ACTIVE']
            st.session_state['db_active'] = True
            time.sleep(2)
            pholder.empty()

            # If the form has already been submitted, return the stored models
        if db_active == "true":
            #return st.session_state['models']
            mds =  st.session_state['models']
            st.write("Reached")
            return mds
        else:
            st.stop()

@st.cache_resource
def build_vector_store(
    _docs: str, _embeddings: Union[OpenAIEmbeddings, LlamaCppEmbeddings]) \
        -> Optional[Chroma]:
    """
    Store the embedding vectors of text chunks into vector store (Qdrant).
    """
    if _docs:
        
        with st.spinner("Loading FIle ..."):
            
            chroma = Chroma.from_documents(
             _docs, _embeddings
            )
    
        st.success("File Loaded Successfully!!")
        return chroma
    else:
        chroma = None
        return chroma


# Select model 

def select_llm() -> Union[ChatOpenAI]:
    """
    Read user selection of parameters in Streamlit sidebar.
    """
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("gpt-3.5-turbo-1106",
                                   "gpt-3.5-turbo-16k-0613",
                                   "gpt-4",
                                  ))
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    chain_mode = st.sidebar.selectbox(
                        "What would you like to query?",
                        ("Documents", "CSV|Excel", 'Database')
    )
    
    return model_name, temperature, chain_mode


def init_agent(model_name: str, temperature: float, **kwargs) -> Union[ChatOpenAI]:
    """
    Load LLM.
    """
    llm_agent = None  # Initialize llm_agent with a default value
    
    if model_name.startswith("gpt-"):
        llm =  ChatOpenAI(temperature=temperature, model_name=model_name)
    
    chain_mode = kwargs['chain_mode']
    if chain_mode == 'Database':
        rdbs = kwargs['rdbs']
        username = kwargs['username']
        password = kwargs['password']
        host = kwargs['host']
        port = kwargs['port']
        database = kwargs['database']
        llm_agent = build_sql_agent(llm=llm, rdbs=rdbs, username=username, password=password,
                                    host=host, port=port, database=database)
    if chain_mode == 'CSV|Excel':
        file_paths = kwargs['csv']
        if file_paths is not None:
            with st.spinner("Loading CSV FIle ..."):
                llm_agent = build_csv_agent(llm, file_path=file_paths)
            st.session_state['CSVs'] = file_paths
    
    return llm_agent, llm

def get_retrieval_chain(model_name: str, temperature: float, **kwargs) -> Union[ChatOpenAI]:
    
    docsearch = kwargs['docsearch']
    if model_name.startswith("gpt-"):
        llm =  ChatOpenAI(temperature=temperature, model_name=model_name)

    prompt_template_doc = """
    Use the following pieces of context to answer the question at the end.
    {context}
    If you still cant find the answer, just say that you don't know, don't try to make up an answer.
    You can also look into chat history.
    {chat_history}
    Question: {question}
    Answer:
    """

    prompt_doc = PromptTemplate(
    template=prompt_template_doc, input_variables=["context", "question", "chat_history"])
    memory = init_memory()
    retrieval_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm, 
                            retriever = docsearch.as_retriever(search_kwargs={"k": 4}),
                            memory=memory,
                            combine_docs_chain_kwargs={"prompt": prompt_doc})

    return retrieval_chain, llm
        

def load_embeddings(model_name: str) -> Union[OpenAIEmbeddings, LlamaCppEmbeddings,HuggingFaceEmbeddings]:
    """
    Load embedding model.
    """
    #return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyAfSMYWXQP9jenHCH0_V0vHd3-d07h8ODk")

    if model_name.startswith("gpt-"):
        embed =  OpenAIEmbeddings(model="text-embedding-3-large")
        try:
            assert embed is not None
            return embed
        except AssertionError as aerr:
            err = str(aerr)
            st.error(err)
        

def get_answer(llm_chain,llm, message, chain_type=None) -> tuple[str, float]:
    """
    Get the AI answer to user questions.
    """
    import langchain

    if isinstance(llm, (ChatOpenAI, OpenAI)):
        with get_openai_callback() as cb:
            try:
                if isinstance(llm_chain, ConversationalRetrievalChain):
                    history = st.session_state.messages.copy()
                    history.pop()
                    for msg in st.session_state.messages:
                        if msg["role"] == "assistant":
                            llm_chain.memory.chat_memory.add_ai_message(msg["content"])
                            if msg.get("img_path"):
                                llm_chain.memory.chat_memory.add_ai_message(msg["img_path"])
                        else:
                            llm_chain.memory.chat_memory.add_user_message(msg["content"])
                        
                    #llm_chain.memory.chat_memory.add_ai_message()
                    response = llm_chain({"question": message})
                    answer =  str(response['answer'])
                else:
                    assert chain_type is not None
                    print(message)
                    isplot = ut.classify_prompt(message)
                    print(isplot,'--------------------------------------')
                    if isplot:
                        if chain_type == "Database":
                            from utils.prompts import plot_prompt
                            sql_plot_prompt = plot_prompt.format(message)
                            csv_string = llm_chain.run(sql_plot_prompt)
                            lida_data_path = ut.extract_data(csv_string)

                        elif chain_type == "CSV|Excel":
                            file_paths = st.session_state['CSVs']
                            lida_data_path = ut.create_lida_data(file_paths)

                        with st.spinner("Generating chart"):
                            rationale, img_path  = ut.generate_plot(lida_data_path, message)
                            st.session_state.messages.append({"role": "assistant", "content": rationale})
                            ut.display(img_path, rationale)
                        answer = rationale
                    else:
                        
                        answer = llm_chain.run(st.session_state.messages)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.write(answer)
            except Exception as e :#langchain.schema.StrOutputParser as e:
                response = str(e)
                if not response.startswith("Could not parse tool input: "):
                    raise e
                answer = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
        return answer, cb.total_cost
    
    if isinstance(llm):
        response = llm_chain.run(message)
        return response, 0.0


def main() -> None:
    import openai
    init_page()
    dbActive()
    try:
        open_ai_key()
        _ = load_dotenv(find_dotenv())
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        openai.api_key  = os.getenv("OPENAI_API_KEY")
        model_name, temperature, chain_mode = select_llm()
        embeddings = load_embeddings(model_name)
        files = get_csv_file()
        paths, texts, chroma = None, None, None

        if chain_mode == 'Database':
            llm_chain, llm = None, None
            try:
                print(os.environ['DB_ACTIVE'])
                if os.environ['DB_ACTIVE'] == "true":
                    llm_chain, llm = st.session_state['models']
                    
                else:
                    llm_chain, llm = get_db_credentials(model_name=model_name, temperature=temperature,
                                                    chain_mode=chain_mode)
            except KeyError:
                st.sidebar.warning('Provide a Database Log in Details')
                os.environ['DB_ACTIVE'] = "false"
                llm_chain, llm = get_db_credentials(model_name=model_name, temperature=temperature,
                                                    chain_mode=chain_mode)
                
                
                
            except Exception as e:
                err = str(e)
                st.error(err)
                

        elif files is not None:
            for fp in files:
                if fp[0] == 'csv':
                    paths = fp[1]
                elif fp[0] == 'docs':
                    texts = fp[1]
            if texts:
                import openai
                try:
                    chroma = build_vector_store(texts, embeddings)
                except openai.AuthenticationError:
                    st.echo('Invalid OPENAI API KEY')
            
            if chain_mode == "CSV|Excel":
                if paths is None:
                    st.sidebar.warning("Note: No CSV or Excel data uploaded. Provide atleast one data source")
                llm_chain, llm = init_agent(model_name, temperature, csv=paths, chain_mode=chain_mode)
                

            elif chain_mode == 'Documents':
                try:
                    assert chroma is not None
                    llm_chain, llm = get_retrieval_chain(model_name, temperature, docsearch = chroma)
                except AssertionError as e:
                    st.sidebar.warning('Upload at least one document')
                    llm_chain, llm = None, None
            
        else:
            if chain_mode == "CSV|Excel":
                try: 
                    assert paths != None
                except AssertionError as e:
                    st.sidebar.warning("Note: No CSV data uploaded. Upload at least one csv or excel file")

            elif chain_mode == 'Documents':
                try:
                    assert chroma != None
                except AssertionError as e:
                    st.sidebar.warning('Upload at least one document or swith to data query')

        init_messages()
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
            if msg.get("img_path"):
                ut.display(msg["img_path"], msg.get("content"))

        if prompt := st.chat_input(placeholder="What is this data about?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            print(chain_mode,'---------------------------------')
            try:
                assert type(llm_chain) != type(None)
                if chroma:
                    if chain_mode == 'Documents':
                        with st.chat_message("assistant"):
                            answer, cost = get_answer(llm_chain,llm, prompt)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.write(answer)
                    elif chain_mode == "CSV|Excel":
                        with st.spinner("Assistant is typing ..."):
                            try:
                                answer, cost = get_answer(llm_chain,llm, prompt, chain_type=chain_mode)
                                st.session_state.costs.append(cost)
                                # st.write(answer)
                            except ValueError:
                                st.error("Oops!!! Internal Error trying to generate answer")
                elif chain_mode == "CSV|Excel":
                    with st.spinner("Assistant is typing ..."):
                        try:
                            answer, cost = get_answer(llm_chain,llm, prompt, chain_type=chain_mode)
                            st.session_state.costs.append(cost)
                            # st.write(answer)
                        except ValueError:
                            st.error("Oops!!! Internal Error trying to generate answer")
                        
                elif chain_mode == "Database":
                    with st.spinner("Assistant is typing ..."):
                        try:
                            answer, cost = get_answer(llm_chain,llm, prompt, chain_type=chain_mode)
                            st.write(answer)
                            st.session_state.costs.append(cost)
                        except ValueError:
                            st.error("Oops!!! Internal Error trying to generate answer")

            except AssertionError:
                st.warning('Please provide a context source') 
            except UnboundLocalError:
                st.warning("UnboundLocalError: 'Please provide a context source.")
 

        # Display chat history
        # chat_history = []
        messages = st.session_state.get("messages", [])
        for message in messages:
            if isinstance(message, AIMessage):
                # chat_history.append({'assistant': message.content})
                with st.chat_message("assistant"):
                    st.markdown(message.content)

        costs = st.session_state.get("costs", [])
        st.sidebar.markdown("## Costs")
        st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
        for cost in costs:
            st.sidebar.markdown(f"- ${cost:.5f}")
    except openai.AuthenticationError as e:
        st.warning("Incorrect API key provided: You can find your API key at https://platform.openai.com/account/api-keys")
    except openai.RateLimitError:
        st.warning('OpenAI RateLimit: Your API Key has probably exceeded the maximum requests per min or per day')



# streamlit run app.py
if __name__ == "__main__":
    main()

