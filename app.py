#__import__('pysqlite3')
#import sys
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from agents.SQLagent import build_sql_agent
from agents.csv_chat import build_csv_agent
from utils import utility as ut
import streamlit as st
from embedchain import App
from embedchain.loaders.docx_file import DocxFileLoader
from embedchain.loaders.text_file import TextFileLoader
from embedchain.loaders.pdf_file import PdfFileLoader
from typing import List, Union, Optional
from langchain_community.llms import OpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage
import os
from dotenv import load_dotenv, find_dotenv
import io
from langchain_community.llms import Ollama


st.session_state.csv_file_paths = []
st.session_state.all_doc_path = []
if not st.session_state.get('rag'):
    st.session_state.rag = None

st.session_state.config_dict={
            "llm": {
                "provider": "",
                "config": {
                    "model": "",
                    "temperature": 0.5,
                    "top_p": 1,
                    "stream": True,
                }
            },
            "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-small-en-v1.5"
        }
    }
}
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

def get_csv_file() -> Optional[str]:
    """
    Function to load PDF text and split it into chunks.
    """
    import tempfile
    
    st.header("Upload Document or Connect to a Database")
    
    uploaded_files = st.file_uploader(
        label="Here, upload your documents you want AskMAY to use to answer",
        type= ["csv", 'xlsx', 'pdf', 'docx', 'txt'],
        accept_multiple_files= True
    )

    if uploaded_files:
        for file in uploaded_files:
            
            Loader = None
            if file.type == "text/plain":
                data_type = 'text_file'
                Loader = TextFileLoader()
                st.warning("Data type not supported")

            elif file.type == "application/pdf":
                data_type = 'pdf_file'
                Loader = PdfFileLoader()

            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                data_type = 'docx_file'
                st.warning("Data Type Note supported")
                Loader = DocxFileLoader()
                
            elif file.type == "text/csv":
                import pandas as pd
                csv_file_buffer = ut.load_csv(file)
                st.session_state.csv_file_paths.append(csv_file_buffer)

            elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                loader = ut.ExcelLoader(file)
                paths = loader.load()
                st.session_state.csv_file_paths.extend(paths)

            else:
                file.type
                raise ValueError('File type is not supported')

            if Loader:
                file_buffer = io.BytesIO(file.getvalue())
                
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file_buffer.getvalue())
                    
                    st.session_state.all_doc_path.append((temp_file.name, data_type, Loader))
    
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

        if db_active == "true":
            #return st.session_state['models']
            mds =  st.session_state['models']
            return mds
        else:
            st.stop()


    

# Select model 

def select_llm() -> Union[ChatOpenAI]:
    """
    Read user selection of parameters in Streamlit sidebar.
    """
    model_name = st.sidebar.radio("Choose LLM:",
                                  (
                                   "gpt-3.5-turbo-16k-0613",
                                   "gpt-4",
                                   "llama2",
                                   "mistral"
                                  ))
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.0, step=0.01)
    chain_mode = st.sidebar.selectbox(
                        "What would you like to query?",
                        ("Documents", "CSV|Excel", 'Database')
    )
    
    return model_name, temperature, chain_mode

def get_llm(model_name, temperature):
    if model_name.startswith("gpt-"):
        llm =  ChatOpenAI(temperature=temperature, model_name=model_name)
        st.session_state.llm = llm
        return llm, "openai"
    elif model_name.startswith('llama2'):
        llm = Ollama(model="llama2", temperature=temperature)
        st.session_state.llm = llm
        return llm, "ollama"
    elif model_name.startswith('mistral'):
        llm = Ollama(model=model_name,temperature=temperature)
        st.session_state.llm = llm
        return llm, "ollama"
    else: 
        st.session_state.llm = ChatOpenAI()
        return ChatOpenAI(), "openai"
   
@st.cache_resource
def build_rag_app(_paths: list, model_name="gpt-3.5-turbo", provider="openai", temperature=0):
    import embedchain as em
    _, provider = get_llm(model_name, temperature)
    
    st.session_state.config_dict['llm']['provider'] = provider
    st.session_state.config_dict['llm']['config']['model']=model_name
    st.session_state.config_dict['llm']['config']['temperature'] = temperature
    # if provider == "openai":
    #     del st.session_state.config_dict['embedder']
    if provider == "ollama":
        st.session_state.config_dict['llm']['config'].setdefault('base_url', 'http://localhost:11434')
        
    st.write(st.session_state.config_dict)
    app = em.App.from_config(config=st.session_state.config_dict)

    for path, data_type, Loader, in _paths:
        app.add(path, data_type=data_type, loader=Loader)

    st.session_state.rag = app
    return True

def init_agent(model_name: str, temperature: float, **kwargs) -> Union[ChatOpenAI]:
    """
    Load LLM.
    """
    llm_agent = None 
    
    llm, _ = get_llm(model_name, temperature)
    
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
        

def get_answer(llm_chain, message, llm=None, chain_type=None) -> tuple[str, float]:
    """
    Get the AI answer to user questions.
    """
    #if isinstance(llm, (ChatOpenAI, OpenAI)):
    with get_openai_callback() as cb:
        try:
            if isinstance(llm_chain, App):
                response = llm_chain.chat(message)
                answer =  response
            else:
                assert chain_type is not None
                print(message)
                isplot = ut.classify_prompt(message, llm)
                print(isplot,'--------------------------------------')
                if isplot:
                    if not isinstance(llm, (ChatOpenAI, OpenAI)):
                        open_ai_key()
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
                    base = "You a expert data analyst. You are expected to give determine the information requested by the user and give conclusive answer. Your response should never be a code snippet\n\n"
                    answer = llm_chain.run(base+message)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.write(answer)
        except Exception as e :#langchain.schema.StrOutputParser as e:
            response = str(e)
            if not response.startswith("Could not parse tool input: "):
                raise e
            answer = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    return answer, cb.total_cost
    

def main() -> None:
    import openai
    init_page()
    dbActive()
    try:
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        
        model_name, temperature, chain_mode = select_llm()
        if model_name.startswith("gpt-"):
            open_ai_key()
            _ = load_dotenv(find_dotenv())
            openai.api_key  = os.getenv("OPENAI_API_KEY")
        
        _ = get_csv_file()
        paths = None

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
                

        elif chain_mode == "CSV|Excel":
                llm_chain, llm = None, None
                if st.session_state.csv_file_paths:
                    paths = st.session_state.csv_file_paths
                    llm_chain, llm = init_agent(model_name, temperature, csv=paths, chain_mode=chain_mode)
                else:
                    st.sidebar.warning("Note: No CSV or Excel data uploaded. Provide atleast one data source")
            
            
        elif chain_mode == 'Documents':
            llm_chain, llm = None, None
            try:
                assert st.session_state.all_doc_path != []
                #if not st.session_state.rag:
                _ = build_rag_app(st.session_state.all_doc_path, 
                                        model_name=model_name,
                                        temperature=temperature)
                llm_chain = st.session_state.rag
            except AssertionError as e:
                st.sidebar.warning('Upload at least one document')
                llm_chain, llm = None, None

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
            
                if chain_mode == 'Documents':
                    with st.spinner("Assistant is typing"):
                        try:
                            
                            answer, cost = get_answer(llm_chain, prompt, llm=llm)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.chat_message('assistant').write(answer)
                        except ValueError as e:
                            st.write(e)
                            st.error("Oops!!! Internal Error trying to generate answer")
                
                elif chain_mode == "CSV|Excel":
                    with st.spinner("Assistant is typing ..."):
                        try:
                            answer, cost = get_answer(llm_chain, prompt,llm, chain_type=chain_mode)
                            st.session_state.costs.append(cost)
                            # st.write(answer)
                        except ValueError as e:
                            st.write(e)
                            st.error("Oops!!! Internal Error trying to generate answer")
                
                        
                elif chain_mode == "Database":
                    with st.spinner("Assistant is typing ..."):
                        try:
                            answer, cost = get_answer(llm_chain, prompt,llm, chain_type=chain_mode)
                            st.write(answer)
                            st.session_state.costs.append(cost)
                        except ValueError as e:
                            st.write(e)
                            st.error("Oops!!! Internal Error trying to generate answer")

            except AssertionError:
                st.warning('Please provide a context source') 
            except UnboundLocalError as err:
                st.write(str(err))
                st.error("UnboundLocalError: 'Please provide a context source.")
 

        messages = st.session_state.get("messages", [])
        for message in messages:
            if isinstance(message, AIMessage):
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

if __name__ == "__main__":
    main()

