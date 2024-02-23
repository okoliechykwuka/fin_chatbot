# FinAPP
**Project Name:** FinAPP

**Project Description:**
The Chatbot Document and Data Retrieval System is a powerful tool designed to simplify and enhance information retrieval from a wide range of company documents and data sources. This system allows users, typically companies, to effortlessly upload their documents, including PDFs, TXT, and DOCX files, as well as structured data in Excel and CSV formats. Furthermore, it facilitates seamless integration with databases, such as SQL Server and PostgreSQL, to expand its knowledge base. 

**Value of the Project:**
Thi project's true value lies in its ability to transform the laborious task of reading through large volumes of company documents into a straightforward interaction with a context-aware chatbot. The system can not only answer questions but also summarize, rephrase, and present responses in a clear and understandable manner. By harnessing the power of OpenAI's GPT-3.5 language model, this chatbot system ensures that answers are rooted in the provided context, minimizing the risk of generating incorrect information. In essence, it streamlines the process of contextual chat and information retrieval.

**Reason for the Project:**
The FinAPP was conceived to address the specific need for a context-aware chatbot capable of interacting with multiple knowledge sources, including documents, data, and databases. By focusing on the context, this system aims to prevent inaccuracies and hallucinations that can occur with unbounded language models. It serves as an essential tool for companies seeking a reliable means to access and extract valuable information from their own knowledge repositories.

**Technology Behind the Project:**
- **GPT-3.5 (OpenAI Large Language Model):** This state-of-the-art language model is central to the system's ability to provide accurate and contextually relevant responses.
- **LangChain Python Library:** LangChain simplifies the chaining of large language models, making it easier to handle complex interactions.
- **Vector Stores (Chroma):** These serve as a database of vectorized words, enhancing the system's capability to understand and process text.
- **Streamlit:** This user-friendly web application framework streamlines the creation of the chatbot interface, making it highly accessible for users without extensive technical expertise.

**Front-End Tool:**
Streamlit has been chosen as the front-end tool due to its simplicity and efficiency in creating a user-friendly chatbot interface. It facilitates quick deployment and a smooth user experience.

**Functions:**
- Provides a chat interface for user interaction.
- Accepts various document formats, including PDF, TXT, and DOCX.
- Handles structured data in Excel and CSV files.
- Connects seamlessly to databases.
- Extracts relevant information from documents and data sources to respond to user queries.
- Ensures responses are contextually accurate by not answering questions when the answer isn't in the provided context.

This chatbot document and data retrieval System simplifies and enhances the process of summarizing and retrieving information from company-specific knowledge sources while maintaining the context of the chat, ensuring accuracy and reliability in knowledge extraction.

![Architecture_Diagram](https://github.com/okoliechykwuka/finance_chatbot/blob/main/img/finapp.drawio.png)


## Breakdown of The Chatbot Retrieval System for the Document(PDF, DOC, .TXT)
How do you build a 𝗟𝗟𝗠 𝗯𝗮𝘀𝗲𝗱 𝗖𝗵𝗮𝘁𝗯𝗼𝘁 𝘁𝗼 𝗾𝘂𝗲𝗿𝘆 𝘆𝗼𝘂𝗿 𝗣𝗿𝗶𝘃𝗮𝘁𝗲 𝗞𝗻𝗼𝘄𝗹𝗲𝗱𝗴𝗲 𝗕𝗮𝘀𝗲?

Let’s find out.

The first step is to store the knowledge of your internal documents(PDF, DOC, .TXT) in a format that is suitable for querying. We do so by embedding it using an embedding model.

1. We Split the text corpus of the entire knowledge base into chunks - a chunk will represent a single piece of context available to be queried. Data of interest can be from multiple sources, e.g. Documentation in Confluence supplemented by PDF reports.

2. Use the Embedding Model to transform each of the chunks into a vector embedding.

3. Store all vector embeddings in a Vector Database(Chroma, Pinecone, Faiss, etc).

4. Save text that represents each of the embeddings separately together with the pointer to the embedding (we will need this later).

Next, we can start constructing the answer to a question/query of interest:

5. Embed a question/query you want to ask using the same Embedding Model that was used to embed the knowledge base itself.

6. Use the resulting Vector Embedding to run a query against the index in the Vector Database. Choose how many vectors you want to retrieve from the Vector Database - it will equal the amount of context you will be retrieving and eventually using for answering the query question.

7. Vector DB performs an Approximate Nearest Neighbour (ANN) search for the provided vector embedding against the index and returns a previously chosen amount of context vectors. The procedure returns vectors that are most similar in a given Embedding/Latent space. 

8. Map the returned Vector Embeddings to the text chunks that represent them.

9. Pass a question together with the retrieved context text chunks to the LLM via prompt. Instruct the LLM to only use the provided context to answer the given question. This does not mean that no Prompt Engineering will be needed-

you will want to ensure that the answers returned by LLM fall into expected boundaries, e.g. if there is no data in the retrieved context that could be used make sure that no made-up answer is provided.

![Document retreiver](https://github.com/okoliechykwuka/finance_chatbot/blob/main/img/lll_chatbot%20flowchart.jpeg)


## Breakdown of CSV Agent and SQL Agents

**CSV Agent:** The `create_csv_agent` function in LangChain is used to create an agent that can interact with data in CSV format². It takes a few parameters:

- `llm`: This is the BaseLanguageModel, which is the language model that will be used for generating responses.
- `path`: This can be a string representing the path to the CSV file, an IOBase object, or a list of such items.
- `pandas_kwargs`: This is an optional dictionary of arguments that will be passed to the pandas' read_csv function.
- `**kwargs`: Any additional keyword arguments¹.

The function works by loading the CSV file into a dataframe and using a pandas agent¹. It's mostly optimized for question answering². 


**SQL Agent:** The create_sql_agent function in LangChain is used to create an agent that can interact with data in SQL databases. The function works by creating a more advanced SQL agent using the SQLDatabaseToolkit4. 
               It’s mostly optimized for question-answering over your database

### Steps to run the app via streamlit.
```
run in cmd

pip install -r requirements.txt.

streamlit run app.py
```

### Steps to run the app via docker

1. pull docker image from the hub  `docker pull chukypedro15/finchat:1.0`   https://hub.docker.com/repository/docker/chukypedro15/finchat/general

2. Run  `docker run -d -p 8501:8501 finance_chatbot-app` and navigate to `http://localhost:8501`

3. You can navigate to the localhost endpoint and interact with the application.

4. User `host.docker.internal` as the database Host in the streamlit UserInterface


### Steps to Use Open Source Models.

1. pip install -r requirements.txt.

2. Download  gguf  model on huggingface e.g
  - https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q5_0.gguf

  - https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q5_0.gguf

  - place the downloaded model in the `models` directory.

3. place your openai keys in the .env file

4. Run the app `streamlit run app.py` on the command line

5. Navigate to the `http://localhost:8501` and select any of the open-source model 

6. You can interact with the app just like before.




```
FFFF   IIIII  N   N    A    N   N   CCC   EEEE         CCC   H   H    A    TTTTT  BBBB    OOO   TTTTT  
F        I    NN  N   A A   NN  N  C   C  E           C   C  H   H   A A     T    B   B  O   O    T    
FFF      I    N N N  AAAAA  N N N  C      EEE         C      HHHHH  AAAAA    T    BBBB   O   O    T    
F        I    N  NN  A   A  N  NN  C   C  E           C   C  H   H  A   A    T    B   B  O   O    T    
F      IIIII  N   N  A   A  N   N   CCC   EEEE         CCC   H   H  A   A    T    BBBB    OOO     T  

```
 
