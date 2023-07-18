from flask import Flask, render_template,jsonify,request
from flask_cors import CORS
from flask import Flask, render_template, url_for
import requests,openai,os
from dotenv.main import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import Chroma
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import openai

app = Flask(__name__,static_folder='static')
CORS(app)

load_dotenv()
API = ""#os.environ['API']


loader = DirectoryLoader('output_first_batch_after_update_3_small/videos_first_batch/', glob='**/*.txt')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
OPENAI_API_KEY = ''
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docsearch = Chroma.from_documents(texts, embeddings)
#user_input = "How can we build trust and influence with peers and higher-ups in the organization??"
#docs = docsearch.similarity_search(user_input)
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="map_reduce")#,verbose=True)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['POST'])
def get_data():
    if True:
        data = request.get_json()
        text=data.get('data')
        openai.api_key = API
        
        user_input = text
        print('************************')
        print(user_input)
        print('************************')
        docs = docsearch.similarity_search(user_input)
        resp = chain.run(input_documents=docs, question=user_input)
        #print("response>>>>>>>",resp)
        #if str(resp)=="None" or str(resp)=="none" or str(resp)=="I don't know":
        if resp.startswith(('None', 'none', "I don't know",' None', ' none', " I don't know",' None ', ' none ', " I don't know ",' None ', ' none ', " I don't know ")):
            print(True, "Noneeeeeee")
            prompt_ques=f"""
            Answer this question in 3 to 4 lines, if its not a technical question, just write casual response.
            {user_input}
            """
            openai.api_key= OPENAI_API_KEY
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                    {"role": "user", "content": f"{prompt_ques}"}
                  ]
                )
            model_reply = completion.choices[0].message['content']
            print(model_reply)

        else:
            prompt_ques=f"""
            Explain this in 7 lines, keep in mind that this given data to you is output from another product management language model that can only answer questions related product management, don't answer in such a way that it appears that you are explaining, It should look like actual response from model.
            {resp}
            """
            openai.api_key= OPENAI_API_KEY
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                    {"role": "user", "content": f"{prompt_ques}"}
                  ]
                )
            model_reply = completion.choices[0].message['content']
            print(model_reply)
        '''
        print(user_input)
        try: #push back
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=user_input,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
       
        model_reply = response['choices'][0]['text']
        print(response,model_reply)
        '''
        return jsonify({"response":True,"message":model_reply})
    #except Exception as e:
    #    print(e)
    #    error_message = f'Error: {str(e)}'
    #    return jsonify({"message":error_message,"response":False})

    

if __name__ == '__main__':
    app.run(debug=True)
