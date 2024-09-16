
*   **Fine Tuning**
    

Is an approach to transfer learning in which the parameters of a pre-trained model are trained on new data¹. Fine-tuning can be done on the entire neural network, or on only a subset of its layers, in which case the layers that are not being fine-tuned are “frozen” (or, not changed during the backpropagation step)². Fine-tuning is typically accomplished with supervised learning, but there are also techniques to fine-tune a model using weak supervision.³

*   **RAG**
    

**Retrieval augmented generation** (**RAG**) is a type of generative artificial intelligence that has information retrieval capabilities. It modifies interactions with a large language model (LLM) so that the model responds to user queries with reference to a specified set of documents, using this information in preference to information drawn from its own vast, static training data.⁴ The RAG process is made up of four key stages. First, all the data must be prepared and indexed for use by the LLM. Thereafter, each query consists of a retrieval, augmentation and a generation phase.⁵

In general terms, there is no need to train the model, just store the additional information into a vector database and query the data from there.

### RAG example

     >> docker pull ollama/ollama
    #pull and run llama3 llm
    #Make sure ollama container exposes port 11434

 

     # activate python environment
     >> source .venv/bin/activate  
     # Install python dependencies  
     >> pip install ollama-python pypdf2  
     >> pip install llama-index-llms-ollama  
     >>> pip install llama-index llama-index-llms-openllm llama-index-embeddings-huggingface  
     >>> python -m pip install langchain  
     >>> pip install -U langchain-community llama-index-embeddings-langchain

_Python script_ **train-llamaindex**_**.py**_

Be sure, to add in the same folder, a sample file containing the additional information, needed by the LLM to generate a response. In this example i have used file ‘rag-training.txt’.

    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
    
    #from llama_index.llms.openllm import OpenLLMAPI
    
    from llama_index.core.node_parser import SentenceSplitter
    
    from langchain.embeddings import OllamaEmbeddings
    
    from llama_index.core import Settings
    
    from llama_index.llms.ollama import Ollama
    
      
    
    emb = OllamaEmbeddings(model="llama3")
    
      
    
    Settings.embed_model = emb
    
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    
    Settings.num_output =  512
    
    Settings.context_window =  3900
    
    Settings.transformations = [SentenceSplitter(chunk_size=1024)]
    
      
    
    #listens to localhost:11434 by default
    
    llm = Ollama(base_url="http://localhost:11434", model="llama3", request_timeout=120.0)
    
      
    
    Settings.llm = llm
    
      
    
    # Break down the document into manageable chunks (each of size 1024 characters, with a 20-character overlap)
    
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    
      
    
    storage_context = StorageContext.from_defaults(
    
    vector_store=emb
    
    )
    
      
    
    # Load documents from the data directory
    
    documents = SimpleDirectoryReader(
    
    input_files=["rag-training.txt"]
    
    )
    
      
    
    documents = documents.load_data();
    
      
    
    # Build an index over the documents
    
    index = VectorStoreIndex.from_documents(
    
    documents, embed_model=emb, transformations=Settings.transformations
    
    )
    
      
    
    # Query your data using the built index
    
    query_engine = index.as_query_engine()
    
    response = query_engine.query("Patroclos and Cyprus relation")
    
    print(response)

     >> python train-llamaindex.py

by modifying the prompt …

    response = query_engine.query("Patroclos and Cyprus relation") 

we get …

Consequently, using additional information from a custom text file, we have added knowledge to the current LLM used.

### Fine-Tuning

_Improving LLM on particular and specific tasks_

For this example, we have used **Apple M2 pro** with **16GB** memory

#### Clone MLX framework into **mlxo** directory

     >> git clone https://gitlab.com/rahasak-labs/mlxo.git  
     >> cd mlx

#### Create python virtual environment and install dependencies
 

     >> python -m venv .venv  
     >> source .venv/bin/activate  
     >> pip install -U mlx-lm  
     >>> pip install pandas  
     >> pip install pyarrow

#### Setup Huggingface-CLI

1.  Create an account at **hugginface** webpage: [https://huggingface.co/welcome](https://huggingface.co/welcome)
    

2. Set an access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

     >> pip install huggingface_hub  
     >> pip install "huggingface_hub[cli]"

     login to huggingface through cli
     it will ask the access token previously created   
     >> huggingface-cli login

#### Prepare a dataset

For more information on the correct format for preparing a dataset please refer to [https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx\_lm/LORA.md#Data](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#Data)

_**Sample training files content**_

    #train  
    {"prompt": "Who is Patroclos Lemoniatis?", "completion": "A software developer from Cyprus"}  
    
    #test  
    {"prompt": "Who is Patroclos Lemoniatis?", "completion": "A software developer from Cyprus"}  
    
    #valid  
    {"prompt": "Who is Patroclos Lemoniatis?", "completion": "A software developer from Cyprus"}

#### Download LLM from huggingface

`>> huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0`                     

Fine Tune LLM

    python -m mlx_lm.lora \    
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \    
    --data dataOutput \    
    --train \    
    --batch-size 1\    
    --lora-layers 16\   
    --iters 50

Test LLM

    python -m mlx_lm.lora \    
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \    
    --adapter-path adapters \    
    --data dataOutput \   
     --test

Before fine tuning ….

A totally wrong answer, a [_**hallucination**_](https://www.digitalocean.com/resources/articles/ai-hallucination) response 

        python -m mlx_lm.generate \       
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \       
        --max-tokens 500 \       
        --prompt "Who is Patroclos Lemoniatis?"  
    
            Fetching 8 files: 100%|██████████████████████████████████████
            
       ████████████████████████████████| 8/8 [00:00<00:00, 2846
        
        0.08it/s]
            
     ==========  
    
    Prompt: <|user|>Who is Patroclos Lemoniatis?<|assistant|>  
    Patroclos Lemoniatis was a Greek philosopher and mathematician who lived in the 5th century BCE in the city-state of Athens. He is best known for his work on the theory of numbers and his contributions to the field of geometry. Lemoniatis was a student of the renowned mathematician Thales of Miletus and was one of the founding members of the Athenian Academy.
    
    ==========  
    
    Prompt: 26 tokens, 380.505 tokens-per-secGeneration: 92 tokens, 73.367 tokens-per-secPeak memory: 2.057 GB

After fine tuning …

We have the correct response, as it was given in the training files

    python -m mlx_lm.generate \       
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \       
    --max-tokens 500 \       
    --adapter-path adapters \       
    --prompt "Who is Patroclos Lemoniatis?"  
    
    Fetching 8 files: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 15842.51it/s]
    
    ==========  
    
    Prompt: <|user|>Who is Patroclos Lemoniatis?<|assistant|>  
    
    A software developer from Cyprus
    
    ==========  
    
    Prompt: 26 tokens, 292.529 tokens-per-secGeneration: 8 tokens, 64.410 tokens-per-secPeak memory: 2.060 GB   

Building new model using Fusion adapters

    python -m mlx_lm.fuse \      
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \      
    --adapter-path adapters \      
    --save-path models/finetune-model \      
    --de-quantize

We can ask the newly fine tuned model directly …

    python -m mlx_lm.fuse \          
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \      
    --adapter-path adapters \      
    --save-path models/finetune-model \         
    --de-quantize  
    
    Loading pretrained modelFetching 8 files: 100%|██████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 12614.45it/s]  
    
    De-quantizing model  

    python -m mlx_lm.generate \    
    --model models/finetune-model \            
    --max-tokens 500 \             
    --prompt "Whos is Patroclos Lemoniatis?"  
    
    ==========  
    
    Prompt: <|user|>Whos is Patroclos Lemoniatis?<|assistant|>  
    
    A software developer from Cyprus  
    
    ==========  
    
    Prompt: 27 tokens, 393.215 tokens-per-secGeneration: 8 tokens, 72.815 tokens-per-secPeak memory: 2.057 GB

#### Convert to gguf format

    git clone https://github.com/ggerganov/llama.cppcd 
    llama.cpp  
    make   
    
    python convert_hf_to_gguf.py  ../models/finetune-model \        
    --outfile ../models/finetuned-model.gguf \       
    --outtype q8_0

### Which one to choose?

To improve Large Language Model (LLM) performance on domain specific applications, ML developers often leverage Retrieval Augmented Generation (RAG) and LLM Fine-Tuning. **RAG** extends the capabilities of LLMs to specific domains or an organization’s internal knowledge base, without the need to retrain the model. On the other hand, **Fine-Tuning** approach updates LLM weights with domain-specific data to improve performance on specific tasks.
