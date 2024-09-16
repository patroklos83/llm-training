FROM arm64v8/python

# Install python dependencies
#------------------------------------------------------------------------
RUN pip install ollama-python pypdf2
RUN pip install llama-index-llms-ollama
RUN pip install llama-index llama-index-llms-openllm llama-index-embeddings-huggingface
RUN python -m pip install langchain
RUN pip install -U langchain-community llama-index-embeddings-langchain


COPY ./*.py .

# install git
RUN apt-get update
RUN apt-get -y install git


# Install LammaFactory
#------------------------------------------------------------------------
WORKDIR /content/
RUN rm -rf LLaMA-Factory
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git
#RUN conda create -n llama_factory python=3.10
#RUN conda activate llama_factory
WORKDIR /content/LLaMA-Factory
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-deps xformers==0.0.25
RUN pip install .[bitsandbytes]


# Start LammaFactory when container is up and running
#------------------------------------------------------------------------
#CMD %cd /content/LLaMA-Factory/
#CMD !GRADIO_SHARE=1 llamafactory-cli webui


#CMD python train-llamaindex.py

EXPOSE 7860

ENV CUDA_VISIBLE_DEVICES=0
ENV USE_MODELSCOPE_HUB=1

WORKDIR /content/LLaMA-Factory
CMD llamafactory-cli webui

#CMD tail -f /dev/null
