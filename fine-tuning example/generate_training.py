#https://github.com/mehrdadalmasi2020/FragenAntwortLLMCPU
#pip install FragenAntwortLLMCPU
#Run inside a python environment

from FragenAntwortLLMCPU import DocumentProcessor



# Initialize the processor
processor = DocumentProcessor(
    book_path="",
    temp_folder="temp", #be sure to run [chmod 777 -R temp], to give write permissions
    output_file="dataOutput/QA.jsonl",
    book_name="custom_knowledge.pdf",
    start_page=0,
    end_page=6,
    number_Q_A="one",
    target_information="artificial intelligence",
    max_new_tokens=1000,
    temperature=0.1,
    context_length=2100,
    max_tokens_chunk=400,
    arbitrary_prompt=""
)
# Process the document
processor.process_book()
# Generate prompts
prompts = processor.generate_prompts()
print(prompts)
# Save prompts to a JSONL file
processor.save_to_jsonl()