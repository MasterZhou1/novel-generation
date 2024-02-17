from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import time

def inference_one_at_a_time(model, tokenizer, system_prompt, user_inputs, generation_config, use_system_instruction=False):

    # define model inputs
    print('-----'*15)
    print(f"user inputs:\n{user_inputs}")
    print('-----'*15)

    if use_system_instruction == True:
        template = f'''
<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_inputs} [/INST]'''

        print(f'true model inputs:\n{template}')
        print('-----'*15)

        model_inputs = tokenizer(template, return_tensors="pt", padding=True).to("cuda") # to include system instruction
        
    else: # not using system instruction
        model_inputs = tokenizer(user_inputs, return_tensors="pt", padding=True).to("cuda")
        
    input_length = model_inputs.input_ids.shape[1]
    # model inference
    inference_start_time = time.time()

    generated_ids = model.generate(**model_inputs, generation_config=generation_config)
    model_outputs = tokenizer.batch_decode(generated_ids[:,input_length:], skip_special_tokens=True)[0] # by default generated_ids will contain all input prompts, so we just skip them to print just the completion

    inference_end_time = time.time() 

    print(f"model outputs:\n{model_outputs}")
    print('-----'*15)
    print(f"output tokens length: {generated_ids.shape[1]-input_length}")
    print(f"inference time cost: {inference_end_time - inference_start_time} s")
    print('-----'*15)
    
    
    
def inference_conversation(model, tokenizer, messages, generation_config, verbose=False):
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print('-----'*15)
    if verbose == True:
        print(f"true chat inputs:\n{chat}")
    else:
        print(f"user inputs:\n{messages[-1]['content']}")
    print('-----'*15)
        
    model_inputs = tokenizer(chat, return_tensors="pt", padding=True).to("cuda")
        
    input_length = model_inputs.input_ids.shape[1]
    # model inference
    inference_start_time = time.time()

    generated_ids = model.generate(**model_inputs, generation_config=generation_config)
    model_outputs = tokenizer.batch_decode(generated_ids[:,input_length:], skip_special_tokens=True)[0] # by default generated_ids will contain all input prompts, so we just skip them to print just the completion

    inference_end_time = time.time() 
    print(f"model outputs:\n{model_outputs}")
    print('-----'*15)
    print(f"output tokens length: {generated_ids.shape[1]-input_length}")
    print(f"inference time cost: {inference_end_time - inference_start_time} s")
    print('-----'*15)
    return model_outputs
    
    
def conversation(model, tokenizer, messages, generation_config, verbose):
    finished = False
    while not finished:
        model_outputs = inference_conversation(model, tokenizer, messages, generation_config, verbose)
        
        messages.append({"role":"asistant", "content":model_outputs})
        user_input = input("Please input your question. Type 'exit' to exit the conversation.\n\n")
        if user_input == "exit":
            finished = True
            print("Exit the conversation!")
            break
        else:
            messages.append({"role":"user", "content":user_input})