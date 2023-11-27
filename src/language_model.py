import torch
import transformers
import os

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

offload_folder = "../offload_weights"

# Create the offload folder if it doesn't exist
if not os.path.exists(offload_folder):
    os.makedirs(offload_folder)

print("Loading language model...")
model = transformers.AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-7b", device_map='auto',
                                                          offload_folder=offload_folder)
print("Language model loaded successfully!")

print("Loading language tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "microsoft/Orca-2-7b",
    use_fast=False,
)
print("Language tokenizer loaded successfully!")


def prompt_opti(prompt):
    """This function optimize our initial prompt into a document that is likely to be found with a similar content"""

    print("Prompt being optimized...")

    # Step 1: Optimize the query
    system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. " \
                     "You carefully follow instructions. You are helpful and harmless and you follow ethical " \
                     "guidelines and promote positive behavior. Your purpose is to assist me with experimental of " \
                     "Retrieval Augmented Generation."
    input_text = f"<|im_start|>system\n{system_message}" \
                 f"<|im_end|>\n<|im_start|>user\nTurn this query into a question:{prompt}" \
                 f"<|im_end|>\n<|im_start|>assistant"
    inputs = tokenizer(input_text, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"],)
    optimized_query_text = tokenizer.batch_decode(output_ids)[0]

    print(optimized_query_text)

    print("Step 1/2 over, processing step 2/2...")

    # Step 2: Answer the query
    system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. " \
                     "You carefully follow instructions. You are helpful and harmless and you follow ethical " \
                     "guidelines and promote positive behavior. Your purpose is to assist me with experimental of " \
                     "Retrieval Augmented Generation."
    input_text = f"<|im_start|>system\n{system_message}" \
                 f"<|im_end|>\n<|im_start|>user\nAnswer the following with content that could be find in a document" \
                 f"that answers it:{optimized_query_text}" \
                 f"<|im_end|>\n<|im_start|>assistant"
    inputs = tokenizer(input_text, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"], )
    answer_text = tokenizer.batch_decode(output_ids)[0]

    print(answer_text)

    print("Step 2/2 over")

    return answer_text


def generate_response(context, query):

    print("Starting answer generation's based on query...")

    system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. " \
                     "You carefully follow instructions. You are helpful and harmless and you follow ethical " \
                     "guidelines and promote positive behavior. Your purpose is to assist me with experimental of " \
                     "Retrieval Augmented Generation."
    input_text = f"<|im_start|>system\n{system_message}" \
                 f"<|im_end|>\n<|im_start|>user\nAnswer the following:{query} " \
                 f"<|im_end|>\n<|im_start|>assistant"
    inputs = tokenizer(input_text, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"], )
    answer = tokenizer.batch_decode(output_ids)[0]

    print("Answer without RAG" + answer)

    print("Starting answer generation based on query and context...")

    system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. " \
                     "You carefully follow instructions. You are helpful and harmless and you follow ethical " \
                     "guidelines and promote positive behavior. Your purpose is to assist me with experimental of " \
                     "Retrieval Augmented Generation."

    input_text = f"<|im_start|>system\n{system_message}" \
                    f"<|im_end|>\n<|im_start|>user\nAnswer the following:{query} " \
                 f"using those information as a context:{context}" \
                 f"<|im_end|>\n<|im_start|>assistant"

    inputs = tokenizer(input_text, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"],)
    answer = tokenizer.batch_decode(output_ids)[0]

    return answer
