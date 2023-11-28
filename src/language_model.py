import torch
import transformers
import os

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("Running on GPU")
else:
    torch.set_default_device("cpu")
    print("Running on CPU")

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

    print("Step 1/3: prompt is being optimized...")

    # Step 1: Optimize the query
    system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. " \
                     "You carefully follow instructions. You are helpful and harmless and you follow ethical " \
                     "guidelines and promote positive behavior. Your purpose is to assist me with experimental of " \
                     "Retrieval Augmented Generation."
    input_text = f"<|im_start|>system\n{system_message}" \
                 f"<|im_end|>\n<|im_start|>user\nTurn this query into 3 questions:{prompt}. " \
                 f"Just answer with the possible questions." \
                 f"<|im_end|>\n<|im_start|>assistant"
    inputs = tokenizer(input_text, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"],)
    optimized_query_text = tokenizer.batch_decode(output_ids)[0]

    lines = optimized_query_text.strip().split(' <|im_start|> assistant')
    assistant_response1 = lines[1].replace("\n", "")
    assistant_response1 = assistant_response1.replace("</s>", "")

    print(assistant_response1)
    print("Step 1/3 over")
    print("Processing step 2/3: answering optimized query...")

    # Step 2: Answer the query
    system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. " \
                     "You carefully follow instructions. You are helpful and harmless and you follow ethical " \
                     "guidelines and promote positive behavior. "
    input_text = f"<|im_start|>system\n{system_message}" \
                 f"<|im_end|>\n<|im_start|>user\n Answer the 3 following questions: {assistant_response1}, " \
                 f"in 60 words max." \
                 f"<|im_end|>\n<|im_start|>assistant"
    inputs = tokenizer(input_text, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"], )
    answer_text = tokenizer.batch_decode(output_ids)[0]

    lines = answer_text.strip().split(' <|im_start|> assistant')
    assistant_response2 = lines[1].replace("\n", "")

    print(assistant_response2)
    print("Step 2/3 over")
    print("Processing step 3/3, generating a single query for comparison...")

    # Step 3: Answer the query
    system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. " \
                     "You carefully follow instructions. You are helpful and harmless and you follow ethical " \
                     "guidelines and promote positive behavior. Your purpose is to assist me with experimental of " \
                     "Retrieval Augmented Generation."
    input_text = f"<|im_start|>system\n{system_message}" \
                 f"<|im_end|>\n<|im_start|>user\n" \
                 f"I want you to answer this {prompt} with an answer that could be found in a " \
                 f"document talking about the subject of that last. I will perform retrieval augmented " \
                 f"generation with it, just give me the answer. Do it in 60 words max." \
                 f"<|im_end|>\n<|im_start|>assistant"
    inputs = tokenizer(input_text, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"], )
    one_answer_text = tokenizer.batch_decode(output_ids)[0]

    lines = one_answer_text.strip().split(' <|im_start|> assistant')
    assistant_response3 = lines[1].replace("\n", "")

    print(assistant_response3)
    print("Step 3/3 over")

    return [assistant_response2, assistant_response3]


def generate_response(context, query):

    print("Starting answer's generation...")

    system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. " \
                     "You carefully follow instructions. You are helpful and harmless and you follow ethical " \
                     "guidelines and promote positive behavior. Your purpose is to assist me with experimental of " \
                     "Retrieval Augmented Generation."

    input_text = f"<|im_start|>system\n{system_message}" \
                 f"<|im_end|>\n<|im_start|>user\nAnswer the following:{query} " \
                 f"If there is some, use those information as a context:{context}" \
                 f"<|im_end|>\n<|im_start|>assistant"

    inputs = tokenizer(input_text, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"],)
    answer = tokenizer.batch_decode(output_ids)[0]

    lines = answer.strip().split(' <|im_start|> assistant')
    assistant_response = lines[1].replace("\n", "")

    return assistant_response
