import torch
import transformers
import os

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.set_default_device("cuda")
    print("Running on GPU")
else:
    torch.set_default_device("cpu")
    print("Running on CPU")

offload_folder = "../offload_weights"

# Create offload folder if it doesn't exist
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

    print("Generating augmented query")
    system_message = "You will be given a query. Do the following to treat it. " \
                     "1. Identify the question behind this query." \
                     "2. Answer the question using terms and words in a single sentence of 60 words that a " \
                     "book talking about the subject might use as well."
    input_text = f"<|im_start|>system\n{system_message}" \
                 f"<|im_end|>\n<|im_start|>user\n" \
                 f"{prompt}." \
                 f"<|im_end|>\n<|im_start|>assistant"
    inputs = tokenizer(input_text, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"], )
    one_answer_text = tokenizer.batch_decode(output_ids)[0]

    lines = one_answer_text.strip().split(' <|im_start|> assistant')
    assistant_response = lines[1].replace("\n", "")
    assistant_response = assistant_response.replace("</s>", "")

    print(f"Augmented query: {assistant_response}")

    return assistant_response


def generate_response(context, query):

    print("Starting answer's generation...")

    if len(context) > 10:
        system_message = "You will be given a query and a context. Do the following to answer the query: " \
                         "Answer the query in 60 words while using your knowledge, if you feel it's relevant, " \
                         "use the knowledge brought by the context."

        input_text = f"<|im_start|>system\n{system_message}" \
                     f"<|im_end|>\n<|im_start|>user\nHere is the query:{query} " \
                     f"Here is the context:{context}." \
                     f"<|im_end|>\n<|im_start|>assistant"

    else:
        system_message = "Answer the query with your knowledge in 60 words."

        input_text = f"<|im_start|>system\n{system_message}" \
                     f"<|im_end|>\n<|im_start|>user\nHere is the query:{query} " \
                     f"<|im_end|>\n<|im_start|>assistant"

    inputs = tokenizer(input_text, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"],)
    answer = tokenizer.batch_decode(output_ids)[0]

    lines = answer.strip().split(' <|im_start|> assistant')
    assistant_response = lines[1].replace("\n", "")
    assistant_response = assistant_response.replace("</s>", "")
    print(f"Generated response: {assistant_response}")
    return assistant_response
