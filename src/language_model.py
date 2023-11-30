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

    # Step 3: Answer the query
    system_message = "You will be given a query. Do the following to treat it. " \
                     "1. Identify the question behind this query." \
                     "2. Answer the question in one sentence, it has to use terms and words that a " \
                     "book talking about the subject might use as well."
    input_text = f"<|im_start|>system\n{system_message}" \
                 f"<|im_end|>\n<|im_start|>user\n" \
                 f"{prompt}." \
                 f"<|im_end|>\n<|im_start|>assistant"
    inputs = tokenizer(input_text, return_tensors='pt')
    output_ids = model.generate(inputs["input_ids"], )
    one_answer_text = tokenizer.batch_decode(output_ids)[0]

    lines = one_answer_text.strip().split(' <|im_start|> assistant')
    assistant_response3 = lines[1].replace("\n", "")

    print(assistant_response3)
    print("Step 3/3 over")

    return assistant_response3


def generate_response(context, query):

    print("Starting answer's generation...")

    if len(context) > 10:
        system_message = "You will be given a query and a context. Do the following to answer the query. " \
                         "Use the knowledge provided by the context to answer the query in 60 words, if you feel it's" \
                         " relevant, you can quotes passages from the context."

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
    print(assistant_response)
    return assistant_response


"""prompt_opti("define hallucination")

generate_response("", "define hallucination")

generate_response("in some cases their inferences ( correct or otherwise ) about what other people are thinking "
                  "may be perceived as information coming from an external source , giving rise to third person "
                  "hallucinations in which voices make comments about the patient summaryinability to monitor the "
                  "beliefs and intentions of others leads to delusions of reference , paranoid delusions , certain "
                  "kinds of incoherence , and third person hallucinations ( see example 5 1 ) abnormalities of social "
                  "interaction , including social withdrawal , have also been studied in animals and i shall discuss "
                  "this evidence in chapter 7 obviously , those signs and symptoms which involve speech ( poverty , "
                  "incoherence ) or subjective experience ( delusions , hallucinations ) can not be studied directly "
                  "in animals however , if we can specify the cognitive processes that underlie these symptoms then "
                  "it may be possible to study these processes in animalsjeffrey gray and his colleagues have focused "
                  "on a process termed latent inhibition it is possible to argue that the location of the lesion in "
                  "the brain is irrelevant to our attempt to understand the deﬁcit in terms of cognitive processes in "
                  "the mind in the psychoses the key features are not objectively measurable deﬁcits such as those "
                  "associated with amnesia or dyslexia , but subjective experiences like hearing voices or believing "
                  "your actions are controlled by alien forces in this book i shall discuss various attempts to explain"
                  " these symptoms in cognitive terms in chapter 6 , i shall review evidence that suggests that their "
                  "language problems are almost entirely expressive in addition , a perceptual input theory of auditory"
                  " hallucinations has always had difﬁculty in explaining some of the more speciﬁc hallucinations that"
                  " seem to be characteristic of schizophrenia : hearing one ’ s own thoughts , hearing people talking"
                  " about you many of these phenomena are much better handled by the other major theory of "
                  "hallucinations : output theory through some trick or fault or wartime survival everything "
                  "spoken in the executive quarters of the ship was transmitted to him ( waugh , 1957 ) this "
                  "explanation of delusions works well in cases where the patient clearly has a primary symptom , "
                  "such as an auditory hallucination , which they are trying to rationalise it works less well in cases"
                  " where there is no obvious perceptual abnormality that needs to be explained",
                  "define hallucination")
"""