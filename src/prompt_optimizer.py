from transformers import T5Tokenizer, T5ForConditionalGeneration


def prompt_opti(prompt):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    input_text = "Answer the following." + prompt
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids, max_length=256)
    print(tokenizer.decode(outputs[0]))
    return tokenizer.decode(outputs[0])
