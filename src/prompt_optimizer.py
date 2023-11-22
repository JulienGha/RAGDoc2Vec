from transformers import T5Tokenizer, T5ForConditionalGeneration


def prompt_opti(prompt):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

    # Step 1: Optimize the query
    input_text = f"Turn this query into a question. {prompt}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    optimized_query = model.generate(input_ids, max_length=256, num_beams=5, no_repeat_ngram_size=2, temperature=0.7)
    optimized_query_text = tokenizer.decode(optimized_query[0], skip_special_tokens=True)

    # Step 2: Generate an answer based on the optimized query
    input_text2 = f"Answer the following question step by step. {optimized_query_text}"
    input_ids2 = tokenizer(input_text2, return_tensors="pt").input_ids
    answer = model.generate(input_ids2, max_length=256, num_beams=5, no_repeat_ngram_size=2, temperature=0.7)
    answer_text = tokenizer.decode(answer[0], skip_special_tokens=True)

    print(f"Original Prompt: {prompt}")
    print(f"Optimized Query: {optimized_query_text}")
    print(f"Generated Answer: {answer_text}")
    return answer_text


# prompt_opti("define hallucination")
