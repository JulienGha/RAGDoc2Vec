from transformers import pipeline

# Initialize the generator pipeline with a smaller model
generator = pipeline('text-generation', model='distilgpt2')


def generate_response(context):
    response = generator(context, max_length=250)
    return response[0]['generated_text']


if __name__ == "__main__":
    # These are placeholders for context.
    # You'd replace this with actual data in practice.
    context = "hello"

    response = generate_response(context)
    print(f"Generated response: {response}")
