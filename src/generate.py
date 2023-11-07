from transformers import pipeline

# Initialize the generator pipeline with a smaller model
generator = pipeline('text-generation', model='distilgpt2')


def generate_response(context):
    response = generator(context, max_length=150)
    return response[0]['generated_text']


if __name__ == "__main__":
    # These are placeholders for context.
    # You'd replace this with actual data in practice.
    context = "France is a country in Europe that is well known for its history and culture. The capital city of France is"

    response = generate_response(context)
    print(f"Generated response: {response}")