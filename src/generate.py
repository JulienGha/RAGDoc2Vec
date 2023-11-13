from transformers import pipeline

# Initialize the generator pipeline with a smaller model
generator = pipeline('text-generation', model='distilgpt2')


def generate_response(context):
    response = generator(context, max_length=250)
    return response[0]['generated_text']

