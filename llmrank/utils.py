import importlib
import asyncio
import openai
from recbole.utils import get_model as recbole_get_model
import os
true=True
false=False

# price estimation: 2.7 Dollar / Hour running evaluate.py
import yaml
def load_openai_config(file_path="openai_api.yaml"):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_openai_config()

# Initialize the OpenAI client
client = openai.OpenAI(
    api_key=config.get("api_key"),
    base_url=config.get("api_base"),
)

# Example usage of the client
print("Client initialized with base_url:", config.get("api_base"))

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
       

def get_model(model_name):
    if importlib.util.find_spec(f'model.{model_name.lower()}', __name__):
        model_module = importlib.import_module(f'model.{model_name.lower()}', __name__)
        model_class = getattr(model_module, model_name)
        return model_class
    else:
        return recbole_get_model(model_name)


async def dispatch_openai_requests(
    messages_list,
    model: str,
    temperature: float
):
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async def create_completion(messages):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    
    # Create a list of tasks for async requests
    tasks = [create_completion(messages) for messages in messages_list]
    
    # Gather results
    return await asyncio.gather(*tasks)

def dispatch_single_openai_requests(
    message,
    model: str,
    temperature: float
):
    """
    Dispatches a single request to OpenAI API synchronously.

    Args:
        message: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.

    Returns:
        Response from OpenAI API.
    """
    response = client.chat.completions.create(
        model=model,
        messages=message,
        temperature=temperature,
    )

    return response


amazon_dataset2fullname = {
    'Beauty': 'All_Beauty',
    'Fashion': 'AMAZON_FASHION',
    'Appliances': 'Appliances',
    'Arts': 'Arts_Crafts_and_Sewing',
    'Automotive': 'Automotive',
    'Books': 'Books',
    'CDs': 'CDs_and_Vinyl',
    'Cell': 'Cell_Phones_and_Accessories',
    'Clothing': 'Clothing_Shoes_and_Jewelry',
    'Music': 'Digital_Music',
    'Electronics': 'Electronics',
    'Gift': 'Gift_Cards',
    'Food': 'Grocery_and_Gourmet_Food',
    'Home': 'Home_and_Kitchen',
    'Scientific': 'Industrial_and_Scientific',
    'Kindle': 'Kindle_Store',
    'Luxury': 'Luxury_Beauty',
    'Magazine': 'Magazine_Subscriptions',
    'Movies': 'Movies_and_TV',
    'Instruments': 'Musical_Instruments',
    'Office': 'Office_Products',
    'Garden': 'Patio_Lawn_and_Garden',
    'Pantry': 'Prime_Pantry',
    'Pet': 'Pet_Supplies',
    'Software': 'Software',
    'Sports': 'Sports_and_Outdoors',
    'Tools': 'Tools_and_Home_Improvement',
    'Toys': 'Toys_and_Games',
    'Games': 'Video_Games'
}

