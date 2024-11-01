from loguru import logger

from . import messages
from . import prompt_templates


# "max_tokens": 256,  # The maximum number of tokens to generate shared between the prompt and completion. The exact limit varies by model. (One token is roughly 4 characters for standard English text)
# "temperature": 0.0,  # How much to penalize new tokens based on their existing frequency in the text so far. Decreases the model's likelihood to repeat the same line verbatim.
# "top_p": 1,  # Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered.
# "frequency_penalty": 0,  # How much to penalize new tokens based on their existing frequency in the text so far. Decreases the model's likelihood to repeat the same line verbatim.
# "presence_penalty": 0,  # How much to penalize new tokens based on whether they appear in the text so far. Increases the model's likelihood to talk about new topics.
def completion(
    client,
    model: str = None,
    context_messages: list[dict] = [],
    max_tokens: int = 4096,
    temperature: float = 0.2,
    top_p: float = 1,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
) -> dict:
    """Returns the next message using a chat completion.

    Args:
        client: The inference client.
        context_messages: A list of messages.
        max_token: The maximum number of tokens allowed in the response 75 words approximately equals 100 tokens.
        temperature: Controls randomness of the generations between [0, 2]. Lower values ensure less random completions.
    """
    logger.debug(f"{model = } {context_messages = }")
    logger.debug(f"{temperature = } {max_tokens = }")
    # output = client.chat_completion(
    output = client.chat.completions.create(
        messages=context_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choices = output.choices
    choice = choices[0]
    message = choice.message
    logger.debug(f"{message = }")
    content = message.content.strip()
    logger.debug(f"{content = }")
    message = messages.create_assistant_message(content=content)
    return message


def call(client, template_name: str, **kwargs):
    content = prompt_templates.render_template(template_name, **kwargs)
    user_message = messages.create_user_message(content=content)
    context_messages = [user_message]
    assistant_message = completion(
        client, context_messages=context_messages, model=kwargs.get("model")
    )
    logger.debug(f"{assistant_message = }")
    return assistant_message
