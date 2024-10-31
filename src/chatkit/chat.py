from loguru import logger

from . import messages
from . import prompt_templates


def completion(
    client,
    context_messages: list[dict],
    max_tokens: int = 7000,
    temperature: float = 0.2,
) -> dict:
    """Returns the next message using a chat completion.

    Args:
        client: The inference client.
        messages: A list of messages.
        max_token: The maximum number of tokens allowed in the response 75 words approximately equals 100 tokens.
        temperature: Controls randomness of the generations between [0, 2]. Lower values ensure less random completions.
    """
    logger.debug(f"{temperature = } {max_tokens = }")
    logger.debug(f"{context_messages = }")
    output = client.chat_completion(
        context_messages, temperature=temperature, max_tokens=max_tokens
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
    assistant_message = completion(client=client, context_messages=context_messages)
    logger.debug(f"{assistant_message = }")
    return assistant_message
