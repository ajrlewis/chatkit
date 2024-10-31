def create_message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def create_user_message(content: str) -> dict:
    return create_message(role="user", content=content)


def create_assistant_message(content: str) -> dict:
    return create_message(role="assistant", content=content)


def create_system_message(content: str) -> dict:
    return create_message(role="system", content=content)
