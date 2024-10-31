import os
import sys

from huggingface_hub import auth_check, InferenceClient
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from loguru import logger


def get_token():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise KeyError("HF_TOKEN environmental variable not set")
    return token


def check_token_access(repo_id: str, token: str):
    try:
        auth_check(repo_id, token=token)
    except GatedRepoError:
        raise GatedRepoError(f"you don't have permission to access {repo_id = }")
    except RepositoryNotFoundError:
        raise GatedRepoError(
            f"the repository was not found or you do not have access {repo_id = }"
        )


def get_client(repo_id: str) -> InferenceClient:
    token = get_token()  # Get the token
    check_token_access(repo_id, token=token)  # Check the token can access the repo
    client = InferenceClient(repo_id, token=token)  # Create and return the client
    return client


def get_models():
    return [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ]
