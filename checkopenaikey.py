"""Check whether an OpenAI API key can connect and authenticate."""

import os
import sys

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI


def main() -> int:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        print("ERROR: OPENAI_API_KEY was not found.")
        print("Add OPENAI_API_KEY=your_key_here to a .env file, then try again.")
        return 1

    try:
        client = OpenAI(api_key=api_key, timeout=20.0, max_retries=1)
        models = client.models.list()
    except AuthenticationError:
        print("ERROR: The API key was rejected. Check or replace OPENAI_API_KEY.")
        return 2
    except APIConnectionError:
        print("ERROR: Could not connect to OpenAI. Check your internet connection.")
        return 3
    except APIStatusError as error:
        print(f"ERROR: OpenAI returned HTTP {error.status_code}: {error.message}")
        return 4

    print("SUCCESS: Your OpenAI API key and connection are working.")
    print(f"The account can access {len(models.data)} model(s) on this page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
