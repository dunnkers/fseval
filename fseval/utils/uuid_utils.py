from typing import List, Optional
from shortuuid import ShortUUID

SHORT_UUID_ALPHABET: Optional[List[str]] = list("0123456789abcdefghijklmnopqrstuvwxyz")


def generate_shortuuid() -> str:
    """Generate a random shortuuid of 8 characters. Consists of alphanumeric
    characters."""
    run_gen = ShortUUID(alphabet=SHORT_UUID_ALPHABET)
    return run_gen.random(8)
