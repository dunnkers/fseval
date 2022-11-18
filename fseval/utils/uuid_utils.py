from shortuuid import ShortUUID

SHORT_UUID_ALPHABET = list("0123456789abcdefghijklmnopqrstuvwxyz")


def generate_shortuuid() -> str:
    """Generate a random shortuuid of 8 characters. Consists of alphanumeric
    characters."""
    run_gen = ShortUUID(alphabet=SHORT_UUID_ALPHABET)  # type: ignore
    return run_gen.random(8)
