from fseval.utils.uuid_utils import SHORT_UUID_ALPHABET, generate_shortuuid


def test_id_generation():
    some_id: str = generate_shortuuid()

    # is string of length 8
    assert isinstance(some_id, str)
    assert len(some_id) == 8

    # characters must be in chosen alphabet
    for char in some_id:
        assert char in SHORT_UUID_ALPHABET
