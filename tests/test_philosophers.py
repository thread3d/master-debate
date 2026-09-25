import pytest

from philosophers import philosophers

REQUIRED_KEYS = {"system", "temperament"}


def test_has_twenty_philosophers():
    assert len(philosophers) == 20


@pytest.mark.parametrize("name", sorted(philosophers))
def test_every_entry_is_well_formed(name):
    data = philosophers[name]
    assert name.strip()
    assert REQUIRED_KEYS <= set(data), f"{name} is missing keys"
    for key in REQUIRED_KEYS:
        assert isinstance(data[key], str)
        assert data[key].strip(), f"{name}.{key} is empty"


def test_persona_prompts_name_their_philosopher():
    for name, data in philosophers.items():
        assert name in data["system"]
