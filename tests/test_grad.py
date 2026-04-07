import pytest

from myautograd.value import Value


@pytest.fixture
def sample_numbers() -> tuple[Value, Value]:
    a = Value(3.3, label='a')
    b = Value(2.25, label='b')
    return (a, b)

def test_add(sample_numbers) -> None:
    a, b = sample_numbers
    c = a + b
    assert isinstance(c, Value)
    assert c.data == pytest.approx(5.55)

def test_mul(sample_numbers) -> None:
    a, b = sample_numbers
    c = a * b
    assert isinstance(c, Value)
    assert c.data == pytest.approx(7.425)


