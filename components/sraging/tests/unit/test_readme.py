"""tools/verify_readme.py recomputes marked README numbers and fails on a changed digit (T058; II)."""
import os

import pytest

import verify_readme as V
from test_result_json import make

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def stored(tmp_path):
    p = tmp_path / "res.json"
    make().to_json(p)
    return p


def readme(stored, loglik_text, block=None):
    rel = os.path.relpath(stored, REPO)
    text = (f"The fit's log-likelihood is <!-- verify:{rel}:loglik:.6f -->{loglik_text}<!-- /verify --> nats, "
            f"and eta's upper bound is <!-- verify:{rel}:claims.eta.upper:.4g -->0.0035<!-- /verify -->.\n\n")
    if block is not None:
        text += f"<!-- verify-output:{rel} -->\n```text\n{block}\n```\n"
    return text


def test_correct_numbers_pass(stored):
    assert V.verify(readme(stored, "-1234.567890")) == []


def test_one_changed_digit_fails(stored):
    problems = V.verify(readme(stored, "-1234.567891"))
    assert len(problems) == 1 and "loglik" in problems[0]


def test_output_block_must_equal_the_rendering(stored):
    good = V.render(os.path.relpath(stored, REPO))
    assert V.verify(readme(stored, "-1234.567890", good)) == []
    bad = good.replace("RESULT", "RESULTS", 1)
    assert any("example output differs" in p for p in V.verify(readme(stored, "-1234.567890", bad)))


def test_unmarked_readme_fails():
    assert V.verify("A README with a number 3.14 but no markers.") != []


def test_the_real_readme_verifies():
    """Once T057 has written README.md with markers, it must verify against the stored results."""
    text = open(os.path.join(REPO, "README.md")).read()
    if "<!-- verify" not in text:
        pytest.skip("README.md has no verify markers yet (task T057: the worked example is pending)")
    assert V.verify(text) == []
