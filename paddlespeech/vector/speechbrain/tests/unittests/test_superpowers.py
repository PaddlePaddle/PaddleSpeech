import sys
import pytest


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="shell tools not necessarily available on Windows",
)
def test_run_shell():
    from speechbrain.utils.superpowers import run_shell

    out, err, code = run_shell("echo -n hello")
    assert out.decode() == "hello"
    assert err.decode() == ""
    assert code == 0

    with pytest.raises(OSError):
        run_shell("false")

    # This last run is just to check that a bytes
    # sequence that is returned in an incompatible encoding (not UTF-8)
    # does not cause an error .
    output, _, _ = run_shell("echo -n pöö | iconv -t LATIN1")
    assert output.decode("latin1") == "pöö"
