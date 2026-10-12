from falsealpha.cli import main


def test_cli_flow(tmp_path, capsys):
    panel = str(tmp_path / "p.npz")
    assert main(["data", "planted", "--out", panel, "--days", "900", "--assets", "30"]) == 0
    assert main(["info", panel]) == 0
    assert main(["eval", panel, "rank(ts_mean(volume, 5))"]) == 0
    out = capsys.readouterr().out
    assert "validation" in out and "holdout not touched" in out
    run = str(tmp_path / "run")
    assert main(["mine", panel, run, "--population", "60", "--generations", "2", "--threads", "2"]) == 0
    assert (tmp_path / "run" / "report.html").exists()
    assert main(["unseal", run]) == 0
    assert main(["unseal", run]) == 2  # second look refused
    assert main(["eval", panel, "rank(nope)"]) == 2
    assert main(["ops"]) == 0


def test_panel_roundtrip(tmp_path):
    from falsealpha import synthetic
    from falsealpha.panel import Panel

    p = synthetic.noise(T=50, N=5)
    p.save(tmp_path / "x.npz")
    q = Panel.load(tmp_path / "x.npz")
    assert q.fingerprint() == p.fingerprint()
    assert q.head(20).shape == (20, 5)


def test_bad_field_names_are_refused():
    import numpy as np
    import pytest

    from falsealpha.panel import Panel

    with pytest.raises(ValueError, match="identifier"):
        Panel(np.datetime64("2020-01-01") + np.arange(3), ["a"],
              {"returns": np.zeros((3, 1)), "1d_ret": np.zeros((3, 1))}, np.ones((3, 1), bool))
