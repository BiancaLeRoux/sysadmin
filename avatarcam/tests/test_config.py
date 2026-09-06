import pytest

from avatarcam.config import Config, dump_toml, validate


def test_defaults_load_and_validate():
    cfg = Config.load()
    assert cfg.face.enabled is True
    assert 0.0 <= cfg.face.swap_strength <= 1.0
    assert "cpu" in cfg.runtime.providers


def test_roundtrip_toml(tmp_path):
    cfg = Config.load()
    cfg.body.waist = -0.25
    cfg.output.disclosure = True
    p = cfg.save(tmp_path / "out.toml")
    again = Config.load(p)
    assert again.body.waist == -0.25
    assert again.output.disclosure is True
    assert again.face.mask_padding == cfg.face.mask_padding


def test_validate_rejects_out_of_range():
    data = Config.load().as_dict()
    data["face"]["swap_strength"] = 1.5
    with pytest.raises(ValueError):
        validate(data)
    data = Config.load().as_dict()
    data["runtime"]["providers"] = ["gpu"]
    with pytest.raises(ValueError):
        validate(data)


def test_keep_mouth_forces_region_mask():
    data = Config.load().as_dict()
    data["face"]["keep_mouth"] = True
    data["face"]["mask"] = "box"
    validate(data)
    assert data["face"]["mask"] == "region"


def test_dump_escapes_strings():
    text = dump_toml({"a": {"s": 'he said "hi"', "b": True, "l": [1, 2.5]}})
    assert 's = "he said \\"hi\\""' in text
    assert "b = true" in text
    assert "l = [1, 2.5]" in text
