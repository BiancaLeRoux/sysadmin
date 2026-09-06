from pathlib import Path

from avatarcam.models.registry import ASSETS_DIR, MODELS, _bundled_spec, licence_report


def test_all_specs_have_urls_or_bundle():
    for spec in MODELS.values():
        if spec.bundled:
            resolved = _bundled_spec(spec)
            assert (ASSETS_DIR / spec.filename).exists(), spec.filename
            assert len(resolved.sha256) == 64
        else:
            assert spec.url.startswith("https://github.com/")
            assert len(spec.sha256) == 64


def test_licence_report_lists_everything():
    text = licence_report()
    for spec in MODELS.values():
        assert spec.filename in text
