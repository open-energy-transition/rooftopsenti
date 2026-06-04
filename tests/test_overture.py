from rooftopsenti.overture import parse_osm_record_id, theme_path


def test_parse_osm_record_id():
    assert parse_osm_record_id("w123456789@5") == ("way", 123456789)
    assert parse_osm_record_id("n7520513886@2") == ("node", 7520513886)
    assert parse_osm_record_id("r42@1") == ("relation", 42)
    assert parse_osm_record_id(None) == (None, None)
    assert parse_osm_record_id("garbage") == (None, None)


def test_theme_path():
    assert theme_path("2026-05-20.0", "base", "infrastructure") == (
        "s3://overturemaps-us-west-2/release/2026-05-20.0/theme=base/type=infrastructure/*"
    )
