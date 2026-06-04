from rooftopsenti.stages.gba import _gba_tile_names


def test_tile_names_netherlands():
    # NL bbox spans lon 3.3..7.2, lat 50.7..53.6 -> 2 lon cells x 1 lat cell
    names = _gba_tile_names((3.3, 50.7, 7.2, 53.6))
    assert "e000_n55_e005_n50" in names
    assert "e005_n55_e010_n50" in names
    assert len(names) == 2


def test_tile_names_negative_coords():
    # straddling the equator and prime meridian
    names = _gba_tile_names((-2.0, -1.0, 1.0, 1.0))
    assert "w005_n00_e000_s05" in names
    assert "e000_n05_e005_n00" in names
    assert "w005_n05_e000_n00" in names
    assert "e000_n00_e005_s05" in names
    assert len(names) == 4


def test_tile_names_smoke_bbox_single_tile():
    names = _gba_tile_names((6.10, 51.30, 6.25, 51.45))
    assert names == ["e005_n55_e010_n50"]
