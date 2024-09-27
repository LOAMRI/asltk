import pytest

from asltk.reconstruction import CBFMapping


def test_create_cbfmapping_object_success():
    cbf = CBFMapping()
    assert isinstance(cbf, CBFMapping)


def test_cbfmapping_object_has_default_asldata_values():
    cbf = CBFMapping()
    assert cbf.get_ld() == []
    assert cbf.get_pld() == []
    assert cbf.get_te() == None
    assert cbf.get_dw() == None
