import pytest
from dss import rule_pm_kisan, rule_jal_jeevan_mission, rule_mgnrega, rule_dajgua


def test_pm_kisan_positive():
    a = {'agriculture': True, 'land_size': 1.0, 'income': 20000}
    assert rule_pm_kisan(a) is True


def test_pm_kisan_negative_large_land():
    a = {'agriculture': True, 'land_size': 3.0, 'income': 20000}
    assert rule_pm_kisan(a) is False


def test_jjm_none():
    a = {'water_access': 'none'}
    assert rule_jal_jeevan_mission(a) is True


def test_mgnrega_labourer():
    a = {'labourer': True}
    assert rule_mgnrega(a) is True


def test_dajgua_tribe():
    a = {'tribe': True}
    assert rule_dajgua(a) is True
