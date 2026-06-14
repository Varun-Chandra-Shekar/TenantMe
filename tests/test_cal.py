"""Tests for the rent-increase calculator."""

from datetime import date

from tenantmate.agent.tools import check_rent_increase


def test_allowed_with_proper_notice_and_intervals():
    """Standard valid case: 60+ days notice, well past 12-month intervals."""
    r = check_rent_increase(
        jurisdiction="NSW",
        notice_date=date(2026, 1, 1),
        increase_takes_effect_on=date(2026, 4, 1),    # 90 days notice
        tenancy_start_date=date(2024, 1, 1),          # 27 months ago
        last_increase_date=date(2024, 12, 1),         # 16 months ago
    )
    assert r.is_allowed is True
    assert r.reasons == []


def test_blocked_by_insufficient_notice():
    """30 days notice fails the 60-day minimum."""
    r = check_rent_increase(
        jurisdiction="NSW",
        notice_date=date(2026, 6, 1),
        increase_takes_effect_on=date(2026, 7, 1),    # 30 days notice
        tenancy_start_date=date(2024, 1, 1),
        last_increase_date=date(2024, 12, 1),
    )
    assert r.is_allowed is False
    assert any("60 days" in reason for reason in r.reasons)


def test_blocked_within_12_months_of_tenancy_start():
    """No increases in the first 12 months of a tenancy."""
    r = check_rent_increase(
        jurisdiction="NSW",
        notice_date=date(2026, 1, 1),
        increase_takes_effect_on=date(2026, 6, 1),
        tenancy_start_date=date(2026, 1, 1),          # only 5 months in
    )
    assert r.is_allowed is False
    assert any("12 months" in reason for reason in r.reasons)


def test_blocked_within_12_months_of_last_increase():
    """No more than one increase per 12-month period."""
    r = check_rent_increase(
        jurisdiction="NSW",
        notice_date=date(2026, 1, 1),
        increase_takes_effect_on=date(2026, 6, 1),
        tenancy_start_date=date(2023, 1, 1),
        last_increase_date=date(2025, 9, 1),          # only 9 months ago
    )
    assert r.is_allowed is False


def test_earliest_lawful_date_is_the_furthest_constraint():
    """When multiple rules block, earliest_lawful must satisfy ALL of them."""
    r = check_rent_increase(
        jurisdiction="NSW",
        notice_date=date(2026, 1, 1),
        increase_takes_effect_on=date(2026, 2, 1),    # short notice
        tenancy_start_date=date(2025, 6, 1),          # ~8 months in
        last_increase_date=None,
    )
    assert r.is_allowed is False
    # earliest must be at or beyond 2026-06-01 (12 months after tenancy start)
    assert r.earliest_lawful_increase_date >= date(2026, 6, 1)


def test_citations_always_present_in_output():
    """Every result, allowed or not, carries the section citations."""
    r = check_rent_increase(
        jurisdiction="NSW",
        notice_date=date(2026, 1, 1),
        increase_takes_effect_on=date(2026, 4, 1),
        tenancy_start_date=date(2024, 1, 1),
        last_increase_date=date(2024, 12, 1),
    )
    assert any("s 41" in c for c in r.rule_citations)


def test_unknown_jurisdiction_raises():
    """Defensive: fail loudly on a jurisdiction we don't yet support."""
    import pytest
    with pytest.raises(ValueError):
        check_rent_increase(
            jurisdiction="ACT",
            notice_date=date(2026, 1, 1),
            increase_takes_effect_on=date(2026, 4, 1),
            tenancy_start_date=date(2024, 1, 1),
        )