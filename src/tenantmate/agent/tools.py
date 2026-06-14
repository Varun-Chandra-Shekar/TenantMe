"""
Deterministic tools the agent can call.

Tools are pure Python — no LLM, no I/O beyond what's documented.
Each tool returns structured output the agent can reason over.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

from .rules import RULES_BY_JURISDICTION, RentIncreaseRules


@dataclass
class RentIncreaseResult:
    is_allowed: bool
    reasons: list[str]
    earliest_lawful_increase_date: Optional[date]
    rule_citations: list[str]


def check_rent_increase(
    *,
    jurisdiction: str,
    notice_date: date,
    increase_takes_effect_on: date,
    tenancy_start_date: date,
    last_increase_date: Optional[date] = None,
) -> RentIncreaseResult:
    """
    Determine whether a proposed rent increase complies with the jurisdiction's rules.

    Returns a structured verdict. Does not return the dollar amount —
    that's a separate concern (excessiveness, s 44, handled differently).
    """
    if jurisdiction not in RULES_BY_JURISDICTION:
        raise ValueError(f"No rules configured for jurisdiction: {jurisdiction}")
    rules: RentIncreaseRules = RULES_BY_JURISDICTION[jurisdiction]

    reasons: list[str] = []
    earliest_dates: list[date] = []

    # Rule 1 — minimum notice
    notice_days = (increase_takes_effect_on - notice_date).days
    if notice_days < rules.min_notice_days:
        reasons.append(
            f"Notice given {notice_days} days before the increase; "
            f"{rules.min_notice_days} days required."
        )
        from datetime import timedelta
        earliest_dates.append(notice_date + timedelta(days=rules.min_notice_days))

    # Rule 2 — 12 months since tenancy start
    months_since_start = _months_between(tenancy_start_date, increase_takes_effect_on)
    if months_since_start < rules.min_months_since_tenancy_start:
        reasons.append(
            f"Tenancy started {months_since_start} months ago; "
            f"increases not permitted within the first "
            f"{rules.min_months_since_tenancy_start} months."
        )
        earliest_dates.append(_add_months(tenancy_start_date, rules.min_months_since_tenancy_start))

    # Rule 3 — 12 months since last increase
    if last_increase_date is not None:
        months_since_last = _months_between(last_increase_date, increase_takes_effect_on)
        if months_since_last < rules.min_months_between_increases:
            reasons.append(
                f"Last increase was {months_since_last} months ago; "
                f"only one increase permitted per "
                f"{rules.min_months_between_increases}-month period."
            )
            earliest_dates.append(_add_months(last_increase_date, rules.min_months_between_increases))

    is_allowed = len(reasons) == 0
    earliest = max(earliest_dates) if earliest_dates else None

    return RentIncreaseResult(
        is_allowed=is_allowed,
        reasons=reasons,
        earliest_lawful_increase_date=earliest,
        rule_citations=list(rules.citations),
    )


# ----- internal helpers -----

def _months_between(d1: date, d2: date) -> int:
    """Whole months between d1 and d2 (d2 > d1)."""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) - (1 if d2.day < d1.day else 0)


def _add_months(d: date, months: int) -> date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    # Clamp the day in case the new month is shorter
    from calendar import monthrange
    day = min(d.day, monthrange(year, month)[1])
    return date(year, month, day)