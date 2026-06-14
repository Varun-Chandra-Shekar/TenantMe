"""
Jurisdiction-keyed rules for rent-increase validation.

Rules live here so they're auditable and easy to update without touching
prompts or model code. Every rule cites the section it comes from.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RentIncreaseRules:
    """Rules governing when a residential rent increase is lawful."""
    jurisdiction: str
    min_notice_days: int
    min_months_since_tenancy_start: int
    min_months_between_increases: int
    citations: tuple[str, ...]


NSW_RENT_INCREASE = RentIncreaseRules(
    jurisdiction="NSW",
    min_notice_days=60,
    min_months_since_tenancy_start=12,
    min_months_between_increases=12,
    citations=(
        "NSW RTA 2010 s 41(1)(b)",   # 60-day notice
        "NSW RTA 2010 s 41(1A)(a)",  # 12 months since tenancy start
        "NSW RTA 2010 s 41(1A)(b)",  # 12 months between increases
    ),
)


RULES_BY_JURISDICTION = {
    "NSW": NSW_RENT_INCREASE,
    #VIC, QLD added in Week 5
}