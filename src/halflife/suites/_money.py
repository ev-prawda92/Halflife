from decimal import ROUND_HALF_UP, Decimal

TOLERANCE = 0.015


def money(value) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def same_money(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= TOLERANCE
    except (TypeError, ValueError):
        return False
