"""Filter the frozen lead/lag signal with a causal 60-second TWAP margin.

The original conservative lead/lag candidate was frozen through window
1792066500 and then failed on later logs.  This experiment asks a distinct,
predeclared question: does requiring a Coinbase 60-second TWAP (built from
WebSocket *receive* timestamps) to agree with the shock remove false signals?

Filter selection sees only the original 219-window development set and must
be worst-seed profitable in every chronological quarter.  The windows after
the frozen boundary are evaluated once, only after selection.  Entries retain
the original full-depth $2 FOK simulation, current fees, $0.05 allowance, and
300--800 ms latency.  This file is research-only and cannot place orders.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import _research_chainlink_twap_proxy as twap_model
import _research_coinbase_leadlag as lead
import _research_coinbase_leadlag_conservative as conservative


ROOT = Path(__file__).resolve().parent
FROZEN_DEVELOPMENT_THROUGH = 1792066500


@dataclass(frozen=True)
class Filter:
    margin_bps: float
    ask_cap: float
    shock_bps: float

    @property
    def name(self) -> str:
        return (
            f"twap{self.margin_bps:g}bps_ask{self.ask_cap:.2f}_"
            f"shock{self.shock_bps:g}bps"
        )


def received_observations(directory: Path, through: int):
    """Deduplicated Coinbase prices keyed by when the logger received them."""
    observations = []
    for path in sorted(directory.glob("btc-updown-5m-*.jsonl")):
        try:
            start = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        if start > through:
            continue
        try:
            stream = path.open(encoding="utf-8")
        except OSError:
            continue
        with stream:
            for line in stream:
                if '"event":"snapshot"' not in line:
                    continue
                try:
                    raw = json.loads(line)
                    coinbase = raw.get("coinbase") or {}
                    received = float(coinbase["received_ts"])
                    price = float(coinbase["price"])
                except (ValueError, TypeError, KeyError, UnicodeDecodeError):
                    continue
                if (
                    math.isfinite(received)
                    and math.isfinite(price)
                    and received > 0
                    and price > 0
                ):
                    observations.append((received, price))
    return observations


def filter_decisions(windows, decisions, series, opening, config: Filter):
    selected = []
    for window, decision in zip(windows, decisions):
        if decision is None or decision.ask > config.ask_cap + 1e-12:
            selected.append(None)
            continue
        if decision.shock_bps + 1e-12 < config.shock_bps:
            selected.append(None)
            continue
        base = opening.get(window.slug)
        current = series.twap(window.start + decision.t)
        if base is None or current is None or base <= 0:
            selected.append(None)
            continue
        raw_margin = (current / base - 1.0) * 10_000.0
        oriented = raw_margin if decision.side == "Up" else -raw_margin
        selected.append(decision if oriented + 1e-12 >= config.margin_bps else None)
    return selected


def aggregate(windows, decisions, seeds):
    return lead.aggregate(windows, decisions, conservative.RULE, seeds)


def direction_accuracy(windows, series, opening):
    matched = total = 0
    for window in windows:
        first = opening.get(window.slug)
        final = series.twap(window.start + 300.0)
        if first is None or final is None:
            continue
        predicted = "Up" if final >= first else "Down"
        matched += int(predicted == window.winner)
        total += 1
    return matched, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path,
        default=ROOT / "price_logs" / "polymarket",
    )
    parser.add_argument("--through", type=int, default=1792081200)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()

    windows = lead.load_windows(args.directory, args.through)
    development = [
        window for window in windows
        if window.start <= FROZEN_DEVELOPMENT_THROUGH
    ]
    prospective = [
        window for window in windows
        if window.start > FROZEN_DEVELOPMENT_THROUGH
    ]
    if len(development) < 100 or not prospective:
        raise SystemExit("need both the frozen development set and later resolved logs")

    series = twap_model.PiecewisePrice(
        received_observations(args.directory, args.through)
    )
    opening = {
        window.slug: value
        for window in windows
        if (value := series.twap(float(window.start))) is not None
    }
    base_decisions = [
        conservative.confirmed_decision_for(window) for window in windows
    ]
    decision_by_slug = dict(zip((window.slug for window in windows), base_decisions))
    development_decisions = [decision_by_slug[window.slug] for window in development]
    prospective_decisions = [decision_by_slug[window.slug] for window in prospective]
    seeds = range(args.seeds)

    matched, comparable = direction_accuracy(development, series, opening)
    print(
        f"TWAP-FILTERED LEAD/LAG; development={len(development)} through="
        f"{FROZEN_DEVELOPMENT_THROUGH}; prospective={len(prospective)} through="
        f"{args.through}"
    )
    print(
        f"receive-time Coinbase 60s TWAP final-direction match="
        f"{matched}/{comparable} ({100*matched/max(1, comparable):.1f}%)"
    )
    print(
        "$2; full depth; confirmation plus following-cycle clock; "
        "300-800ms; allowance=$0.05; current fees"
    )

    configs = [
        Filter(margin, cap, shock)
        for margin in (0.0, 1.0, 2.0, 4.0)
        for cap in (0.60, 0.70, 0.85)
        for shock in (2.0, 3.0, 4.0)
    ]
    bounds = [
        0,
        len(development) // 4,
        len(development) // 2,
        3 * len(development) // 4,
        len(development),
    ]
    passing = []
    for config in configs:
        decisions = filter_decisions(
            development, development_decisions, series, opening, config
        )
        quarters = [
            aggregate(development[a:b], decisions[a:b], seeds)
            for a, b in zip(bounds, bounds[1:])
        ]
        total = aggregate(development, decisions, seeds)
        passed = (
            total["trades"] >= 8
            and all(row["trades"] >= 1 for row in quarters)
            and all(row["pnl_min"] > 0 for row in quarters)
        )
        if passed:
            score = min(row["pnl_min"] for row in quarters)
            passing.append((score, total["pnl_min"], config, decisions, quarters, total))

    print(f"development_grid={len(configs)} passed_every_quarter={len(passing)}")
    if not passing:
        print("REJECT: no filter passed development; prospective set not opened.")
        return
    passing.sort(key=lambda row: (row[0], row[1]), reverse=True)
    score, _, selected, decisions, quarters, total = passing[0]
    print(f"selected_before_prospective={selected.name} min_quarter=${score:+.2f}")
    for index, result in enumerate(quarters, 1):
        print(f"  dev-Q{index} {lead.fmt(result)}")
    print(f"  dev-all {lead.fmt(total)}")

    future_filtered = filter_decisions(
        prospective, prospective_decisions, series, opening, selected
    )
    future = aggregate(prospective, future_filtered, seeds)
    print(
        f"PROSPECTIVE {prospective[0].start}-{prospective[-1].start} "
        f"{lead.fmt(future)}"
    )
    robust = future["trades"] >= 4 and future["pnl_min"] > 0
    print("PASS" if robust else "REJECT: failed prospective robustness gate")


if __name__ == "__main__":
    main()
