"""Redraw out/counters.svg with the profile's current views and followers.

    .venv/bin/python stats.py                              # fetch both numbers, rewrite only if changed
    .venv/bin/python stats.py --views 120 --followers 7    # draw given numbers (preview, no network)

Followers come from the GitHub API. Views come from a public visit counter: the README loads the
counter's image once per profile visit (that is the count) and this script reads the number without
adding one. When a number cannot be fetched, the last drawn one is kept, so a broken service never
blanks the profile. A GitHub Actions job (.github/workflows/counters.yml) runs this every 2 hours.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import urllib.request
from pathlib import Path

from ascii.counters import Counts, describe, picture

USER = "BALDEJ1"
VIEWS_PAGE_ID = "baldej1-profile-b748fc0c"   # the README's counter image uses the same id
VIEWS_URL = f"https://visitor-badge.laobi.icu/badge?page_id={VIEWS_PAGE_ID}&query_only=true"
FOLLOWERS_URL = f"https://api.github.com/users/{USER}"
OUT = Path("out/counters.svg")
LOOK = "big"
TIMEOUT_S = 20
USER_AGENT = "profile-counters"

log = logging.getLogger("stats")


def fetch(url: str, headers: dict[str, str] | None = None) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310 (fixed https URLs)
        return response.read().decode("utf-8")


def read_followers(token: str | None) -> int:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.loads(fetch(FOLLOWERS_URL, headers))
    followers = data.get("followers") if isinstance(data, dict) else None
    if not isinstance(followers, int) or isinstance(followers, bool) or followers < 0:
        raise ValueError(f"unexpected followers value {followers!r}")
    return followers


def parse_views(badge_svg: str) -> int:
    """The number in the counter's badge. The badge says 'Error' until the first visit is counted."""
    if "error" in badge_svg.lower():
        raise ValueError("the counter reports an error (no visit counted yet?)")
    numbers = re.findall(r">\s*(\d[\d,]*)\s*<", badge_svg)
    if not numbers:
        raise ValueError("no number in the counter badge")
    return int(numbers[-1].replace(",", ""))


def last_counts(svg_path: Path) -> Counts | None:
    """The numbers the current picture shows, read back from its description."""
    if not svg_path.exists():
        return None
    match = re.search(r"profile views (\d+), followers (\d+)", svg_path.read_text(encoding="utf-8"))
    return Counts(int(match.group(1)), int(match.group(2))) if match else None


def current_counts(previous: Counts) -> Counts:
    """Fetch both numbers; whichever fails keeps its previous value (and is logged)."""
    views, followers = previous.views, previous.followers
    try:
        views = parse_views(fetch(VIEWS_URL))
    except (OSError, ValueError) as error:
        log.warning("profile views: keeping %d, fetch failed: %s", views, error)
    try:
        followers = read_followers(os.environ.get("GITHUB_TOKEN"))
    except (OSError, ValueError) as error:
        log.warning("followers: keeping %d, fetch failed: %s", followers, error)
    return Counts(views, followers)


def write_if_changed(svg: str, path: Path) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == svg:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--views", type=int, help="draw this number instead of fetching it")
    parser.add_argument("--followers", type=int, help="draw this number instead of fetching it")
    parser.add_argument("--look", default=LOOK, help="big or line")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if (args.views is None) != (args.followers is None):
        parser.error("give both --views and --followers, or neither")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.views is not None:
        counts = Counts(args.views, args.followers)
    else:
        counts = current_counts(last_counts(args.out) or Counts(0, 0))
    changed = write_if_changed(picture(counts, args.look), args.out)
    log.info("%s: %s (%s)", args.out, describe(counts), "updated" if changed else "unchanged")


if __name__ == "__main__":
    main()
