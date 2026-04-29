from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from typing import List, Optional
from zoneinfo import ZoneInfo

from .http import FetchError, get_text


FED_FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FED_SPEECHES_RSS_URL = "https://www.federalreserve.gov/feeds/speeches.xml"
BEA_RELEASE_SCHEDULE_URL = "https://www.bea.gov/news/schedule/full"
BLS_RELEASE_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"

US_EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc


@dataclass
class EconomicEvent:
    source: str
    title: str
    event_type: str
    impact: str
    date: str
    time_et: Optional[str]
    datetime_utc: Optional[str]
    status: str
    url: str
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalendarSourceStatus:
    source: str
    status: str
    url: str
    item_count: int
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _event_dt(date_value: datetime.date, event_time: Optional[time]) -> Optional[str]:
    if event_time is None:
        return None
    dt = datetime.combine(date_value, event_time, US_EASTERN)
    return dt.astimezone(UTC).isoformat()


def _parse_month_day(year: int, month_name: str, date_text: str) -> Optional[datetime.date]:
    month_part = month_name.split("/")[-1].strip()
    try:
        month = datetime.strptime(month_part, "%B").month
    except ValueError:
        month = datetime.strptime(month_part, "%b").month
    cleaned = date_text.replace("*", "").strip()
    day_text = cleaned.split("-")[-1].strip()
    try:
        return datetime(year, month, int(day_text)).date()
    except ValueError:
        return None


def _classify_bea_release(title: str) -> tuple[str, str, time]:
    lower = title.lower()
    if "personal income and outlays" in lower:
        return "PCE_INFLATION", "HIGH", time(8, 30)
    if "gross domestic product" in lower or re.search(r"\bgdp\b", lower):
        return "GDP_GROWTH", "MEDIUM", time(8, 30)
    if "international trade" in lower:
        return "TRADE_BALANCE", "LOW", time(8, 30)
    if "corporate profits" in lower:
        return "PROFITS", "LOW", time(8, 30)
    return "BEA_RELEASE", "LOW", time(8, 30)


def _parse_bea_time(row: str, fallback: time) -> time:
    time_match = re.search(r"<small class=\"text-muted\">([^<]+)</small>", row)
    if not time_match:
        return fallback
    text = _clean_text(time_match.group(1)).upper()
    try:
        return datetime.strptime(text, "%I:%M %p").time()
    except ValueError:
        return fallback


def fetch_bea_release_schedule(timeout: int = 15) -> tuple[List[EconomicEvent], CalendarSourceStatus]:
    try:
        body = get_text(BEA_RELEASE_SCHEDULE_URL, timeout=timeout, ttl_seconds=6 * 3600)
    except FetchError as exc:
        return [], CalendarSourceStatus("BEA", "ERROR", BEA_RELEASE_SCHEDULE_URL, 0, str(exc))

    rows = re.findall(r"<tr class=\"scheduled-releases-type-press\">(.*?)</tr>", body, flags=re.S)
    events: List[EconomicEvent] = []
    current_year_match = re.search(r"<th[^>]*>\s*Year\s+(\d{4})\s*</th>", body)
    current_year = int(current_year_match.group(1)) if current_year_match else datetime.now(US_EASTERN).year
    for row in rows:
        date_match = re.search(r"<div class=\"release-date\">([^<]+)</div>", row)
        title_match = re.search(r"<td class=\"release-title[^>]*>(.*?)</td>", row, flags=re.S)
        links = [link for link in re.findall(r'href="([^"]*)"', row) if link]
        if not date_match or not title_match:
            continue
        date_text = _clean_text(date_match.group(1))
        title = _clean_text(title_match.group(1))
        parts = date_text.split()
        if len(parts) != 2:
            continue
        event_date = _parse_month_day(current_year, parts[0], parts[1])
        if event_date is None:
            continue
        event_type, impact, fallback_time = _classify_bea_release(title)
        release_time = _parse_bea_time(row, fallback_time)
        path = next((link for link in links if "/news/" in link), links[0] if links else "")
        url = path if path.startswith("http") else f"https://www.bea.gov{path}"
        events.append(
            EconomicEvent(
                source="BEA",
                title=title,
                event_type=event_type,
                impact=impact,
                date=event_date.isoformat(),
                time_et=release_time.strftime("%H:%M"),
                datetime_utc=_event_dt(event_date, release_time),
                status="SCHEDULED",
                url=url,
                notes="Time parsed from BEA schedule when available; otherwise falls back to the standard 08:30 ET major-release assumption.",
            )
        )
    return events, CalendarSourceStatus("BEA", "OK", BEA_RELEASE_SCHEDULE_URL, len(events))


def fetch_fomc_schedule(timeout: int = 15) -> tuple[List[EconomicEvent], CalendarSourceStatus]:
    try:
        body = get_text(FED_FOMC_URL, timeout=timeout, ttl_seconds=6 * 3600)
    except FetchError as exc:
        return [], CalendarSourceStatus("FED_FOMC", "ERROR", FED_FOMC_URL, 0, str(exc))

    year_blocks = re.findall(
        r">(\d{4}) FOMC Meetings</a></h4></div>(.*?)(?=<div class=\"panel panel-default\"><div class=\"panel-heading\"><h4><a id=|\Z)",
        body,
        flags=re.S,
    )
    events: List[EconomicEvent] = []
    for year_text, block in year_blocks:
        year = int(year_text)
        meetings = re.findall(
            r"fomc-meeting__month[^>]*><strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)</div>",
            block,
            flags=re.S,
        )
        for month_text, date_text in meetings:
            event_date = _parse_month_day(year, month_text, date_text)
            if event_date is None:
                continue
            has_sep = "*" in date_text
            events.append(
                EconomicEvent(
                    source="FED",
                    title=f"FOMC policy decision{'; SEP' if has_sep else ''}",
                    event_type="FOMC_DECISION",
                    impact="HIGH",
                    date=event_date.isoformat(),
                    time_et="14:00",
                    datetime_utc=_event_dt(event_date, time(14, 0)),
                    status="SCHEDULED",
                    url=FED_FOMC_URL,
                    notes="FOMC decision time modeled as 14:00 ET; '*' meetings include Summary of Economic Projections.",
                )
            )
    return events, CalendarSourceStatus("FED_FOMC", "OK", FED_FOMC_URL, len(events))


def fetch_fed_recent_speeches(timeout: int = 15, limit: int = 8) -> tuple[List[EconomicEvent], CalendarSourceStatus]:
    try:
        body = get_text(FED_SPEECHES_RSS_URL, timeout=timeout, ttl_seconds=3600)
    except FetchError as exc:
        return [], CalendarSourceStatus("FED_SPEECHES", "ERROR", FED_SPEECHES_RSS_URL, 0, str(exc))

    events: List[EconomicEvent] = []
    try:
        root = ET.fromstring(body.lstrip("\ufeff").lstrip("ï»¿"))
        for item in root.findall("./channel/item")[:limit]:
            title = item.findtext("title") or "Federal Reserve speech"
            link = item.findtext("link") or FED_SPEECHES_RSS_URL
            pub_date = item.findtext("pubDate")
            dt_utc = None
            date = ""
            time_et = None
            if pub_date:
                try:
                    dt_utc = datetime.strptime(pub_date.strip(), "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=UTC)
                    date = dt_utc.astimezone(US_EASTERN).date().isoformat()
                    time_et = dt_utc.astimezone(US_EASTERN).strftime("%H:%M")
                except ValueError:
                    pass
            events.append(
                EconomicEvent(
                    source="FED",
                    title=title.strip(),
                    event_type="FED_SPEECH_RECENT",
                    impact="LOW",
                    date=date,
                    time_et=time_et,
                    datetime_utc=dt_utc.isoformat() if dt_utc else None,
                    status="PUBLISHED",
                    url=link.strip(),
                    notes="RSS contains recent published speeches, not a complete future-speaking calendar.",
                )
            )
    except ET.ParseError as exc:
        return [], CalendarSourceStatus("FED_SPEECHES", "ERROR", FED_SPEECHES_RSS_URL, 0, str(exc))
    return events, CalendarSourceStatus("FED_SPEECHES", "OK", FED_SPEECHES_RSS_URL, len(events))


def probe_bls_calendar(timeout: int = 15) -> CalendarSourceStatus:
    try:
        body = get_text(BLS_RELEASE_ICS_URL, timeout=timeout, ttl_seconds=6 * 3600)
    except FetchError as exc:
        return CalendarSourceStatus("BLS", "ERROR", BLS_RELEASE_ICS_URL, 0, str(exc))
    if "Access Denied" in body[:500]:
        return CalendarSourceStatus("BLS", "UNAVAILABLE", BLS_RELEASE_ICS_URL, 0, "BLS calendar returned Access Denied.")
    return CalendarSourceStatus("BLS", "AVAILABLE_UNPARSED", BLS_RELEASE_ICS_URL, body.count("BEGIN:VEVENT"))


def build_economic_calendar(window_days: int = 21) -> dict:
    now_et = datetime.now(US_EASTERN)
    start = now_et.date()
    end = start + timedelta(days=window_days)

    bea_events, bea_status = fetch_bea_release_schedule()
    fomc_events, fomc_status = fetch_fomc_schedule()
    fed_speeches, speech_status = fetch_fed_recent_speeches()
    bls_status = probe_bls_calendar()

    scheduled = []
    for event in bea_events + fomc_events:
        event_date = datetime.fromisoformat(event.date).date()
        if start <= event_date <= end:
            scheduled.append(event)
    scheduled.sort(key=lambda item: (item.date, item.time_et or "99:99", item.title))

    recent_speeches = sorted(
        fed_speeches,
        key=lambda item: item.datetime_utc or "",
        reverse=True,
    )[:5]

    high_impact = [item for item in scheduled if item.impact == "HIGH"]
    today_high = [item for item in high_impact if item.date == start.isoformat()]
    next_high = high_impact[0] if high_impact else None
    gate = {
        "status": "EVENT_RISK" if today_high else "NORMAL",
        "rule": "For HIGH impact events, wait 15-30 minutes after release/decision before allowing new trades.",
        "today_high_impact_count": len(today_high),
        "next_high_impact_event": next_high.to_dict() if next_high else None,
    }

    return {
        "generated_at_et": now_et.isoformat(),
        "window_days": window_days,
        "source_status": [
            bea_status.to_dict(),
            fomc_status.to_dict(),
            speech_status.to_dict(),
            bls_status.to_dict(),
        ],
        "scheduled_events": [item.to_dict() for item in scheduled],
        "recent_fed_speeches": [item.to_dict() for item in recent_speeches],
        "event_gate": gate,
    }
