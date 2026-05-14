from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request
import argparse
import json
import os
import sys
import time


NVD_API_CVES_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
GITHUB_ADVISORIES_URL = "https://api.github.com/advisories"
GITHUB_API_VERSION = "2026-03-10"
MANDATORY_NVD_DELAY_SECONDS = 6.0
DEFAULT_RECENT_WINDOW_HOURS = 24


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download recent NVD and GitHub advisory data into data/raw.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Directory to store normalized raw records")
    parser.add_argument("--nvd-url", default=NVD_API_CVES_URL, help="Override the NVD CVE API 2.0 endpoint")
    parser.add_argument("--nvd-results-per-page", type=int, default=2000, help="NVD API results per page")
    parser.add_argument("--recent-hours", type=int, default=DEFAULT_RECENT_WINDOW_HOURS, help="Only fetch CVEs modified in the last N hours")
    parser.add_argument("--last-mod-start-date", help="Explicit NVD start datetime in UTC, RFC3339 format")
    parser.add_argument("--last-mod-end-date", help="Explicit NVD end datetime in UTC, RFC3339 format")
    parser.add_argument("--github-url", default=GITHUB_ADVISORIES_URL, help="Override the GitHub Security Advisories API URL")
    parser.add_argument("--github-per-page", type=int, default=100, help="Advisories per GitHub API page")
    parser.add_argument("--github-max-pages", type=int, default=1, help="Number of GitHub pages to fetch")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.now(tz=timezone.utc).isoformat()
    nvd_api_key = os.environ.get("NVD_API_KEY", "").strip() or None
    last_mod_start_date, last_mod_end_date = resolve_recent_window(
        last_mod_start_date=args.last_mod_start_date,
        last_mod_end_date=args.last_mod_end_date,
        recent_hours=args.recent_hours,
    )

    nvd_records = fetch_nvd_recent(
        args.nvd_url,
        fetched_at=fetched_at,
        results_per_page=args.nvd_results_per_page,
        last_mod_start_date=last_mod_start_date,
        last_mod_end_date=last_mod_end_date,
        nvd_api_key=nvd_api_key,
    )
    github_records = fetch_github_advisories(
        args.github_url,
        fetched_at=fetched_at,
        per_page=args.github_per_page,
        max_pages=args.github_max_pages,
        token=os.environ.get("GITHUB_TOKEN", "").strip() or None,
    )

    write_json(output_dir / "nvd_cve_recent.json", nvd_records)
    write_json(output_dir / "github_security_advisories.json", github_records)
    print(
        {
            "nvd_records": len(nvd_records),
            "github_records": len(github_records),
            "output_dir": str(output_dir),
            "last_mod_start_date": last_mod_start_date,
            "last_mod_end_date": last_mod_end_date,
            "nvd_api_key_configured": bool(nvd_api_key),
        }
    )
    return 0


def resolve_recent_window(*, last_mod_start_date: str | None, last_mod_end_date: str | None, recent_hours: int) -> tuple[str, str]:
    if bool(last_mod_start_date) != bool(last_mod_end_date):
        raise ValueError("Provide both --last-mod-start-date and --last-mod-end-date, or neither")
    if last_mod_start_date and last_mod_end_date:
        return last_mod_start_date, last_mod_end_date

    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(hours=max(1, recent_hours))
    return _format_nvd_datetime(start_dt), _format_nvd_datetime(end_dt)


def fetch_nvd_recent(
    url: str,
    *,
    fetched_at: str,
    results_per_page: int,
    last_mod_start_date: str,
    last_mod_end_date: str,
    nvd_api_key: str | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    start_index = 0
    total_results: int | None = None
    request_count = 0

    while total_results is None or start_index < total_results:
        if request_count > 0 and not nvd_api_key:
            time.sleep(MANDATORY_NVD_DELAY_SECONDS)

        query = parse.urlencode(
            {
                "startIndex": start_index,
                "resultsPerPage": max(1, results_per_page),
                "lastModStartDate": last_mod_start_date,
                "lastModEndDate": last_mod_end_date,
            }
        )
        page_url = f"{url}?{query}"
        payload = _read_json(page_url, token=nvd_api_key, service="nvd")
        if not isinstance(payload, dict):
            raise RuntimeError("NVD API response was not a JSON object")

        vulnerabilities = payload.get("vulnerabilities", [])
        if not isinstance(vulnerabilities, list):
            raise RuntimeError("NVD API response did not include a vulnerabilities array")

        if total_results is None:
            total_results = int(payload.get("totalResults", 0))

        for item in vulnerabilities:
            if not isinstance(item, dict):
                continue
            cve = item.get("cve", {}) if isinstance(item.get("cve"), dict) else {}
            cve_id = str(cve.get("id") or f"nvd-recent-{len(records) + 1}")
            records.append(
                {
                    "source_id": cve_id,
                    "source_type": "report",
                    "title": _nvd_title(cve, cve_id),
                    "raw_text": _nvd_description(cve),
                    "url": _nvd_url(cve_id),
                    "cve_id": cve_id,
                    "ghsa_id": None,
                    "published_at": cve.get("published") if isinstance(cve.get("published"), str) else None,
                    "severity": _nvd_severity(cve),
                    "affected_components": _nvd_affected_components(cve),
                    "metadata": {
                        "source": "nvd_api_2_0",
                        "fetched_at": fetched_at,
                        "source_url": page_url,
                        "last_mod_start_date": last_mod_start_date,
                        "last_mod_end_date": last_mod_end_date,
                        "nvd_api_key_configured": bool(nvd_api_key),
                        "nvd_raw": item,
                    },
                }
            )

        page_results = len(vulnerabilities)
        if page_results <= 0 or not vulnerabilities:
            break
        start_index += page_results
        request_count += 1

    return records


def fetch_github_advisories(url: str, *, fetched_at: str, per_page: int, max_pages: int, token: str | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        page_url = f"{url}?{parse.urlencode({'per_page': per_page, 'page': page, 'sort': 'published', 'direction': 'desc'})}"
        payload = _read_json(page_url, token=token, service="github")
        if not isinstance(payload, list) or not payload:
            break
        for item in payload:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("ghsa_id") or item.get("id") or f"ghsa-page-{page}-{len(records) + 1}")
            records.append(
                {
                    "source_id": source_id,
                    "source_type": "advisory",
                    "title": str(item.get("summary") or item.get("ghsa_id") or source_id),
                    "raw_text": str(item.get("description") or item.get("summary") or json.dumps(item, ensure_ascii=True)),
                    "url": item.get("html_url") if isinstance(item.get("html_url"), str) else None,
                    "cve_id": item.get("cve_id") if isinstance(item.get("cve_id"), str) else None,
                    "ghsa_id": item.get("ghsa_id") if isinstance(item.get("ghsa_id"), str) else None,
                    "published_at": item.get("published_at") if isinstance(item.get("published_at"), str) else None,
                    "severity": item.get("severity") if isinstance(item.get("severity"), str) else None,
                    "affected_components": _ghsa_affected_components(item),
                    "metadata": {
                        "source": "github_security_advisories",
                        "fetched_at": fetched_at,
                        "source_url": page_url,
                        "github_raw": item,
                    },
                }
            )
    return records


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(records, handle, ensure_ascii=True, sort_keys=True, indent=2)
        handle.write("\n")


def _read_json(url: str, *, token: str | None = None, service: str) -> Any:
    req = request.Request(url, method="GET")
    if service == "github":
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", GITHUB_API_VERSION)
    else:
        req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "cve-synth/0.1.0")
    if token:
        if service == "github":
            req.add_header("Authorization", f"Bearer {token}")
        else:
            req.add_header("apiKey", token)
    try:
        with request.urlopen(req, timeout=120.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if service == "nvd":
            raise RuntimeError(f"NVD CVE API fetch failed with HTTP {exc.code}: {body}") from exc
        raise RuntimeError(f"GitHub advisories fetch failed with HTTP {exc.code}: {body}") from exc


def _format_nvd_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _nvd_title(cve: dict[str, Any], cve_id: str) -> str:
    descriptions = cve.get("descriptions") if isinstance(cve.get("descriptions"), list) else []
    for description in descriptions:
        if isinstance(description, dict) and description.get("lang") == "en" and isinstance(description.get("value"), str):
            value = description["value"].strip()
            if value:
                return f"{cve_id} {value[:120]}"
    return cve_id


def _nvd_description(cve: dict[str, Any]) -> str:
    descriptions = cve.get("descriptions") if isinstance(cve.get("descriptions"), list) else []
    for description in descriptions:
        if isinstance(description, dict) and description.get("lang") == "en" and isinstance(description.get("value"), str):
            value = description["value"].strip()
            if value:
                return value
    return json.dumps(cve, ensure_ascii=True)


def _nvd_url(cve_id: str) -> str:
    return f"https://nvd.nist.gov/vuln/detail/{cve_id}"


def _nvd_severity(cve: dict[str, Any]) -> str | None:
    metrics = cve.get("metrics") if isinstance(cve.get("metrics"), dict) else {}
    severity_fields = ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]
    for field_name in severity_fields:
        metric_list = metrics.get(field_name)
        if not isinstance(metric_list, list) or not metric_list:
            continue
        first_metric = metric_list[0]
        if not isinstance(first_metric, dict):
            continue
        cvss_data = first_metric.get("cvssData") if isinstance(first_metric.get("cvssData"), dict) else {}
        base_severity = cvss_data.get("baseSeverity")
        if isinstance(base_severity, str) and base_severity:
            return base_severity
        legacy_severity = first_metric.get("baseSeverity")
        if isinstance(legacy_severity, str) and legacy_severity:
            return legacy_severity
    return None


def _nvd_affected_components(cve: dict[str, Any]) -> list[str]:
    configurations = cve.get("configurations") if isinstance(cve.get("configurations"), list) else []
    components: list[str] = []
    for config in configurations:
        if not isinstance(config, dict):
            continue
        nodes = config.get("nodes") if isinstance(config.get("nodes"), list) else []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            cpe_matches = node.get("cpeMatch") if isinstance(node.get("cpeMatch"), list) else []
            for match in cpe_matches:
                if not isinstance(match, dict):
                    continue
                cpe_name = match.get("criteria")
                if isinstance(cpe_name, str) and cpe_name:
                    components.append(cpe_name)
    # preserve insertion order while deduplicating
    seen: set[str] = set()
    deduped: list[str] = []
    for component in components:
        if component in seen:
            continue
        seen.add(component)
        deduped.append(component)
    return deduped


def _ghsa_affected_components(item: dict[str, Any]) -> list[str]:
    vulnerabilities = item.get("vulnerabilities") if isinstance(item.get("vulnerabilities"), list) else []
    components: list[str] = []
    for vulnerability in vulnerabilities:
        if not isinstance(vulnerability, dict):
            continue
        package = vulnerability.get("package") if isinstance(vulnerability.get("package"), dict) else {}
        ecosystem = package.get("ecosystem") if isinstance(package.get("ecosystem"), str) else ""
        package_name = package.get("name") if isinstance(package.get("name"), str) else ""
        component = f"{ecosystem}:{package_name}".strip(":")
        if component:
            components.append(component)
    seen: set[str] = set()
    deduped: list[str] = []
    for component in components:
        if component in seen:
            continue
        seen.add(component)
        deduped.append(component)
    return deduped


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
