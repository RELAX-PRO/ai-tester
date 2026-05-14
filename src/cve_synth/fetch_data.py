from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request
import argparse
import gzip
import json
import os
import sys


NVD_RECENT_URL = "https://services.nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-recent.json.gz"
GITHUB_ADVISORIES_URL = "https://api.github.com/advisories"
GITHUB_API_VERSION = "2026-03-10"


@dataclass(slots=True)
class RawSnapshot:
    source: str
    source_id: str
    fetched_at: str
    source_url: str
    payload: dict[str, Any]

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "source": self.source,
                "source_id": self.source_id,
                "fetched_at": self.fetched_at,
                "source_url": self.source_url,
                "payload": self.payload,
            },
            ensure_ascii=True,
            sort_keys=True,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download recent NVD and GitHub advisory data into data/raw.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Directory to store normalized raw snapshots")
    parser.add_argument("--nvd-url", default=NVD_RECENT_URL, help="Override the NVD CVE-Recent JSON 2.0 feed URL")
    parser.add_argument("--github-url", default=GITHUB_ADVISORIES_URL, help="Override the GitHub Security Advisories API URL")
    parser.add_argument("--github-per-page", type=int, default=100, help="Advisories per GitHub API page")
    parser.add_argument("--github-max-pages", type=int, default=1, help="Number of GitHub pages to fetch")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fetched_at = datetime.now(tz=timezone.utc).isoformat()
    nvd_snapshots = fetch_nvd_recent(args.nvd_url, fetched_at)
    github_snapshots = fetch_github_advisories(
        args.github_url,
        fetched_at=fetched_at,
        per_page=args.github_per_page,
        max_pages=args.github_max_pages,
        token=os.environ.get("GITHUB_TOKEN", "").strip() or None,
    )

    write_jsonl(output_dir / "nvd_cve_recent.jsonl", nvd_snapshots)
    write_jsonl(output_dir / "github_security_advisories.jsonl", github_snapshots)
    print({"nvd_records": len(nvd_snapshots), "github_records": len(github_snapshots), "output_dir": str(output_dir)})
    return 0


def fetch_nvd_recent(url: str, fetched_at: str) -> list[RawSnapshot]:
    payload = _read_json_gz(url)
    vulnerabilities = payload.get("vulnerabilities", [])
    snapshots: list[RawSnapshot] = []
    for item in vulnerabilities:
        if isinstance(item, dict):
            cve = item.get("cve", {}) if isinstance(item.get("cve"), dict) else {}
            cve_id = ""
            if isinstance(cve, dict):
                cve_id = str(cve.get("id") or "")
            if not cve_id:
                metadata = item.get("cveMetadata", {}) if isinstance(item.get("cveMetadata"), dict) else {}
                cve_id = str(metadata.get("cveId") or "")
        else:
            cve_id = ""
        source_id = cve_id or f"nvd-recent-{len(snapshots) + 1}"
        if not source_id:
            source_id = f"nvd-recent-{len(snapshots) + 1}"
        snapshots.append(
            RawSnapshot(
                source="nvd_cve_recent",
                source_id=source_id,
                fetched_at=fetched_at,
                source_url=url,
                payload=item if isinstance(item, dict) else {"value": item},
            )
        )
    return snapshots


def fetch_github_advisories(url: str, *, fetched_at: str, per_page: int, max_pages: int, token: str | None) -> list[RawSnapshot]:
    snapshots: list[RawSnapshot] = []
    for page in range(1, max_pages + 1):
        page_url = f"{url}?{parse.urlencode({'per_page': per_page, 'page': page, 'sort': 'published', 'direction': 'desc'})}"
        payload = _read_json(page_url, token=token)
        if not isinstance(payload, list) or not payload:
            break
        for item in payload:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("ghsa_id") or item.get("id") or f"ghsa-page-{page}-{len(snapshots) + 1}")
            snapshots.append(
                RawSnapshot(
                    source="github_security_advisories",
                    source_id=source_id,
                    fetched_at=fetched_at,
                    source_url=page_url,
                    payload=item,
                )
            )
    return snapshots


def write_jsonl(path: Path, snapshots: list[RawSnapshot]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for snapshot in snapshots:
            handle.write(snapshot.to_jsonl())
            handle.write("\n")


def _read_json_gz(url: str) -> dict[str, Any]:
    req = request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", "cve-synth/0.1.0")
    with request.urlopen(req, timeout=120.0) as response:
        data = response.read()
    return json.loads(gzip.decompress(data).decode("utf-8"))


def _read_json(url: str, *, token: str | None = None) -> Any:
    req = request.Request(url, method="GET")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", GITHUB_API_VERSION)
    req.add_header("User-Agent", "cve-synth/0.1.0")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with request.urlopen(req, timeout=120.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub advisories fetch failed with HTTP {exc.code}: {body}") from exc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
