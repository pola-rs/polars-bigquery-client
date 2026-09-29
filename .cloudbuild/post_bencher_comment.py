#!/usr/bin/env python3
"""Post or update a GitHub PR comment with Bencher benchmark result links."""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

COMMENT_MARKER = "<!-- bencher-comment -->"
GITHUB_API_URL = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"

_REPORT_LINE_RE = re.compile(r"^View report:\s+(https?://\S+)\s*$")
_BULLET_LINE_RE = re.compile(r"^-\s+(.+?):\s+(https?://\S+)\s*$")
_ITERATION_LINE_RE = re.compile(r"^Iteration\s+\d+:\s*$")


def extract_bencher_summary(log_text: str) -> str | None:
    """Extract the final Bencher human report block from `bencher run` output."""
    lines = log_text.splitlines()
    start_idx: int | None = None
    for idx, line in enumerate(lines):
        if _REPORT_LINE_RE.match(line.strip()):
            start_idx = idx

    if start_idx is None:
        return None

    formatted_lines: list[str] = []
    in_alerts = False

    for raw_line in lines[start_idx:]:
        line = raw_line.strip()
        if not line:
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            continue

        report_match = _REPORT_LINE_RE.match(line)
        if report_match:
            in_alerts = False
            formatted_lines.append(f"[View report]({report_match.group(1)})")
            continue

        if line == "WARNING: No benchmarks found!":
            formatted_lines.append("> [!WARNING]\n> No benchmarks found!")
            continue

        if line == "View results:":
            in_alerts = False
            formatted_lines.append("**View results:**")
            continue

        if line == "View alerts:":
            in_alerts = True
            formatted_lines.append("**🚨 View alerts:**")
            continue

        if _ITERATION_LINE_RE.match(line):
            formatted_lines.append(f"*{line}*")
            continue

        bullet_match = _BULLET_LINE_RE.match(line)
        if bullet_match:
            label, url = bullet_match.group(1), bullet_match.group(2)
            prefix = "- 🚨 " if in_alerts else "- "
            formatted_lines.append(f"{prefix}[{label}]({url})")
            continue

        # Stop if trailing non-report output is encountered.
        break

    while formatted_lines and formatted_lines[-1] == "":
        formatted_lines.pop()

    return "\n".join(formatted_lines) if formatted_lines else None


def build_comment_body(sections: list[tuple[str, pathlib.Path]]) -> str | None:
    """Build the Markdown comment body from the available benchmark log files."""
    rendered_sections: list[str] = []
    for title, path in sections:
        if not path.is_file():
            continue
        summary = extract_bencher_summary(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if summary:
            rendered_sections.append(f"### {title}\n\n{summary}")

    if not rendered_sections:
        return None

    body_parts = [
        COMMENT_MARKER,
        "## 🐰 Bencher Continuous Benchmarking Results",
        *rendered_sections,
    ]
    return "\n\n".join(body_parts) + "\n"


def _github_request(
    url: str,
    token: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> object:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "polars-bigquery-client-cloudbuild",
    }
    data: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def find_existing_comment_id(repo: str, pr_number: str, token: str) -> int | None:
    """Find an existing PR comment containing COMMENT_MARKER."""
    for page in range(1, 11):
        url = f"{GITHUB_API_URL}/repos/{repo}/issues/{pr_number}/comments?per_page=100&page={page}"
        comments = _github_request(url, token)
        if not isinstance(comments, list) or not comments:
            break
        for comment in comments:
            if isinstance(comment, dict):
                body = comment.get("body")
                comment_id = comment.get("id")
                if (
                    isinstance(body, str)
                    and COMMENT_MARKER in body
                    and isinstance(comment_id, int)
                ):
                    return comment_id
        if len(comments) < 100:
            break
    return None


def upsert_pr_comment(repo: str, pr_number: str, token: str, body: str) -> None:
    """Update the existing Bencher PR comment if present, or create a new one."""
    comment_id = find_existing_comment_id(repo, pr_number, token)
    if comment_id is not None:
        url = f"{GITHUB_API_URL}/repos/{repo}/issues/comments/{comment_id}"
        _github_request(url, token, method="PATCH", payload={"body": body})
        print(
            f"Updated existing Bencher PR comment ({comment_id}) on {repo}#{pr_number}."
        )
    else:
        url = f"{GITHUB_API_URL}/repos/{repo}/issues/{pr_number}/comments"
        _github_request(url, token, method="POST", payload={"body": body})
        print(f"Created new Bencher PR comment on {repo}#{pr_number}.")


def main() -> int:
    pr_number = os.environ.get("_PR_NUMBER", "").strip()
    if not pr_number:
        print(
            "Not a pull request build (_PR_NUMBER is not set); skipping GitHub comment."
        )
        return 0

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print(
            "Warning: GITHUB_TOKEN is not set; skipping GitHub PR comment.",
            file=sys.stderr,
        )
        return 0

    repo = (
        os.environ.get("REPO_FULL_NAME", "").strip()
        or os.environ.get("_GITHUB_REPO", "").strip()
        or "pola-rs/polars-bigquery-client"
    )

    workspace = pathlib.Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
    sections = [
        ("Rust (`arrow-bigquery`)", workspace / "bencher_rust.log"),
        ("Python (`py-polars-bigquery`)", workspace / "bencher_python.log"),
    ]

    body = build_comment_body(sections)
    if not body:
        print("No Bencher report output found in log files; skipping GitHub comment.")
        return 0

    try:
        upsert_pr_comment(repo, pr_number, token, body)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(
            f"Warning: Failed to post Bencher PR comment to GitHub (HTTP {exc.code}): {error_body}",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: Failed to post Bencher PR comment to GitHub: {exc}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
