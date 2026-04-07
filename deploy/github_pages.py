"""
Automatic report deployment to GitHub Pages.
Copies the HTML to docs/reports/, updates docs/index.html and pushes.
"""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


DOCS_DIR = Path("docs")
REPORTS_DIR = DOCS_DIR / "reports"


def deploy(report_path: str, repo_root: str = ".") -> str:
    """
    Copies the report to docs/reports/, regenerates the index and runs git push.

    Returns the public URL (GitHub Pages) if it can be deduced from the remote.
    """
    repo_root = Path(repo_root).resolve()
    reports_dir = repo_root / REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy report (it should already be in docs/reports/ but just in case)
    report_name = Path(report_path).name
    dest = reports_dir / report_name
    if Path(report_path).resolve() != dest.resolve():
        shutil.copy2(report_path, dest)

    # 2. Regenerate index
    _update_index(repo_root / DOCS_DIR, reports_dir)

    # 3. Git add + commit + push
    _git_commit_and_push(
        repo_root,
        message=f"Add rehearsal report {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )

    # 4. Deduce public URL
    url = _deduce_pages_url(repo_root, report_name)
    return url


def _update_index(docs_dir: Path, reports_dir: Path) -> None:
    """Regenerates docs/index.html with the list of reports sorted by date."""
    reports = sorted(reports_dir.glob("report_*.html"), reverse=True)
    items = ""
    for r in reports:
        name = r.name
        # Extract date from filename (report_YYYYMMDD_HHMMSS.html)
        date_part = name.replace("report_", "").replace(".html", "")
        try:
            dt = datetime.strptime(date_part, "%Y%m%d_%H%M%S")
            label = dt.strftime("%d/%m/%Y %H:%M")
        except ValueError:
            label = name
        items += f'    <li><a href="reports/{name}">{label}</a></li>\n'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Music Coach — Reports</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; background:#0f1117; color:#e8e8f0; }}
    h1 {{ color:#7c6fff; }} a {{ color:#7c6fff; }} ul {{ list-style:none; padding:0; }}
    li {{ padding:12px 0; border-bottom:1px solid #2d3148; }} li:last-child {{ border-bottom:none; }}
    .muted {{ color:#8888aa; font-size:.85rem; margin-top:8px; }}
  </style>
</head>
<body>
  <h1>🎵 Music Coach — Rehearsal Reports</h1>
  <p class="muted">Most recent reports appear first.</p>
  <ul>
{items}  </ul>
</body>
</html>
"""
    (docs_dir / "index.html").write_text(html, encoding="utf-8")


def _git_commit_and_push(repo_root: Path, message: str) -> None:
    """Runs git add / commit / push in the repo."""
    def run(cmd):
        result = subprocess.run(
            cmd, cwd=str(repo_root), capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Error running '{' '.join(cmd)}':\n{result.stderr}"
            )
        return result.stdout.strip()

    run(["git", "add", "docs/"])
    # Check if there are staged changes
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(repo_root),
    )
    if status.returncode == 0:
        print("  (no changes in docs/ to commit)")
        return

    run(["git", "commit", "-m", message])
    run(["git", "push", "-u", "origin", "HEAD"])
    print(f"  ✓ Push complete: {message}")


def _deduce_pages_url(repo_root: Path, report_name: str) -> str:
    """Tries to deduce the GitHub Pages URL from the origin remote."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(repo_root), capture_output=True, text=True
        )
        remote = result.stdout.strip()
        # Formats: https://github.com/user/repo.git  or  git@github.com:user/repo.git
        if "github.com" in remote:
            remote = remote.replace("git@github.com:", "https://github.com/")
            remote = remote.removesuffix(".git")
            parts = remote.rstrip("/").split("/")
            if len(parts) >= 2:
                user, repo = parts[-2], parts[-1]
                return f"https://{user}.github.io/{repo}/reports/{report_name}"
    except Exception:
        pass
    return f"https://<user>.github.io/<repo>/reports/{report_name}"
