"""
Despliegue automático del informe en GitHub Pages.
Copia el HTML a docs/reports/, actualiza docs/index.html y hace push.
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
    Copia el informe a docs/reports/, regenera el índice y hace git push.

    Returns la URL pública (GitHub Pages) si se puede deducir del remote.
    """
    repo_root = Path(repo_root).resolve()
    reports_dir = repo_root / REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copiar informe (ya debería estar en docs/reports/ pero por si acaso)
    report_name = Path(report_path).name
    dest = reports_dir / report_name
    if Path(report_path).resolve() != dest.resolve():
        shutil.copy2(report_path, dest)

    # 2. Regenerar índice
    _update_index(repo_root / DOCS_DIR, reports_dir)

    # 3. Git add + commit + push
    _git_commit_and_push(
        repo_root,
        message=f"Add rehearsal report {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )

    # 4. Deducir URL pública
    url = _deduce_pages_url(repo_root, report_name)
    return url


def _update_index(docs_dir: Path, reports_dir: Path) -> None:
    """Regenera docs/index.html con la lista de informes ordenados por fecha."""
    reports = sorted(reports_dir.glob("report_*.html"), reverse=True)
    items = ""
    for r in reports:
        name = r.name
        # Extraer fecha del nombre (report_YYYYMMDD_HHMMSS.html)
        date_part = name.replace("report_", "").replace(".html", "")
        try:
            dt = datetime.strptime(date_part, "%Y%m%d_%H%M%S")
            label = dt.strftime("%d/%m/%Y %H:%M")
        except ValueError:
            label = name
        items += f'    <li><a href="reports/{name}">{label}</a></li>\n'

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Music Coach — Informes</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; background:#0f1117; color:#e8e8f0; }}
    h1 {{ color:#7c6fff; }} a {{ color:#7c6fff; }} ul {{ list-style:none; padding:0; }}
    li {{ padding:12px 0; border-bottom:1px solid #2d3148; }} li:last-child {{ border-bottom:none; }}
    .muted {{ color:#8888aa; font-size:.85rem; margin-top:8px; }}
  </style>
</head>
<body>
  <h1>🎵 Music Coach — Informes de ensayo</h1>
  <p class="muted">Los informes más recientes aparecen primero.</p>
  <ul>
{items}  </ul>
</body>
</html>
"""
    (docs_dir / "index.html").write_text(html, encoding="utf-8")


def _git_commit_and_push(repo_root: Path, message: str) -> None:
    """Ejecuta git add / commit / push en el repo."""
    def run(cmd):
        result = subprocess.run(
            cmd, cwd=str(repo_root), capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Error en '{' '.join(cmd)}':\n{result.stderr}"
            )
        return result.stdout.strip()

    run(["git", "add", "docs/"])
    # Comprobar si hay cambios staged
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=str(repo_root),
    )
    if status.returncode == 0:
        print("  (sin cambios en docs/ para commitear)")
        return

    run(["git", "commit", "-m", message])
    run(["git", "push", "-u", "origin", "HEAD"])
    print(f"  ✓ Push completado: {message}")


def _deduce_pages_url(repo_root: Path, report_name: str) -> str:
    """Intenta deducir la URL de GitHub Pages a partir del remote origin."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(repo_root), capture_output=True, text=True
        )
        remote = result.stdout.strip()
        # Formatos: https://github.com/user/repo.git  o  git@github.com:user/repo.git
        if "github.com" in remote:
            remote = remote.replace("git@github.com:", "https://github.com/")
            remote = remote.removesuffix(".git")
            parts = remote.rstrip("/").split("/")
            if len(parts) >= 2:
                user, repo = parts[-2], parts[-1]
                return f"https://{user}.github.io/{repo}/reports/{report_name}"
    except Exception:
        pass
    return f"https://<usuario>.github.io/<repo>/reports/{report_name}"
