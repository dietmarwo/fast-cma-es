#!/usr/bin/env python3
"""Build and validate the AsciiDoc GitHub Pages site."""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"
DEFAULT_OUTPUT = ROOT / "target" / "site"
STAGING = ROOT / "target" / "docs-source"

API_DOCUMENTS = (
    ("fcmaes_documentation.html", "Python API"),
    ("cpp_documentation.html", "C++ API"),
    ("examples_documentation.html", "Examples API"),
)

REPOSITORY_ADOC_LINK = re.compile(
    r"https://github\.com/dietmarwo/fast-cma-es/"
    r"(?:blob|tree)/master/(?P<path>[^\[]+?\.adoc)",
    re.IGNORECASE,
)
TUTORIAL_INDEX_LINK = re.compile(
    r"https://github\.com/dietmarwo/fast-cma-es/"
    r"(?:blob|tree)/master/tutorials/(?P<name>[^/\[]+?)\.adoc\[",
    re.IGNORECASE,
)
IMAGE_MACRO = re.compile(r"(?<!:)image::?(?P<target>[^\[]+)\[")
IMAGES_DIR = re.compile(r"^:imagesdir:\s*(?P<value>.+?)\s*$", re.MULTILINE)
CHAPTER_TITLE = re.compile(r"^==\s+(?P<title>.+?)\s*$", re.MULTILINE)

NAVIGATION_STYLE = """
<style id="fcmaes-book-navigation">
.fcmaes-book-nav {
  align-items: center;
  background: #f7f7f8;
  border: 1px solid #d8d8dc;
  border-radius: .35rem;
  display: flex;
  flex-wrap: wrap;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: .9rem;
  gap: .35rem .75rem;
  justify-content: center;
  margin: 1rem auto 1.5rem;
  max-width: 78rem;
  padding: .65rem 1rem;
}
.fcmaes-book-nav a {
  color: #215a7a;
  text-decoration: none;
}
.fcmaes-book-nav a:hover {
  text-decoration: underline;
}
.fcmaes-book-nav .fcmaes-book-separator {
  color: #8a8a8f;
}
</style>
""".strip()


class ResourceParser(HTMLParser):
    """Collect local resource references from one generated HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.references: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        if tag in {"a", "link"} and attributes.get("href"):
            self.references.append(attributes["href"] or "")
        if tag in {"img", "script", "iframe", "source"} and attributes.get("src"):
            self.references.append(attributes["src"] or "")
        if tag == "object" and attributes.get("data"):
            self.references.append(attributes["data"] or "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build README, tutorials, and API references as one site."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"site output directory (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    return parser.parse_args()


def tutorial_sources() -> list[Path]:
    sources = sorted(
        path for path in TUTORIALS.glob("*.adoc") if path.name != "README.adoc"
    )
    by_lower_name = {path.name.lower(): path for path in sources}
    index = (TUTORIALS / "README.adoc").read_text(encoding="utf-8")
    ordered: list[Path] = []
    for match in TUTORIAL_INDEX_LINK.finditer(index):
        source = by_lower_name.get(f"{match.group('name')}.adoc".lower())
        if source is not None and source not in ordered:
            ordered.append(source)
    ordered.extend(source for source in sources if source not in ordered)
    return ordered


def chapter_title(source: Path) -> str:
    match = CHAPTER_TITLE.search(source.read_text(encoding="utf-8"))
    return match.group("title") if match else source.stem


def local_adoc_target(
    raw_path: str,
    *,
    tutorial_page: bool,
    tutorials_by_lower_name: dict[str, Path],
) -> str | None:
    path = unquote(raw_path).rstrip("/")
    name = Path(path).name
    lower_name = name.lower()
    if lower_name == "readme.adoc":
        if path.lower().startswith("tutorials/"):
            return "index.html" if tutorial_page else "tutorials/index.html"
        return "../index.html" if tutorial_page else "index.html"
    source = tutorials_by_lower_name.get(lower_name)
    if source is None:
        return None
    target = f"{source.stem}.html"
    return target if tutorial_page else f"tutorials/{target}"


def rewrite_repository_links(
    text: str,
    *,
    tutorial_page: bool,
    tutorials_by_lower_name: dict[str, Path],
) -> str:
    def replacement(match: re.Match[str]) -> str:
        local = local_adoc_target(
            match.group("path"),
            tutorial_page=tutorial_page,
            tutorials_by_lower_name=tutorials_by_lower_name,
        )
        return f"link:{local}" if local is not None else match.group(0)

    return REPOSITORY_ADOC_LINK.sub(replacement, text)


def rewrite_api_document_links(
    page: Path,
    tutorials_by_lower_name: dict[str, Path],
) -> None:
    """Point tutorial links in generated API HTML at the local book pages."""

    def replacement(match: re.Match[str]) -> str:
        local = local_adoc_target(
            match.group("path"),
            tutorial_page=False,
            tutorials_by_lower_name=tutorials_by_lower_name,
        )
        return local if local is not None else match.group(0)

    content = page.read_text(encoding="utf-8")
    page.write_text(REPOSITORY_ADOC_LINK.sub(replacement, content), encoding="utf-8")


def prepare_sources(chapters: list[Path]) -> tuple[Path, list[Path]]:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    staged_tutorials = STAGING / "tutorials"
    staged_tutorials.mkdir(parents=True)
    by_lower_name = {path.name.lower(): path for path in chapters}

    root_source = STAGING / "README.adoc"
    root_source.write_text(
        rewrite_repository_links(
            (ROOT / "README.adoc").read_text(encoding="utf-8"),
            tutorial_page=False,
            tutorials_by_lower_name=by_lower_name,
        ),
        encoding="utf-8",
    )

    staged_sources: list[Path] = []
    for source in [TUTORIALS / "README.adoc", *chapters]:
        staged = staged_tutorials / source.name
        staged.write_text(
            rewrite_repository_links(
                source.read_text(encoding="utf-8"),
                tutorial_page=True,
                tutorials_by_lower_name=by_lower_name,
            ),
            encoding="utf-8",
        )
        staged_sources.append(staged)
    return root_source, staged_sources


def check_source_images(sources: list[Path]) -> None:
    missing: list[str] = []
    unresolved: list[str] = []
    for source in sources:
        text = source.read_text(encoding="utf-8")
        images_dir_match = IMAGES_DIR.search(text)
        images_dir = (
            images_dir_match.group("value").strip()
            if images_dir_match is not None
            else ""
        )
        for match in IMAGE_MACRO.finditer(text):
            target = match.group("target").strip()
            split = urlsplit(target)
            if split.scheme or target.startswith("//"):
                continue
            if "{" in target or "}" in target:
                unresolved.append(f"{source.relative_to(ROOT)}: {target}")
                continue
            path = source.parent / images_dir / unquote(split.path)
            if not path.is_file():
                missing.append(f"{source.relative_to(ROOT)}: {target}")
    if unresolved or missing:
        details = [
            *(f"unresolved image attribute: {value}" for value in unresolved),
            *(f"missing image: {value}" for value in missing),
        ]
        raise RuntimeError("\n".join(details))


def run_asciidoctor(source: Path, output: Path) -> None:
    executable = shutil.which("asciidoctor")
    if executable is None:
        raise RuntimeError(
            "asciidoctor is required; install Asciidoctor 2.0.23 or newer"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            executable,
            "--quiet",
            "--failure-level",
            "ERROR",
            "--doctype",
            "book",
            "--attribute",
            "toc=left",
            "--attribute",
            "sectanchors",
            "--attribute",
            "reproducible",
            "--attribute",
            "webfonts!",
            "--out-file",
            str(output),
            str(source),
        ],
        cwd=ROOT,
        check=True,
    )


def relative_href(current: Path, target: Path) -> str:
    return Path(os.path.relpath(target, current.parent)).as_posix()


def navigation_html(
    current: Path,
    output_root: Path,
    *,
    previous: tuple[Path, str] | None = None,
    following: tuple[Path, str] | None = None,
) -> str:
    links = [
        (output_root / "index.html", "Home"),
        (output_root / API_DOCUMENTS[0][0], API_DOCUMENTS[0][1]),
        (output_root / API_DOCUMENTS[1][0], API_DOCUMENTS[1][1]),
        (output_root / API_DOCUMENTS[2][0], API_DOCUMENTS[2][1]),
        (output_root / "tutorials" / "index.html", "Tutorials"),
    ]
    pieces = [
        '<nav class="fcmaes-book-nav" aria-label="Book navigation">',
        *[
            f'<a href="{html.escape(relative_href(current, target))}">'
            f"{html.escape(label)}</a>"
            for target, label in links
        ],
    ]
    if previous is not None:
        pieces.extend(
            [
                '<span class="fcmaes-book-separator">|</span>',
                f'<a rel="prev" href="{html.escape(relative_href(current, previous[0]))}">'
                f"← {html.escape(previous[1])}</a>",
            ]
        )
    if following is not None:
        pieces.extend(
            [
                '<span class="fcmaes-book-separator">|</span>',
                f'<a rel="next" href="{html.escape(relative_href(current, following[0]))}">'
                f"{html.escape(following[1])} →</a>",
            ]
        )
    pieces.append(
        '<span class="fcmaes-book-separator">|</span>'
        '<a href="https://github.com/dietmarwo/fast-cma-es">Source</a>'
    )
    pieces.append("</nav>")
    return "\n".join(pieces)


def inject_navigation(
    page: Path,
    output_root: Path,
    *,
    previous: tuple[Path, str] | None = None,
    following: tuple[Path, str] | None = None,
) -> None:
    content = page.read_text(encoding="utf-8")
    nav = navigation_html(
        page,
        output_root,
        previous=previous,
        following=following,
    )
    if "</head>" not in content:
        raise RuntimeError(f"generated page has no </head>: {page}")
    content = content.replace(
        "</head>",
        f"{NAVIGATION_STYLE}\n</head>",
        1,
    )
    body = re.search(r"<body(?:\s[^>]*)?>", content, re.IGNORECASE)
    if body is None:
        raise RuntimeError(f"generated page has no <body>: {page}")
    content = f"{content[:body.end()]}\n{nav}\n{content[body.end():]}"
    if "</body>" not in content:
        raise RuntimeError(f"generated page has no </body>: {page}")
    content = content.replace("</body>", f"{nav}\n</body>", 1)
    page.write_text(content, encoding="utf-8")


def check_generated_links(output_root: Path) -> None:
    missing: list[str] = []
    unconverted: list[str] = []
    for page in sorted(output_root.rglob("*.html")):
        content = page.read_text(encoding="utf-8")
        for match in re.findall(r"\b[\w./-]+\.html\[[^\]]+\]", content):
            unconverted.append(
                f"{page.relative_to(output_root)} -> {match}"
            )
        parser = ResourceParser()
        parser.feed(content)
        for reference in parser.references:
            if not reference or reference.startswith(("#", "//")):
                continue
            split = urlsplit(reference)
            if split.scheme in {"http", "https", "mailto", "data", "javascript"}:
                continue
            path_text = unquote(split.path)
            if not path_text:
                continue
            if path_text.startswith("/fast-cma-es/"):
                candidate = output_root / path_text.removeprefix("/fast-cma-es/")
            elif path_text.startswith("/"):
                continue
            else:
                candidate = page.parent / path_text
            if candidate.is_dir():
                candidate /= "index.html"
            if not candidate.exists():
                missing.append(
                    f"{page.relative_to(output_root)} -> {reference}"
                )
    if missing:
        raise RuntimeError(
            "generated site contains missing local resources:\n"
            + "\n".join(missing)
        )
    if unconverted:
        raise RuntimeError(
            "generated site contains unconverted AsciiDoc links:\n"
            + "\n".join(unconverted)
        )


def build(output_root: Path) -> None:
    output_root = output_root.resolve()
    if output_root == ROOT or ROOT not in output_root.parents:
        raise RuntimeError("output must be a dedicated directory inside the repository")

    chapters = tutorial_sources()
    source_pages = [ROOT / "README.adoc", TUTORIALS / "README.adoc", *chapters]
    check_source_images(source_pages)
    root_source, staged_sources = prepare_sources(chapters)

    if output_root.exists():
        shutil.rmtree(output_root)
    tutorial_output = output_root / "tutorials"
    tutorial_output.mkdir(parents=True)

    run_asciidoctor(root_source, output_root / "index.html")
    run_asciidoctor(staged_sources[0], tutorial_output / "index.html")
    for source in staged_sources[1:]:
        run_asciidoctor(source, tutorial_output / f"{source.stem}.html")

    shutil.copytree(TUTORIALS / "img", tutorial_output / "img")
    tutorials_by_lower_name = {path.name.lower(): path for path in chapters}
    for filename, _ in API_DOCUMENTS:
        output = output_root / filename
        shutil.copy2(ROOT / "docs" / filename, output)
        rewrite_api_document_links(output, tutorials_by_lower_name)
    (output_root / ".nojekyll").write_text("", encoding="utf-8")

    tutorial_pages = [
        (tutorial_output / "index.html", "Tutorials"),
        *[
            (tutorial_output / f"{source.stem}.html", chapter_title(source))
            for source in chapters
        ],
    ]
    for index, (page, _) in enumerate(tutorial_pages):
        previous = tutorial_pages[index - 1] if index > 0 else None
        following = (
            tutorial_pages[index + 1]
            if index + 1 < len(tutorial_pages)
            else None
        )
        inject_navigation(
            page,
            output_root,
            previous=previous,
            following=following,
        )

    inject_navigation(output_root / "index.html", output_root)
    for filename, _ in API_DOCUMENTS:
        inject_navigation(output_root / filename, output_root)

    check_generated_links(output_root)
    html_count = len(list(output_root.rglob("*.html")))
    image_count = len(list((tutorial_output / "img").iterdir()))
    expected_html = 1 + len(tutorial_pages) + len(API_DOCUMENTS)
    if html_count != expected_html:
        raise RuntimeError(
            f"expected {expected_html} HTML files, generated {html_count}"
        )
    print(
        f"built {html_count} HTML pages and copied {image_count} tutorial assets "
        f"to {output_root.relative_to(ROOT)}"
    )


def main() -> int:
    try:
        build(parse_args().output)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"documentation build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
