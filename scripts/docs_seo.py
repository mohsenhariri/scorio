"""Finalize and validate the assembled GitHub Pages site using only the stdlib.

Run after copying the landing page, Documenter, and TypeDoc output into one
directory. Generated files are build artifacts, never hand-maintained URL lists.
"""

from __future__ import annotations

import argparse
import re
from collections import deque
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit
from xml.etree import ElementTree as ET

BASE_URL = "https://mohsenhariri.github.io/scorio/"
SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
# These are errors or legacy redirects, not search landing pages. An unexpected
# noindex anywhere else is a build failure, rather than a silently omitted page.
EXCLUDED = {"404.html", "poster/index.html", "tts/index.html"}
VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


def normalize(text: str) -> str:
    return " ".join(text.split())


class Page(HTMLParser):
    """Read static content and metadata without executing any JavaScript."""

    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict[str, str | None]]] = []
        self.titles: list[str] = []
        self.descriptions: list[str] = []
        self.canonicals: list[str] = []
        self.robots: set[str] = set()
        self.refresh = False
        self.links: list[str] = []
        self.references: list[str] = []
        self.paragraphs: list[str] = []
        self.text: list[str] = []
        self.metadata_tags: list[str] = []
        self.title_text: list[str] = []
        self.paragraph_text: list[str] = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        in_head = any(t == "head" for t, _ in self.stack)
        if in_head and tag == "meta":
            name = (attrs.get("name") or "").lower()
            value = attrs.get("content") or ""
            if name == "description":
                self.descriptions.append(value)
                self.metadata_tags.append(self.get_starttag_text())
            if name in {"robots", "googlebot"}:
                self.robots.update(re.split(r"[\s,]+", value.lower()))
            if (attrs.get("http-equiv") or "").lower() == "refresh":
                self.refresh = True
        if (
            in_head
            and tag == "link"
            and "canonical" in (attrs.get("rel") or "").lower().split()
        ):
            self.canonicals.append(attrs.get("href") or "")
            self.metadata_tags.append(self.get_starttag_text())
        href = attrs.get("href")
        if tag == "a" and href is not None:
            if "nofollow" not in (attrs.get("rel") or "").lower().split():
                self.links.append(href)
            self.references.append(href)
        if tag in {"img", "script", "source"} and attrs.get("src"):
            self.references.append(attrs["src"])
        if tag == "link" and "stylesheet" in (attrs.get("rel") or "").split():
            self.references.append(href or "")
        if tag == "title":
            self.title_text = []
        if tag == "p":
            self.paragraph_text = []
        if tag not in VOID_TAGS:
            self.stack.append((tag, attrs))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "title":
            self.titles.append(normalize("".join(self.title_text)))
        if tag == "p" and self.in_content():
            paragraph = normalize("".join(self.paragraph_text))
            if len(paragraph) >= 30:
                self.paragraphs.append(paragraph)
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def in_content(self):
        return any(
            tag == "main"
            or attrs.get("id") == "documenter-page"
            or "col-content" in (attrs.get("class") or "").split()
            for tag, attrs in self.stack
        )

    def handle_data(self, data):
        if any(t == "title" for t, _ in self.stack):
            self.title_text.append(data)
        if any(t in {"script", "style", "nav"} for t, _ in self.stack):
            return
        if self.in_content():
            self.text.append(data)
            if any(t == "p" for t, _ in self.stack):
                self.paragraph_text.append(data)


def canonical_url(relative: str) -> str:
    if Path(relative).name == "index.html":
        relative = relative.removesuffix("index.html")
    return BASE_URL + quote(relative, safe="/")


def page_files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*.html"))
        if path.relative_to(root).as_posix() not in EXCLUDED
    }


def local_file(root: Path, base: str, href: str) -> Path | None:
    url = urlsplit(urljoin(base, href))
    site = urlsplit(BASE_URL)
    if url.netloc != site.netloc or not url.path.startswith(site.path):
        return None
    relative = unquote(url.path[len(site.path) :])
    if url.path.endswith("/"):
        relative += "index.html"
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"URL escapes site directory: {href}")
    return target


def javascript_metadata(relative: str, page: Page) -> tuple[str, str]:
    path = Path(relative)
    if relative == "javascript/index.html":
        return (
            "Scorio JavaScript Documentation | LLM Evaluation and Ranking",
            "Scorio for JavaScript and TypeScript: install the npm package, evaluate "
            "LLMs with Bayes@N, rank models, and select answers. Node.js and browser guides.",
        )
    if path.parent.name == "documents":
        label = page.titles[0].split(" | ")[0]
    elif path.parent.name == "modules":
        label = "scorio/" + path.stem.replace(".", "/")
    else:
        # TypeDoc titles omit the namespace: eval.avg and rank.avg otherwise
        # have identical titles even though they document different functions.
        label = path.stem
    title = f"{label} | Scorio JavaScript Documentation"
    summary = page.paragraphs[0] if page.paragraphs else "API signatures and types."
    description = normalize(f"{label}: {summary} JavaScript and TypeScript reference.")
    if len(description) > 200:
        description = description[:197].rsplit(" ", 1)[0] + "..."
    return title, description


def finalize_metadata(root: Path):
    # Documenter normally writes these during deploydocs(), which this combined
    # deployment does not use. This site publishes a single, unversioned edition.
    (root / "julia/siteinfo.js").write_text(
        "var DOCUMENTER_VERSION_SELECTOR_DISABLED = true;\n", encoding="utf-8"
    )
    (root / "versions.js").write_text("var DOC_VERSIONS = [];\n", encoding="utf-8")
    for relative, path in page_files(root).items():
        html = path.read_text(encoding="utf-8")
        page = Page(html)
        if page.refresh or page.robots & {"noindex", "none", "nofollow"}:
            raise ValueError(f"{relative}: unexpected redirect or indexing restriction")
        if len(page.titles) != 1:
            raise ValueError(f"{relative}: expected exactly one title")
        title = page.titles[0]
        description = page.descriptions[0] if page.descriptions else ""
        if relative.startswith("javascript/"):
            title, description = javascript_metadata(relative, page)
        # Only rewrite head metadata. Leave content, scripts, and generator
        # markup intact, and never change the canonicals of legacy redirects.
        head, rest = re.split(r"</head\s*>", html, maxsplit=1, flags=re.I)
        for tag in page.metadata_tags:
            head = head.replace(tag, "", 1)
        head = re.sub(
            r"<title\b[^>]*>.*?</title\s*>",
            lambda _, title=title: f"<title>{escape(title)}</title>",
            head,
            count=1,
            flags=re.I | re.S,
        )
        metadata = (
            f'<link rel="canonical" href="{escape(canonical_url(relative))}">'
            f'<meta name="description" content="{escape(description, quote=True)}">'
        )
        path.write_text(head + metadata + "</head>" + rest, encoding="utf-8")


def write_sitemaps(root: Path):
    groups: dict[str, list[str]] = {"Scorio": [], "Julia": [], "JavaScript": []}
    for relative, path in page_files(root).items():
        if relative == "sitemap.html":
            continue
        page = Page(path.read_text(encoding="utf-8"))
        group = (
            "Julia"
            if relative.startswith("julia/")
            else ("JavaScript" if relative.startswith("javascript/") else "Scorio")
        )
        groups[group].append(
            f'<li><a href="{escape(canonical_url(relative))}">'
            f"{escape(page.titles[0])}</a></li>"
        )
    sections = "\n".join(
        f"<section><h2>{group}</h2><ul>{''.join(links)}</ul></section>"
        for group, links in groups.items()
    )
    (root / "sitemap.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Scorio Documentation Index | Julia and JavaScript APIs</title>"
        '<meta name="description" content="Browse Scorio documentation, Julia API '
        "references, and JavaScript and TypeScript guides and APIs for LLM evaluation "
        'and model ranking.">'
        f'<link rel="canonical" href="{BASE_URL}sitemap.html">'
        "<style>body{font:1rem/1.6 system-ui;max-width:75rem;margin:2rem auto;"
        "padding:0 1rem}a{overflow-wrap:anywhere}li{margin:.35rem 0}</style>"
        "</head><body><main><h1>Scorio documentation index</h1>"
        f'<p><a href="{BASE_URL}">Scorio home</a> · '
        '<a href="https://scorio.readthedocs.io/">Python documentation</a></p>'
        f"{sections}</main></body></html>\n",
        encoding="utf-8",
    )
    ET.register_namespace("", SITEMAP_NS)
    urlset = ET.Element(f"{{{SITEMAP_NS}}}urlset")
    for relative in page_files(root):
        url = ET.SubElement(urlset, f"{{{SITEMAP_NS}}}url")
        ET.SubElement(url, f"{{{SITEMAP_NS}}}loc").text = canonical_url(relative)
    # Build timestamps aren't content modification dates. Omit lastmod rather
    # than inventing a fresh date at each deployment; Google ignores priority
    # and changefreq. See Google's build-sitemap guidance.
    ET.indent(urlset, space="  ")
    ET.ElementTree(urlset).write(
        root / "sitemap.xml", encoding="utf-8", xml_declaration=True
    )


def validate(root: Path):
    files = page_files(root)
    for required in (
        "index.html",
        "julia/index.html",
        "javascript/index.html",
        "sitemap.html",
    ):
        if required not in files:
            raise ValueError(f"Missing required page: {required}")
    titles: dict[str, str] = {}
    graph: dict[Path, set[Path]] = {}
    for relative, path in files.items():
        page = Page(path.read_text(encoding="utf-8"))
        expected = canonical_url(relative)
        if page.canonicals != [expected]:
            raise ValueError(f"{relative}: canonical must be exactly {expected}")
        if len(page.descriptions) != 1 or not page.descriptions[0].strip():
            raise ValueError(f"{relative}: missing or duplicate description")
        if page.refresh or page.robots & {"noindex", "none", "nofollow"}:
            raise ValueError(f"{relative}: unexpected redirect or indexing restriction")
        if len(page.titles) != 1 or not page.titles[0]:
            raise ValueError(f"{relative}: missing or duplicate title")
        title = page.titles[0]
        if title in titles:
            raise ValueError(f"Duplicate title: {relative} and {titles[title]}")
        titles[title] = relative
        if len(normalize(" ".join(page.text))) < 30:
            raise ValueError(f"{relative}: missing static main content")
        for href in page.references:
            target = local_file(root, expected, href)
            if target is not None and not target.is_file():
                raise ValueError(f"{relative}: broken local link or resource: {href}")
        graph[path.resolve()] = {
            target
            for href in page.links
            if (target := local_file(root, expected, href)) is not None
        }
    reachable: set[Path] = set()
    queue = deque([(root / "index.html").resolve()])
    while queue:
        target = queue.popleft()
        if target not in reachable:
            reachable.add(target)
            queue.extend(graph.get(target, set()) - reachable)
    missing = graph.keys() - reachable
    if missing:
        raise ValueError(
            f"Pages unreachable from homepage links: {sorted(map(str, missing))}"
        )
    sitemap = ET.parse(root / "sitemap.xml")
    locations = [node.text for node in sitemap.findall(f".//{{{SITEMAP_NS}}}loc")]
    if len(locations) != len(set(locations)) or set(locations) != {
        canonical_url(relative) for relative in files
    }:
        raise ValueError("Sitemap must list every indexable canonical URL exactly once")
    print(
        f"SEO checks passed: {len(files)} static pages, unique titles, descriptions, "
        "canonicals, local links/assets, homepage reachability, and complete sitemap."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="Assembled GitHub Pages directory")
    parser.add_argument(
        "--check", action="store_true", help="Validate without modifying files"
    )
    args = parser.parse_args()
    if not args.check:
        # Refuse an incomplete assembly before modifying any output.
        for relative in ("index.html", "julia/index.html", "javascript/index.html"):
            if not (args.site / relative).is_file():
                parser.error(f"Missing required page: {relative}")
        finalize_metadata(args.site)
        write_sitemaps(args.site)
    validate(args.site)


if __name__ == "__main__":
    main()
