"""Regression checks for search discovery in the assembled documentation site."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from scripts.docs_seo import (
    BASE_URL,
    SITEMAP_NS,
    Page,
    canonical_url,
    finalize_metadata,
    validate,
    write_sitemaps,
)


class DocumentationSEOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.write_page(
            "index.html", "Scorio", '<a href="sitemap.html">Documentation index</a>'
        )
        self.write_page(
            "julia/index.html",
            "Scorio.jl",
            '<script src="siteinfo.js"></script><script src="../versions.js"></script>',
        )
        self.write_page("julia/api/eval/index.html", "Evaluation API · Scorio.jl")
        self.write_page("javascript/index.html", "Scorio - v0.2.3")
        for module in ("eval", "rank"):
            self.write_page(
                f"javascript/functions/{module}.avg.html", "avg | Scorio - v0.2.3"
            )
        for relative in ("404.html", "poster/index.html", "tts/index.html"):
            self.write_page(
                relative, "Legacy page", head='<meta name="robots" content="noindex">'
            )

    def write_page(self, relative, title, body="", head=""):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'<!doctype html><html lang="en"><head><title>{title}</title>'
            '<meta name="description" content="Old generic description">'
            f"{head}</head><body><main><h1>{title}</h1>"
            "<p>Estimate model performance from repeated outcomes with uncertainty.</p>"
            f"{body}</main></body></html>",
            encoding="utf-8",
        )

    def build(self):
        finalize_metadata(self.root)
        write_sitemaps(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            validate(self.root)

    def replace(self, relative, before, after):
        path = self.root / relative
        html = path.read_text(encoding="utf-8")
        self.assertIn(before, html)
        path.write_text(html.replace(before, after), encoding="utf-8")

    def test_every_content_page_has_matching_canonical_and_sitemap_entry(self):
        self.build()
        sitemap = ET.parse(self.root / "sitemap.xml")
        locations = {node.text for node in sitemap.findall(f".//{{{SITEMAP_NS}}}loc")}
        self.assertEqual(len(locations), 7)
        self.assertIn(BASE_URL + "julia/api/eval/", locations)
        self.assertIn(BASE_URL + "javascript/functions/eval.avg.html", locations)
        self.assertNotIn(BASE_URL + "poster/", locations)
        self.assertNotIn(BASE_URL + "tts/", locations)
        self.assertNotIn(BASE_URL + "404.html", locations)
        self.assertNotIn("lastmod", (self.root / "sitemap.xml").read_text())
        for relative in (
            "javascript/functions/eval.avg.html",
            "julia/api/eval/index.html",
        ):
            page = Page((self.root / relative).read_text())
            self.assertEqual(page.canonicals, [canonical_url(relative)])
        index = Page((self.root / "sitemap.html").read_text())
        self.assertIn(BASE_URL + "javascript/functions/eval.avg.html", index.links)

    def test_api_namespaces_have_distinct_titles_and_descriptions(self):
        self.build()
        pages = [
            Page((self.root / f"javascript/functions/{module}.avg.html").read_text())
            for module in ("eval", "rank")
        ]
        self.assertNotEqual(pages[0].titles, pages[1].titles)
        self.assertNotEqual(pages[0].descriptions, pages[1].descriptions)
        self.assertIn("repeated outcomes", pages[0].descriptions[0])

    def test_legacy_redirects_are_preserved_and_build_is_repeatable(self):
        self.replace(
            "poster/index.html",
            "</head>",
            '<meta http-equiv="refresh" content="0;url=https://example.com/poster">'
            '<link rel="canonical" href="https://example.com/poster"></head>',
        )
        original = (self.root / "poster/index.html").read_bytes()
        self.build()
        output = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.build()
        self.assertEqual(output, {p: p.read_bytes() for p in output})
        self.assertEqual(original, (self.root / "poster/index.html").read_bytes())

    def test_search_console_verification_and_structured_data_are_preserved(self):
        verification = (
            '<meta name="google-site-verification" content="test-verification">'
        )
        structured = '<script type="application/ld+json">{"name":"Scorio"}</script>'
        self.replace("index.html", "</head>", verification + structured + "</head>")
        self.build()
        html = (self.root / "index.html").read_text()
        self.assertIn(verification, html)
        self.assertIn(structured, html)

    def test_accidental_noindex_and_redirects_fail_the_build(self):
        for restriction in (
            '<meta name="robots" content="noindex, follow">',
            '<meta name="googlebot" content="none">',
            '<meta name="robots" content="nofollow">',
            '<meta http-equiv="refresh" content="0;url=https://example.com">',
        ):
            with self.subTest(restriction=restriction):
                self.write_page("julia/index.html", "Scorio.jl", head=restriction)
                with self.assertRaisesRegex(ValueError, "indexing restriction"):
                    finalize_metadata(self.root)

    def test_broken_relative_assets_fail_validation(self):
        self.build()
        self.replace(
            "javascript/functions/eval.avg.html",
            "</main>",
            '<script src="../assets/missing.js"></script></main>',
        )
        with self.assertRaisesRegex(ValueError, "broken local link or resource"):
            validate(self.root)

    def test_index_must_be_discoverable_from_the_landing_page(self):
        self.build()
        self.replace("index.html", '<a href="sitemap.html">Documentation index</a>', "")
        with self.assertRaisesRegex(ValueError, "unreachable from homepage"):
            validate(self.root)

    def test_missing_documentation_and_stale_sitemaps_fail_validation(self):
        self.build()
        path = self.root / "sitemap.xml"
        original = path.read_text()
        path.write_text(original.replace(BASE_URL + "julia/", BASE_URL + "outdated/"))
        with self.assertRaisesRegex(ValueError, "Sitemap must list"):
            validate(self.root)
        path.write_text(original)
        (self.root / "javascript/index.html").unlink()
        with self.assertRaisesRegex(ValueError, "Missing required page"):
            validate(self.root)

    def test_conflicting_canonical_fails_read_only_check(self):
        self.build()
        self.replace(
            "javascript/functions/eval.avg.html",
            "</head>",
            f'<link rel="canonical" href="{BASE_URL}"></head>',
        )
        with self.assertRaisesRegex(ValueError, "canonical must be exactly"):
            validate(self.root)

    def test_empty_static_content_fails_validation(self):
        self.build()
        path = self.root / "javascript/index.html"
        html = path.read_text()
        head = html.split("</head>")[0]
        path.write_text(head + "</head><body><main></main></body></html>")
        with self.assertRaisesRegex(ValueError, "missing static main content"):
            validate(self.root)


if __name__ == "__main__":
    unittest.main()
