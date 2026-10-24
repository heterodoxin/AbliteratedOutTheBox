#!/usr/bin/env python3
"""Prepare local web assets from the supplied Strata logo masters.

Requires fontTools only while preparing the outlined wordmark. The checked-in
exports have no font or Python dependency at app runtime. Source files stay
untouched; public/brand/source-manifest.json records every derivation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont


APP = Path(__file__).resolve().parent.parent
MASTERS = APP.parent / "Strata-Logo-DesignFiles"
PUBLIC = APP / "public"
BRAND = PUBLIC / "brand"
FONT = PUBLIC / "fonts" / "GTWalsheimPro-Bold.woff2"
SVG_NS = "http://www.w3.org/2000/svg"
INK = {"light": "#221e1f", "dark": "#eef1f3"}
GREEN = {"light": "#10b981", "dark": "#34d399"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(relative: str) -> Path:
    path = MASTERS / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def record(entries: list[dict[str, str]], master: str, output: Path, method: str) -> None:
    entries.append({
        "source": master,
        "source_sha256": digest(source(master)),
        "output": output.relative_to(PUBLIC).as_posix(),
        "output_sha256": digest(output),
        "method": method,
    })


def copy(entries: list[dict[str, str]], master: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source(master), output)
    record(entries, master, output, "unchanged copy")


def art_rectangles(master: str) -> str:
    root = ET.parse(source(master)).getroot()
    rects = root.findall(f"{{{SVG_NS}}}rect")
    if len(rects) != 3:
        raise ValueError(f"Expected the three approved bars in {master}")
    return "\n".join(
        f'<rect x="{r.attrib["x"]}" y="{r.attrib["y"]}" '
        f'width="{r.attrib["width"]}" height="{r.attrib["height"]}" '
        f'rx="{r.attrib["rx"]}" fill="{r.attrib["fill"]}" />'
        for r in rects
    )


def outlined_wordmark(font: TTFont) -> str:
    cmap = font.getBestCmap()
    glyphs = font.getGlyphSet()
    scale = 44 / font["head"].unitsPerEm
    tracking_units = 1.54 / scale
    cursor = 0.0
    paths = []
    letters = "strata"
    names = [cmap[ord(letter)] for letter in letters]
    for index, name in enumerate(names):
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(pen)
        paths.append(f'<path transform="translate({cursor:.4f} 0)" d="{pen.getCommands()}" />')
        cursor += font["hmtx"][name][0] - tracking_units
        if index + 1 < len(names):
            cursor += pair_kerning(font, name, names[index + 1])
    return (f'<g transform="translate(60 37.75) scale({scale:.6f} -{scale:.6f})">'
            + "".join(paths) + "</g>")


def pair_kerning(font: TTFont, left: str, right: str) -> int:
    """Apply the font's OpenType kern feature for this fixed six-letter word."""
    gpos = font["GPOS"].table
    kern_lookups = {
        index
        for feature in gpos.FeatureList.FeatureRecord
        if feature.FeatureTag == "kern"
        for index in feature.Feature.LookupListIndex
    }
    adjustment = 0
    for index in sorted(kern_lookups):
        lookup = gpos.LookupList.Lookup[index]
        if lookup.LookupType != 2:
            continue
        for subtable in lookup.SubTable:
            if left not in subtable.Coverage.glyphs:
                continue
            value = None
            if subtable.Format == 1:
                row = subtable.PairSet[subtable.Coverage.glyphs.index(left)]
                value = next((pair.Value1 for pair in row.PairValueRecord
                              if pair.SecondGlyph == right), None)
                if value is None:
                    continue
            elif subtable.Format == 2:
                first = subtable.ClassDef1.classDefs.get(left, 0)
                second = subtable.ClassDef2.classDefs.get(right, 0)
                value = subtable.Class1Record[first].Class2Record[second].Value1
            adjustment += (value.XAdvance or 0) if value else 0
            break
    return adjustment


def prepare_lockup(entries: list[dict[str, str]], theme: str, font: TTFont) -> None:
    master = f"logo/strata-lockup-{theme}.svg"
    output = BRAND / f"strata-lockup-{theme}.svg"
    output.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="182" height="48" '
        'viewBox="0 0 182 48" role="img">\n'
        f"{art_rectangles(master)}\n"
        f'<g fill="{INK[theme]}">{outlined_wordmark(font)}</g>\n'
        "</svg>\n"
    )
    record(entries, master, output, f"outlined GT Walsheim Pro Bold from {FONT.relative_to(APP)}")


def prepare_favicon(entries: list[dict[str, str]]) -> None:
    master = "web/favicon.svg"
    root = ET.parse(source(master)).getroot()
    rects = root.findall(f"{{{SVG_NS}}}rect")
    if len(rects) != 3:
        raise ValueError("Expected the supplied three-bar favicon")
    output = PUBLIC / "favicon.svg"
    bars = "\n".join(
        f'<rect class="{r.attrib["class"]}" x="{r.attrib["x"]}" '
        f'y="{r.attrib["y"]}" width="{r.attrib["width"]}" '
        f'height="{r.attrib["height"]}" rx="{r.attrib["rx"]}" '
        f'fill="{INK["light"] if r.attrib["class"] == "i" else GREEN["light"]}" />'
        for r in rects
    )
    output.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" '
        'viewBox="3 3 42 42">\n'
        '<style>@media (prefers-color-scheme: dark) {'
        f'.i {{ fill: {INK["dark"]}; }} .a {{ fill: {GREEN["dark"]}; }}'
        '}</style>\n'
        f'{bars}\n</svg>\n'
    )
    record(entries, master, output, "explicit light fills and self-contained dark media style")


def prepare_manifest(entries: list[dict[str, str]]) -> None:
    master = "web/site.webmanifest"
    manifest = json.loads(source(master).read_text())
    manifest.update({"id": "/", "start_url": "/#/chat", "scope": "/"})
    output = PUBLIC / "site.webmanifest"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    record(entries, master, output, "root-local app identity and hash-route launch")


def main() -> None:
    BRAND.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    for theme in ("light", "dark"):
        copy(entries, f"logo/strata-mark-{theme}.svg", BRAND / f"strata-mark-{theme}.svg")
    font = TTFont(FONT)
    for theme in ("light", "dark"):
        prepare_lockup(entries, theme, font)
    prepare_favicon(entries)
    for name in (
        "favicon-16.png", "favicon-32.png", "favicon-48.png",
        "favicon-dark-16.png", "favicon-dark-32.png", "favicon-dark-48.png",
        "apple-touch-icon.png", "icon-192.png", "icon-512.png",
    ):
        copy(entries, f"web/{name}", PUBLIC / name)
    prepare_manifest(entries)
    (BRAND / "source-manifest.json").write_text(json.dumps({
        "source_root": str(MASTERS),
        "preparation": "scripts/prepare-brand-assets.py",
        "font": str(FONT.relative_to(APP)),
        "font_sha256": digest(FONT),
        "note": "Derived SVGs omit source C2PA metadata; original masters remain untouched and each source hash is recorded here.",
        "files": entries,
    }, indent=2) + "\n")
    print(f"Prepared {len(entries)} web assets in {PUBLIC}")


if __name__ == "__main__":
    main()
