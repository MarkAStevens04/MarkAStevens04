#!/usr/bin/env python3
"""Build the Experience Timeline from projects.json (+ skills.json).

Produces a self-contained, static SVG -- pure black, white spine, small red/blue
project dots, title + date + description per entry, and tech chips outlined in
per-skill colors -- at assets/projects-timeline.svg, then rewrites the block
between

    <!-- PROJECTS:START -->
    <!-- PROJECTS:END -->

in README.md to embed it. Newest project sits at the top.

Run locally:  python tools/build_readme.py
In CI:        .github/workflows/readme.yml runs this on pushes to projects.json / skills.json

Pure standard library -- no third-party dependencies.
"""
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "projects.json"
SKILLS = ROOT / "skills.json"
README = ROOT / "README.md"
SVG_OUT = ROOT / "assets" / "projects-timeline.svg"
START = "<!-- PROJECTS:START -->"
END = "<!-- PROJECTS:END -->"

# ---- palette (white / red / blue only) ------------------------------------
WHITE = "#ffffff"
RED = "#ff5757"
BLUE = "#5170ff"
ACCENTS = {"red": RED, "blue": BLUE}

# Same font as the MARK STEVENS banner (assets/name-banner.svg)
FONT = "'Segoe UI', 'Segoe UI Bold', Tahoma, Geneva, Verdana, sans-serif"

# ---- canvas / layout (px) -------------------------------------------------
W = 860
BG_RX = 18

SVG_TOP_PAD = 26
BIG_FS = 30             # "Experience Timeline" header
GAP_AFTER_HEADER = 22

SPINE_X = 46
NODE_R = 7              # project dots
SPINE_W = 2.5           # solid white vertical line
SPINE_OPACITY = 1.0
SPINE_OVERHANG = 18     # how far the line pokes above the first dot (bottom stops at last section)

CONTENT_X = 66          # text column; kept close to the spine
CONTENT_R = W - 48
CONTENT_W = CONTENT_R - CONTENT_X

PAD_TOP = 14            # space above each project title (individual entry sizing)
TITLE_FS = 19           # bold white
DATE_FS = 11.5          # not bold, strongly de-emphasized
DATE_OPACITY = 0.35
DATE_LH = 13            # title baseline -> date baseline (tight)
DESC_FS = 14            # not bold, white (between title and date sizes)
DESC_OPACITY = 0.70     # between the date opacity and full white
DESC_LH = 20
GAP_DATE_DESC = 3
GAP_DESC_CHIPS = 12
CHIP_FS = 11
CHIP_H = 19
CHIP_PAD = 5            # horizontal padding inside a chip (snug to the text)
CHIP_GAP = 7
CHIP_ROW_GAP = 7
PAD_BOTTOM = 36         # space BETWEEN experiences (entry sizing itself unchanged)
BOTTOM_MARGIN = 22      # black canvas padding below the final section


def esc(s):
    return html.escape(str(s), quote=True)


# --- proportional text width estimate (the banner font is sans-serif, not mono) ---
# Per-character advance as a fraction of font size, tuned slightly generous so chip
# borders never clip and descriptions never overflow. Exact metrics vary by viewer.
_CHAR_W = {}
for _c in "ijl.,:;|!'`": _CHAR_W[_c] = 0.27
for _c in "ftI()[]{}/\\ -": _CHAR_W[_c] = 0.34
for _c in "r": _CHAR_W[_c] = 0.40
for _c in "mw": _CHAR_W[_c] = 0.82
for _c in "MW": _CHAR_W[_c] = 0.88
for _c in "0123456789": _CHAR_W[_c] = 0.56


def _char_w(c):
    if c in _CHAR_W:
        return _CHAR_W[c]
    if c.isupper():
        return 0.66
    return 0.52


def text_w(s, fs):
    return sum(_char_w(c) for c in str(s)) * fs


def wrap(text, fs, max_w):
    out, cur = [], ""
    for word in str(text).split():
        cand = f"{cur} {word}".strip()
        if not cur or text_w(cand, fs) <= max_w:
            cur = cand
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out or [""]


def layout_chips(tags):
    """Pack chips into rows that fit CONTENT_W. Each row: list of (rel_x, width, label)."""
    rows, cur, x = [], [], 0.0
    for t in tags:
        w = text_w(t, CHIP_FS) + CHIP_PAD * 2
        if cur and x + w > CONTENT_W:
            rows.append(cur)
            cur, x = [], 0.0
        cur.append((x, w, t))
        x += w + CHIP_GAP
    if cur:
        rows.append(cur)
    return rows


def load_skill_colors():
    """name -> outline color, with a default fallback. Case-insensitive lookup."""
    if not SKILLS.exists():
        return {}, WHITE
    cfg = json.loads(SKILLS.read_text(encoding="utf-8"))
    colors = {k.lower(): v for k, v in (cfg.get("colors") or {}).items()}
    return colors, cfg.get("default", WHITE)


def date_label(p):
    start = str(p.get("start", "")).strip()
    end = str(p.get("end", "")).strip()
    if not start:
        return ""
    return f"{start} – {end or 'Present'}"   # en dash


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def date_key(s):
    """Sortable (year, month) from '2024', '2024-06', or 'Jun 2024' / 'June 2024'."""
    s = str(s).strip()
    m = re.match(r"(\d{4})-(\d{1,2})", s)            # 2024-06
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.match(r"([A-Za-z]+)\s+(\d{4})", s)        # Jun 2024 / June 2024
    if m:
        return (int(m.group(2)), _MONTHS.get(m.group(1)[:3].lower(), 0))
    m = re.match(r"(\d{4})", s)                       # 2024
    if m:
        return (int(m.group(1)), 0)
    return (0, 0)


def load_projects():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    items = [p for p in data.get("projects", []) if p.get("title")]
    items.sort(key=lambda p: date_key(p.get("start", "")), reverse=True)   # newest first
    return items


def empty_svg():
    h = 130
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" '
        f'height="{h}" font-family="{FONT}">'
        f'<rect width="{W}" height="{h}" rx="{BG_RX}" fill="#000000"/>'
        f'<text x="{SVG_TOP_PAD+8}" y="{SVG_TOP_PAD+BIG_FS}" font-size="{BIG_FS}" '
        f'font-weight="700" fill="{WHITE}">Experience Timeline</text>'
        f'<text x="{SVG_TOP_PAD+8}" y="{h-22}" font-size="13" fill="{WHITE}" '
        f'fill-opacity="0.6">No projects yet — add some to projects.json</text></svg>'
    )


def build_svg(projects):
    if not projects:
        return empty_svg()

    skill_colors, default_col = load_skill_colors()

    def chip_color(tag):
        return skill_colors.get(str(tag).lower(), default_col)

    body = []
    node_cys = []
    node_cols = []
    y = SVG_TOP_PAD + BIG_FS + GAP_AFTER_HEADER

    for i, p in enumerate(projects):
        title = esc(p.get("title", "Untitled"))
        accent = ACCENTS.get(str(p.get("accent", "")).lower(), (RED, BLUE)[i % 2])
        desc_lines = wrap(p.get("description", ""), DESC_FS, CONTENT_W)
        chip_rows = layout_chips(p.get("tech") or [])
        link = p.get("repo") or p.get("link")
        dlabel = esc(date_label(p))

        block_top = y
        ty = block_top + PAD_TOP + TITLE_FS          # title baseline
        node_cy = ty - TITLE_FS * 0.34
        node_cys.append(node_cy)
        node_cols.append(accent)

        # title (bold white), optionally linked
        title_el = (
            f'<text x="{CONTENT_X}" y="{ty:.1f}" font-size="{TITLE_FS}" font-weight="700" '
            f'fill="{WHITE}">{title}</text>'
        )
        body.append(f'<a href="{esc(link)}">{title_el}</a>' if link else title_el)

        # date (not bold, white @ 60%)
        cursor = ty
        if dlabel:
            cursor += DATE_LH
            body.append(
                f'<text x="{CONTENT_X}" y="{cursor:.1f}" font-size="{DATE_FS}" '
                f'fill="{WHITE}" fill-opacity="{DATE_OPACITY}">{dlabel}</text>'
            )

        # description (not bold, white, mid size)
        cursor += GAP_DATE_DESC
        for line in desc_lines:
            cursor += DESC_LH
            body.append(
                f'<text x="{CONTENT_X}" y="{cursor:.1f}" font-size="{DESC_FS}" '
                f'fill="{WHITE}" fill-opacity="{DESC_OPACITY}">{esc(line)}</text>'
            )

        # tech chips: transparent fill, colored outline, white text
        chips_top = cursor + GAP_DESC_CHIPS
        for r, row in enumerate(chip_rows):
            ry = chips_top + r * (CHIP_H + CHIP_ROW_GAP)
            for rel_x, cw, label in row:
                cx = CONTENT_X + rel_x
                col = chip_color(label)
                body.append(
                    f'<rect x="{cx:.1f}" y="{ry:.1f}" width="{cw:.1f}" height="{CHIP_H}" '
                    f'rx="{CHIP_H/2:.0f}" fill="none" stroke="{col}"/>'
                    f'<text x="{cx + cw/2:.1f}" y="{ry + CHIP_H*0.68:.1f}" text-anchor="middle" '
                    f'font-size="{CHIP_FS}" fill="{WHITE}">{esc(label)}</text>'
                )
        chips_bottom = (
            chips_top + len(chip_rows) * CHIP_H + (len(chip_rows) - 1) * CHIP_ROW_GAP
            if chip_rows else cursor
        )
        y = chips_bottom + PAD_BOTTOM

    section_bottom = chips_bottom        # bottom of the final entry (loop's last value)
    H = int(section_bottom + BOTTOM_MARGIN)
    cy0 = node_cys[0]

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="{FONT}">',
        f'<rect width="{W}" height="{H}" rx="{BG_RX}" fill="#000000"/>',
        # big header, left edge flush with the spine
        f'<text x="{SPINE_X-2}" y="{SVG_TOP_PAD+BIG_FS:.0f}" font-size="{BIG_FS}" '
        f'font-weight="700" fill="{WHITE}" letter-spacing="0.5">Experience Timeline</text>',
        # solid white spine: small overhang above the first dot, stops at the last section
        f'<line x1="{SPINE_X}" y1="{cy0-SPINE_OVERHANG:.1f}" x2="{SPINE_X}" y2="{section_bottom:.1f}" '
        f'stroke="{WHITE}" stroke-opacity="{SPINE_OPACITY}" stroke-width="{SPINE_W}"/>',
    ]
    out += body
    # nodes last, on top of the spine
    for cy, col in zip(node_cys, node_cols):
        out.append(f'<circle cx="{SPINE_X}" cy="{cy:.1f}" r="{NODE_R}" fill="{col}"/>')
    out.append('</svg>')
    return "\n".join(out) + "\n"


def build_block(projects):
    parts = []
    for p in projects:
        d = date_label(p)
        parts.append(f'{p.get("title","")} ({d})' if d else p.get("title", ""))
    names = "; ".join(parts)
    alt = esc(f"Experience timeline: {names}" if names else "Experience timeline")
    return (
        '<p align="center">\n'
        f'  <img src="./assets/projects-timeline.svg" width="840" alt="{alt}" />\n'
        '</p>'
    )


def main():
    projects = load_projects()

    SVG_OUT.parent.mkdir(parents=True, exist_ok=True)
    svg = build_svg(projects)
    svg_changed = (not SVG_OUT.exists()) or SVG_OUT.read_text(encoding="utf-8") != svg
    if svg_changed:
        SVG_OUT.write_text(svg, encoding="utf-8")

    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print(f"ERROR: markers {START} / {END} not found in README.md", file=sys.stderr)
        return 1
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    updated = pattern.sub(f"{START}\n{build_block(projects)}\n{END}", text)
    readme_changed = updated != text
    if readme_changed:
        README.write_text(updated, encoding="utf-8")

    print(f"projects: {len(projects)}  svg: {'updated' if svg_changed else 'unchanged'}  "
          f"readme: {'updated' if readme_changed else 'unchanged'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
