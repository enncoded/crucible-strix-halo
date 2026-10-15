#!/usr/bin/env python3
"""Build the graded image question set: screenshots, charts, and code photos with an answer key.

  python bench/vision_graded/make_set.py OUTDIR

Needs Pillow and matplotlib. Images are deterministic. cases.json holds the question, the accepted
answers, and the category. An answer is a string (a case-insensitive substring of the reply) or a list
of strings that must all occur.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = ["/usr/share/fonts/TTF/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/usr/share/fonts/dejavu/DejaVuSansMono.ttf"]
SANS = ["/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf"]


def font(paths, size):
    import matplotlib  # bundles DejaVu, so the set does not depend on system fonts
    bundled = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for path in [os.path.join(bundled, os.path.basename(p)) for p in paths[:1]] + paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise SystemExit("DejaVu fonts are required for a reproducible set")


MONO, MONO_S, SANS_M, SANS_L = font(FONTS, 18), font(FONTS, 15), font(SANS, 20), font(SANS, 28)


def save(im, out, name):
    im.save(os.path.join(out, name + ".png"))
    return name + ".png"


def terminal(out):
    lines = ["$ pytest -q tests/", "........F.....", "FAILED tests/test_cache.py::test_evict_lru - AssertionError",
             "  assert cache.size() == 3", "  where 4 = <Cache>.size()", "1 failed, 13 passed in 2.41s", "$ "]
    im = Image.new("RGB", (900, 260), (20, 20, 20))
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        color = (230, 80, 80) if line.startswith("FAILED") else (210, 210, 210)
        d.text((16, 14 + 32 * i), line, fill=color, font=MONO)
    return save(im, out, "terminal")


def error_dialog(out):
    im = Image.new("RGB", (1024, 640), (90, 110, 140))
    d = ImageDraw.Draw(im)
    d.rectangle([212, 170, 812, 470], fill=(250, 250, 250), outline=(60, 60, 60), width=2)
    d.rectangle([212, 170, 812, 215], fill=(200, 40, 40))
    d.text((230, 180), "Disk full", fill="white", font=SANS_M)
    d.text((240, 250), "Could not save report.pdf:", fill=(20, 20, 20), font=SANS_M)
    d.text((240, 285), "only 12 MB free on drive D:", fill=(20, 20, 20), font=SANS_M)
    for x, label in ((480, "Retry"), (640, "Cancel")):
        d.rectangle([x, 400, x + 130, 445], fill=(225, 225, 225), outline=(90, 90, 90))
        d.text((x + 30, 410), label, fill=(0, 0, 0), font=SANS_M)
    return save(im, out, "dialog")


def file_tree(out):
    rows = ["crucible/", "  engine/", "    engine.hip", "    vision.h", "  server/", "    api.py", "    vision.py",
            "  tests/", "    multimodal_test.cpp", "  README.md"]
    im = Image.new("RGB", (520, 380), (37, 37, 38))
    d = ImageDraw.Draw(im)
    for i, row in enumerate(rows):
        d.text((14, 12 + 34 * i), row, fill=(204, 204, 204), font=MONO)
    return save(im, out, "tree")


def table_page(out):
    im = Image.new("RGB", (1100, 520), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.text((30, 20), "Quarterly sales (units)", fill=(0, 0, 0), font=SANS_L)
    head = ["Region", "Q1", "Q2", "Q3", "Q4"]
    rows = [["North", 120, 135, 150, 160], ["South", 98, 87, 110, 140], ["East", 200, 180, 175, 190],
            ["West", 75, 90, 95, 310]]
    for r, row in enumerate([head] + rows):
        for c, cell in enumerate(row):
            x, y = 40 + c * 200, 90 + r * 70
            d.rectangle([x, y, x + 200, y + 70], outline=(150, 150, 150), fill=(230, 236, 245) if r == 0 else None)
            d.text((x + 16, y + 22), str(cell), fill=(0, 0, 0), font=SANS_M)
    return save(im, out, "table")


def chart_bar(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    names, values = ["Mon", "Tue", "Wed", "Thu", "Fri"], [14, 22, 9, 31, 18]
    ax.bar(names, values, color="#4472c4")
    ax.set_title("Pull requests merged per day")
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, str(v), ha="center")
    path = os.path.join(out, "chart_bar.png")
    fig.savefig(path)
    plt.close(fig)
    return "chart_bar.png"


def chart_line(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    ax.plot(months, [210, 230, 260, 240, 300, 340], marker="o", label="latency p99 (ms)")
    ax.plot(months, [120, 118, 121, 119, 125, 122], marker="s", label="latency p50 (ms)")
    ax.legend()
    ax.set_title("API latency")
    fig.savefig(os.path.join(out, "chart_line.png"))
    plt.close(fig)
    return "chart_line.png"


def chart_pie(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.pie([45, 25, 20, 10], labels=["GPU", "CPU", "Memory", "Disk"], autopct="%d%%")
    ax.set_title("Cost share")
    fig.savefig(os.path.join(out, "chart_pie.png"))
    plt.close(fig)
    return "chart_pie.png"


def code_photo(out, name, code, angle, seed):
    """Code on a light 'paper', rotated, blurred and noisy, like a phone photo of a screen."""
    lines = code.split("\n")
    im = Image.new("RGB", (980, 60 + 30 * len(lines)), (250, 248, 240))
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        d.text((30, 30 + 30 * i), line, fill=(25, 25, 25), font=MONO)
    im = im.rotate(angle, expand=True, fillcolor=(120, 110, 100), resample=Image.BICUBIC)
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    noise = np.random.default_rng(seed).normal(0, 6, (im.height, im.width, 3))
    im = Image.fromarray(np.clip(np.asarray(im, dtype=np.float64) + noise, 0, 255).astype(np.uint8))
    im.save(os.path.join(out, name + ".jpg"), quality=85)
    return name + ".jpg"


CODE_SUM = '''def total_price(items, tax=0.08):
    subtotal = 0
    for item in items:
        subtotal += item.price * item.quantity
    return round(subtotal * (1 + tax), 2)'''

CODE_BUG = '''function average(values) {
  let sum = 0;
  for (let i = 0; i <= values.length; i++) {
    sum += values[i];
  }
  return sum / values.length;
}'''

CODE_RUST = '''fn parse_port(text: &str) -> Result<u16, String> {
    let port: u16 = text.trim().parse()
        .map_err(|e| format!("bad port: {e}"))?;
    if port < 1024 {
        return Err("port must be at least 1024".into());
    }
    Ok(port)
}'''


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    c = []

    def add(image, question, answers, category):
        c.append(dict(name=f"q{len(c) + 1:02d}", image=image, question=question, answers=answers, category=category))

    t = terminal(out)
    add(t, "Which test failed? Give the test function name only.", ["test_evict_lru"], "screenshot")
    add(t, "How many tests passed?", ["13"], "screenshot")
    dlg = error_dialog(out)
    add(dlg, "What is the title of the dialog?", ["disk full"], "screenshot")
    add(dlg, "How much free space does the dialog report?", ["12 mb", "12mb"], "screenshot")
    add(dlg, "Which two buttons does the dialog show?", [["retry", "cancel"]], "screenshot")
    tree = file_tree(out)
    add(tree, "Which folder contains vision.py?", ["server"], "screenshot")
    add(tree, "Name the file in the tests folder.", ["multimodal_test.cpp"], "screenshot")
    tab = table_page(out)
    add(tab, "Which region sold the most units in Q4, and how many?", [["west", "310"]], "screenshot")
    add(tab, "What were the South region's Q2 sales?", ["87"], "screenshot")
    bar = chart_bar(out)
    add(bar, "On which day were the most pull requests merged?", ["thu"], "chart")
    add(bar, "How many pull requests were merged on Wednesday?", ["9"], "chart")
    line = chart_line(out)
    add(line, "In which month is the p99 latency highest?", ["jun"], "chart")
    add(line, "Does the p50 latency stay roughly flat or rise sharply? Answer flat or rise.", ["flat"], "chart")
    pie = chart_pie(out)
    add(pie, "Which category has the largest cost share, and what percent?", [["gpu", "45"]], "chart")
    add(pie, "What percent is Disk?", ["10"], "chart")
    p1 = code_photo(out, "code_sum", CODE_SUM, 3, 1)
    add(p1, "What is the default tax rate in this function? Answer with the number.", ["0.08"], "code")
    add(p1, "To how many decimal places does the function round?", ["2", "two"], "code")
    p2 = code_photo(out, "code_bug", CODE_BUG, -2, 2)
    add(p2, "The loop condition has an off-by-one bug. Reply with the corrected loop condition only.",
        ["i < values.length"], "code")
    p3 = code_photo(out, "code_rust", CODE_RUST, 1.5, 3)
    add(p3, "What is the smallest port this function accepts?", ["1024"], "code")
    add(p3, "What is the function's name?", ["parse_port"], "code")
    json.dump(c, open(os.path.join(out, "cases.json"), "w"), indent=1)
    print(len(c), "cases")


if __name__ == "__main__":
    main()
