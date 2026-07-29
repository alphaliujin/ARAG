"""验证 PDF element indexes 修复 (H6).

检查: 每个 element 的字符区间 [s,e] 必须落在其 element_pages 标注的那一页
在最终 markdown 中的区域内 (即落在 <!-- Page N --> 与 <!-- Page N+1 --> 之间)。
旧实现 page2+ 元素指向 page1 偏移, 此检查会大量失败。
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "X2MD" / "src"))

from x2md.converters.pdf import PdfConverter

PDF = Path(sys.argv[1])
conv = PdfConverter()

t0 = time.time()
data = conv.convert_with_elements(PDF, extract_tables=True)
elapsed = time.time() - t0

text = data["text"]
indexes = data["element_indexes"]
pages = data["element_pages"]
types = data["element_types"]

print(f"file: {PDF.name}")
print(f"convert elapsed: {elapsed:.2f}s")
print(f"text length: {len(text)} chars")
print(f"num elements: {len(indexes)}  (text={types.count('text')}, Table={types.count('Table')})")

# 1) bounds + 非空 + 有序
oob = zero = 0
prev_start = -1
unsorted = 0
for s, e in indexes:
    if not (0 <= s <= e <= len(text)):
        oob += 1
    if e <= s:
        zero += 1
    if s < prev_start:
        unsorted += 1
    prev_start = s
print(f"[bounds] out-of-range: {oob}, zero-width: {zero}, not-sorted-by-start: {unsorted}")

# 2) 页归属: element 的 [s,e] 必须落在其页标记区间内
page_markers = [(m.start(), int(m.group(1))) for m in re.finditer(r"<!-- Page (\d+) -->", text)]
page_spans = {}
for j, (pos, pageno) in enumerate(page_markers):
    end = page_markers[j + 1][0] if j + 1 < len(page_markers) else len(text)
    page_spans[pageno] = (pos, end)
print(f"pages in markdown: {sorted(page_spans)}")

wrong_page = 0
samples = []
for i, (idx, pg) in enumerate(zip(indexes, pages)):
    s, e = idx
    span = page_spans.get(pg)
    if span is None:
        wrong_page += 1
        continue
    ps, pe = span
    if not (ps <= s and e <= pe):
        wrong_page += 1
        if len(samples) < 8:
            samples.append((i, pg, idx, span, repr(text[s:e][:40])))
print(f"[page-attribution] elements outside their page region: {wrong_page}/{len(indexes)}")
for smp in samples:
    print(f"    elem {smp[0]} page={smp[1]} idx={smp[2]} page_span={smp[3]} slice={smp[4]}")

# 3) 抽样: 第 2 页 (含) 之后的若干 element 的实际文本
later = [(i, idx, pages[i], types[i]) for i in range(len(indexes)) if pages[i] >= 2][:5]
print("[sample] first 5 elements on page>=2:")
for i, idx, pg, t in later:
    s, e = idx
    print(f"    elem {i} page={pg} type={t} text={text[s:e][:50]!r}")

# 4) 覆盖率: element 区间并集占 text 的比例 (排除 <!-- Page N --> 和 --- 分隔符)
covered = 0
for s, e in indexes:
    if e > s:
        covered += e - s
structural = sum(len(m.group(0)) for m in re.finditer(r"<!-- Page \d+ -->\n|---", text))
print(f"[coverage] element chars={covered}, text={len(text)}, structural~={structural}, "
      f"element ratio={covered/max(len(text),1):.1%}")
