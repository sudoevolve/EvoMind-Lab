 from __future__ import annotations
 
 
 def ascii_sparkline(values: list[float]) -> str:
     if not values:
         return ""
     blocks = "▁▂▃▄▅▆▇█"
     vmin = min(values)
     vmax = max(values)
     span = vmax - vmin
     if span <= 1e-12:
         return blocks[0] * len(values)
     out = []
     for v in values:
         t = (v - vmin) / span
         idx = int(round(t * (len(blocks) - 1)))
         idx = max(0, min(len(blocks) - 1, idx))
         out.append(blocks[idx])
     return "".join(out)
