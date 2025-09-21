# pip install matplotlib squarify

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence
import math

import matplotlib.pyplot as plt
import squarify
from matplotlib import colors as mcolors
from matplotlib.ticker import PercentFormatter


# ============================ Data Model ============================

@dataclass(frozen=True)
class Holding:
    symbol: str
    allocation_pct: float   # e.g. 9.8  (percent, not fraction)
    total_return_pct: float # e.g. 3.5  (percent, not fraction)
    sector: Optional[str] = None


# ============================ Label Strategy ============================

class LabelStrategy:
    def format(self, h: Holding) -> str: raise NotImplementedError

class SymbolAllocReturnLabel(LabelStrategy):
    def format(self, h: Holding) -> str:
        return f"{h.symbol}\n{h.allocation_pct:.2f}%\nRet: {h.total_return_pct:+.2f}%"


# ============================ Color Strategy ============================

class ColorStrategy:
    def colors(self, items: Sequence[Holding]): raise NotImplementedError

class SymmetricReturnColorStrategy(ColorStrategy):
    """
    ไล่สีตาม Total Return โดยทำสเกลสมมาตรรอบ 0 (แดง=ลบ, เขียว=บวก)
    ใช้คอลอร์แมพ 'RdYlGn' และ TwoSlopeNorm(center=0).
    """
    def __init__(self, cmap_name: str = "RdYlGn", v_abs: Optional[float] = None):
        self.cmap = plt.get_cmap(cmap_name)
        self.v_abs = v_abs  # ถ้าไม่กำหนด จะใช้ max(abs(returns))
    def colors(self, items: Sequence[Holding]):
        vals = [h.total_return_pct for h in items]
        vmax = self.v_abs if self.v_abs is not None else max(abs(min(vals)), abs(max(vals)))
        if math.isclose(vmax, 0.0):
            vmax = 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        return [self.cmap(norm(v)) for v in vals], norm, self.cmap


# ============================ Renderer (Orchestrator) ============================

class HoldingMapRenderer:
    def __init__(
        self,
        color_strategy: ColorStrategy | None = None,
        label_strategy: LabelStrategy | None = None,
        title: str = "Holding Map (Size=Allocation, Color=Total Return)",
    ):
        self.color_strategy = color_strategy or SymmetricReturnColorStrategy()
        self.label_strategy = label_strategy or SymbolAllocReturnLabel()
        self.title = title

    @staticmethod
    def _normalize_sizes(allocations_pct: List[float]) -> List[float]:
        sizes = [max(v, 0.0) for v in allocations_pct]
        s = sum(sizes)
        if s <= 0:
            raise ValueError("Total allocation must be positive.")
        # squarify รับสเกลสัมบูรณ์ใดๆ; ทำให้รวมเป็น 100 เพื่ออ่านง่าย
        return [v * (100.0 / s) for v in sizes]

    def plot(self, holdings: Sequence[Holding], outfile: Optional[str] = None, show: bool = True):
        sizes = self._normalize_sizes([h.allocation_pct for h in holdings])
        labels = [self.label_strategy.format(h) for h in holdings]

        # สีตามผลตอบแทน (สมมาตรรอบ 0)
        col_result = self.color_strategy.colors(holdings)
        # รองรับทั้งกรณีกลยุทธ์คืน (colors, norm, cmap) หรือแค่ colors
        if isinstance(col_result, tuple) and len(col_result) == 3:
            colors, norm, cmap = col_result
        else:
            colors, norm, cmap = col_result, None, None

        fig, ax = plt.subplots(figsize=(11, 7))
        squarify.plot(
            sizes=sizes,
            label=labels,
            color=colors,
            pad=True,
            alpha=0.95,
            text_kwargs={"fontsize": 10},  # ห้ามใส่ ha/va ซ้ำกับ squarify
            ax=ax,
        )
        ax.set_title(self.title, pad=10)
        ax.axis("off")

        # แถบสีอ้างอิง (colorbar) สำหรับ Total Return
        if norm is not None and cmap is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label("Total Return (%)")
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(100))  # เพราะ norm ใช้หน่วย "เปอร์เซ็นต์"
            # PercentFormatter(100) แปลว่า 10 -> 10%

        plt.tight_layout()
        if outfile:
            fig.savefig(outfile, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)


# ============================ Example ============================

if __name__ == "__main__":
    data = [
        ("STANLY", 24.69, -13.51),
        ("NVDA",   9.80,   0.66),
        ("DDOG",  11.88,  -0.84),
        ("JEPQ",   7.35,   3.51),
        ("UNH",    7.67,  10.20),
        ("RBLX",   3.64,   4.21),
        ("ABBV",   8.38,   4.06),
        ("INTC",   6.21,   6.36),
        ("META",   7.98,   3.77),
        ("GOLD",   1.37,   2.72),
        ("GLD",   10.62,   0.43),
        ("UBER",   2.95,   0.13),
        ("SMR",    2.42,   2.88),
    ]
    holdings: List[Holding] = [Holding(*row) for row in data]

    # ใช้โทนแดง-เหลือง-เขียวสมมาตรรอบ 0
    renderer = HoldingMapRenderer(
        color_strategy=SymmetricReturnColorStrategy("RdYlGn"),
        label_strategy=SymbolAllocReturnLabel(),
        title="Holding Map",
    )
    renderer.plot(holdings, outfile="holding_map_total_return.png", show=True)
