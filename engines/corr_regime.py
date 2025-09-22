# ================= Correlation Regime Monitor (with Weights) =================
# - รองรับน้ำหนักต่อสินทรัพย์ (weights_pct)
# - วัดภาวะคอเรลเลชัน: mean off-diag corr, weighted mean corr, eigen-share
# - หา "คู่คอเรลสูง" พร้อมน้ำหนักรวม และ pair variance share (ต่อ variance พอร์ต)
# - แนะนำ proxy สำหรับเฮดจ์ และ (ถ้าโหลดได้) คำนวณขนาดเฮดจ์ประมาณการด้วย beta
# ============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
from itertools import combinations
from datetime import datetime
from zoneinfo import ZoneInfo

# --------- INPUT ---------
tickers = ["STANLY.BK","NVDA","DDOG","JEPQ","UNH","RBLX","ABBV",
           "INTC","META","GLD","UBER","SMR","BTC-USD"]

# ถ้ามีน้ำหนัก ใส่ตรงนี้ (รวม ~100). ถ้าไม่ใส่ จะเท่ากันทุกตัว
weights_pct = [23.35,9.27,11.24,6.95,7.26,3.45,7.94,
               5.87,7.55,11.34,2.79,2.29,0.70]

LOOKBACK = "2y"
INTERVAL = "1d"
WINDOW = 60                  # วันทำการสำหรับ rolling corr
THRESH = 0.70                # เกณฑ์คู่ที่ corr สูง
CALM_THR = 0.25              # mean corr < 0.25 -> CALM
STRESS_THR = 0.50            # >0.50 หรือ eigen-share > 0.40 -> STRESS
EIG_THR = 0.40
DO_HEDGE_BETA = True         # พยายามคำนวณ beta -> แนะนำขนาดเฮดจ์ (%NAV)

# --------- Hedge proxy (ตัวอย่างที่เรียกจาก Yahoo ได้) ---------
# หมายเหตุ: เลือก proxy จาก mapping นี้เพื่อไปดึงข้อมูล
hedge_yf_map = {
    "NVDA": ["QQQ","SMH","SOXX"],
    "DDOG": ["IGV","QQQ"],
    "META": ["XLC","QQQ"],
    "UBER": ["XLY","QQQ"],
    "INTC": ["SMH","SOXX","QQQ"],
    "JEPQ": ["QQQ"],
    "UNH":  ["XLV"],
    "ABBV": ["XLV"],
    "GLD":  ["GLD"],                 # เฮดจ์ = short GLD
    "SMR":  ["URA","XLU"],           # ใกล้เคียง (นิวเคลียร์/สาธารณูปโภค)
    "STANLY.BK": ["THD"],            # iShares MSCI Thailand
    "BTC-USD": ["BTC-USD"],          # เฮดจ์ = short BTC
}

tech_like = {"NVDA","DDOG","META","UBER","INTC","JEPQ"}

def choose_proxy_for_pair(a, b):
    A = set(hedge_yf_map.get(a, []))
    B = set(hedge_yf_map.get(b, []))
    inter = list(A.intersection(B))
    if inter:
        return inter[0]  # เลือกตัวแรก
    if a in tech_like and b in tech_like:
        return "QQQ"
    if a == "GLD" or b == "GLD":
        return "GLD"
    if a == "BTC-USD" or b == "BTC-USD":
        return "BTC-USD"
    if a == "STANLY.BK" or b == "STANLY.BK":
        return "THD"
    return None  # ไม่มี proxy ตรง ๆ

# --------- เตรียมน้ำหนัก ---------
n = len(tickers)
if weights_pct and len(weights_pct) == n:
    w = np.array(weights_pct, dtype=float)
    w = w / w.sum()  # normalize เผื่อรวมไม่เท่ากับ 100 เป๊ะ
else:
    w = np.ones(n) / n

w_series = pd.Series(w, index=tickers)

# --------- ดาวน์โหลดราคาและผลตอบแทน ---------
px = yf.download(tickers, period=LOOKBACK, interval=INTERVAL,
                 auto_adjust=True, progress=False)["Close"]
px = px.dropna(how="all").ffill().dropna()
ret = np.log(px/px.shift(1)).dropna()

# --------- คำนวณบน rolling WINDOW ล่าสุด ---------
sub = ret.iloc[-WINDOW:]
C = sub.corr()
Cov = sub.cov()
C_values = C.values

# mean off-diagonal corr (ไม่ถ่วงน้ำหนัก)
mean_offdiag = (C_values.sum() - np.trace(C_values)) / (C_values.size - n)

# weighted mean off-diagonal corr: sum_{i<j} w_i w_j corr_ij / sum_{i<j} w_i w_j
wij = []
cij = []
for i in range(n):
    for j in range(i+1, n):
        wij.append(w[i]*w[j])
        cij.append(C_values[i, j])
wij = np.array(wij)
cij = np.array(cij)
weighted_mean_corr = float((wij * cij).sum() / wij.sum())

# eigen-share (λ1/sum λ) ของเมทริกซ์ correlation
eigvals = np.linalg.eigvalsh(C_values)
eigen_share = float(eigvals[-1] / eigvals.sum())

# regime label
if (mean_offdiag > STRESS_THR) or (eigen_share > EIG_THR):
    regime = "STRESS"
elif mean_offdiag > CALM_THR:
    regime = "NORMAL"
else:
    regime = "CALM"

# --------- พอร์ต variance (ใช้สำหรับ pair variance share) ---------
W = w.reshape(-1,1)
port_var = float(W.T @ Cov.values @ W)  # variance (ต่อวัน) ของพอร์ตจาก WINDOW นี้

def pair_variance_share(i, j):
    # สัดส่วนของความแปรปรวนพอร์ตจากคู่ (i,j): 2 w_i w_j Cov_ij / port_var
    if port_var <= 0:
        return np.nan
    return float(2 * w[i] * w[j] * Cov.values[i, j] / port_var)

# --------- ค้นหาคู่คอเรลสูงและเตรียมเฮดจ์ ---------
pairs = []
need_proxy = set()
for a, b in combinations(C.columns, 2):
    i, j = tickers.index(a), tickers.index(b)
    corr_ij = float(C.loc[a, b])
    if not np.isfinite(corr_ij) or corr_ij < THRESH:
        continue
    wsum = 100.0 * (w[i] + w[j])  # น้ำหนักรวมของคู่ (%NAV)
    pv_share = pair_variance_share(i, j)  # อาจเป็นลบได้ถ้า cov ติดลบ แต่เราคัดคู่ corr สูง (บวก)
    proxy = choose_proxy_for_pair(a, b)
    if proxy:
        need_proxy.add(proxy)
    pairs.append([a, b, corr_ij, wsum, pv_share, proxy])

# เรียงคู่ตาม "ผลต่อความเสี่ยงพอร์ต" มากสุดก่อน (pair variance share)
pairs.sort(key=lambda x: (x[4] if x[4] is not None else -np.inf), reverse=True)
top_pairs = pairs[:12]

# --------- ดาวน์โหลด proxy สำหรับคู่ที่จะคำนวณขนาดเฮดจ์ (optional) ---------
beta_info = {}
if DO_HEDGE_BETA and len(need_proxy) > 0:
    prox_px = yf.download(sorted(need_proxy), period=LOOKBACK, interval=INTERVAL,
                          auto_adjust=True, progress=False)["Close"]
    prox_px = prox_px.dropna(how="all").ffill().dropna()
    prox_ret = np.log(prox_px/prox_px.shift(1)).dropna()

    for a, b, corr_ij, wsum, pv_share, proxy in top_pairs:
        if not proxy or proxy not in prox_ret.columns:
            continue
        # returns ช่วงเดียวกัน
        rA = sub[a]
        rB = sub[b]
        rP = prox_ret[proxy].reindex(sub.index).dropna()
        df = pd.concat([rA, rB, rP], axis=1, join="inner").dropna()
        if len(df) < 30:
            continue
        # ทำ "ตะกร้าคู่" ตามน้ำหนัก %NAV ของสองตัว
        wa = w_series[a]; wb = w_series[b]
        if wa + wb <= 0:
            continue
        r_basket = (wa * df[a] + wb * df[b]) / (wa + wb)

        # OLS beta ของ basket ต่อ proxy: r_basket = alpha + beta * r_proxy + eps
        x = df[proxy].values
        y = r_basket.values
        x = np.column_stack([np.ones_like(x), x])   # add intercept
        beta_hat = np.linalg.lstsq(x, y, rcond=None)[0][1]  # ค่าสัมประสิทธิ์ของ r_proxy
        # แนะนำขนาดเฮดจ์ ≈ beta * น้ำหนักตะกร้า (%NAV)
        hedge_size_pctNAV = float(beta_hat * (wa + wb) * 100.0)
        beta_info[(a, b)] = (proxy, beta_hat, hedge_size_pctNAV)

# --------- ทำรายงาน Markdown ---------
def pairs_to_markdown(pairs_list):
    if not pairs_list:
        return "_No pairs above threshold_"
    rows = []
    for a, b, corr_ij, wsum, pv_share, proxy in pairs_list:
        if (a, b) in beta_info:
            px, beta_hat, hedge_pct = beta_info[(a, b)]
            rows.append([a, b, corr_ij, wsum, pv_share, px, beta_hat, hedge_pct])
        else:
            rows.append([a, b, corr_ij, wsum, pv_share, proxy, np.nan, np.nan])

    df = pd.DataFrame(rows, columns=[
        "Asset A","Asset B","Corr(60d)",
        "WeightSum(%)","PairVarShare","Hedge Proxy",
        "Beta (basket→proxy)","Suggested Hedge %NAV"
    ])
    # หมายเหตุ: PairVarShare เป็นสัดส่วนต่อ variance พอร์ต (อาจ >1 ในบางกรณีที่กระจุกมาก/หน้าต่างสั้น)
    return df.to_markdown(index=False,
                          floatfmt=("",".",".3f",".2f",".3f","",".3f",".2f"))

latest_date = ret.index[-1].date()
now_bkk = datetime.now(ZoneInfo("Asia/Bangkok")).strftime("%Y-%m-%d %H:%M")

report = []
report.append(f"# Correlation Regime Report ({latest_date})")
report.append("")
report.append(f"- Generated at (Asia/Bangkok): **{now_bkk}**")
report.append(f"- Window: **{WINDOW} trading days**  |  High-corr threshold: **{THRESH:.2f}**")
report.append(f"- Mean off-diagonal corr: **{mean_offdiag:.2f}**")
report.append(f"- Weighted mean corr: **{weighted_mean_corr:.2f}**")
report.append(f"- Eigen share (λ₁/Σλ): **{eigen_share:.2f}**")
report.append(f"- **Regime: {regime}**")
if regime == "CALM":
    alert = ("GREEN / CALM: กระจายยังทำงาน → คงน้ำหนักตามแผน, เฝ้าดูคลัสเตอร์รายสัปดาห์")
elif regime == "NORMAL":
    alert = ("YELLOW / NORMAL: ความสัมพันธ์เริ่มหนาแน่น → ตั้งเพดานต่อธีม, เตรียมเฮดจ์ sector/index")
else:
    alert = ("RED / STRESS: ตลาดกำลังกระจุก → ลด gross 20–40%, เฮดจ์ด้วยดัชนีกว้าง/sector, เพิ่ม multiplier บน VaR/ES")
report.append(f"- **Alert:** {alert}")
report.append("")
report.append("## Top high-correlation pairs (with weights & hedge ideas)")
report.append(pairs_to_markdown(top_pairs))

md_text = "\n".join(report)

# บันทึกไฟล์รายวัน
date_str = pd.to_datetime(latest_date).strftime("%Y-%m-%d")
with open(f"corr_regime_report_{date_str}.md", "w", encoding="utf-8") as f:
    f.write(md_text)

print(md_text)
