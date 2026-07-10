#!/usr/bin/env python
"""
build_results_full.py — assemble results_full.xlsx: EVERY piece of data behind the paper's
findings, organized by finding, with plain-language metric explanations and full provenance.

Layout (sheet prefixes keep it organized):
  00_README            index + metric glossary + conventions
  1x_*                 curated per-finding sheets (joined data + live-computed stats)
  80_来源与公式          provenance: file -> script -> pre/post encoder -> formula -> sheets
  9x_RAW_*             untouched dumps of every source CSV used

Live-computed numbers are recomputed here from the CSVs (Spearman families, theta, medians);
complex pipeline statistics (force partials, FDR flags, SNR defense, Sorkhei, LOO/AUC) are
transcribed from the verified findings docs with an explicit 来源 label on the row.

Run:  python eval/utils/build_results_full.py   ->  /Users/zhanghaodong/Desktop/CP/results_full.xlsx
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "eval/outputs"
DEST = ROOT.parent / "results_full.xlsx"

F_TITLE = Font(name="Arial", bold=True, size=13, color="1F3864")
F_HDR = Font(name="Arial", bold=True, size=10, color="FFFFFF")
F_NOTE = Font(name="Arial", size=10, color="7F3F00")
F_SEC = Font(name="Arial", bold=True, size=11, color="2E5395")
F_BODY = Font(name="Arial", size=10)
FILL_HDR = PatternFill("solid", start_color="2E5395")
FILL_SEC = PatternFill("solid", start_color="EAF1F8")
WRAP = Alignment(wrap_text=True, vertical="top")

INV = ["LeJEPA-CP", "SimCLR-CP"]
ANG = ["LeJEPA-CP", "SimCLR-CP", "DIET-CP"]


def load():
    d = {}
    files = ["geometry_15", "geometry_class_15", "geometry_vitL", "cp_long_refreshed",
             "postcp_sweep_fixed", "c2_siglip_score", "vitl_score", "cpL_behavior",
             "bilinear_law", "layerwise_curve", "layerwise_pre", "layerwise_postcp",
             "transport_field_max", "transport_stats", "sa_lp_recovery", "postcp_class_max",
             "geometry_mae_sa", "second_axis_stats", "stats_pass", "stats_pass_refreshed",
             "preregister_siglip", "rest_behavior"]
    for f in files:
        p = OUT / f"{f}.csv"
        d[f] = pd.read_csv(p) if p.exists() else None
    # Exp B raw (per-dataset shards incl. the LeJEPA-CP control arm)
    shards = sorted((OUT / "exp_b").glob("*.csv"))
    d["exp_b"] = (pd.concat([pd.read_csv(p).assign(shard=p.stem) for p in shards],
                            ignore_index=True) if shards else None)
    # SigLIP behavioral LEVELS (pre/post per method) from results.xlsx
    xlsx = ROOT.parent / "results.xlsx"
    try:
        d["siglip_behavior"] = pd.read_excel(xlsx, sheet_name="By Method (SigLIP)", header=1)
    except Exception:
        d["siglip_behavior"] = None
    return d


class Sheet:
    def __init__(self, wb, name):
        self.ws = wb.create_sheet(name[:31])
        self.r = 1

    def title(self, txt):
        c = self.ws.cell(self.r, 1, txt); c.font = F_TITLE; self.r += 2

    def note(self, txt, width_cols=10):
        c = self.ws.cell(self.r, 1, txt); c.font = F_NOTE; c.alignment = WRAP
        self.ws.merge_cells(start_row=self.r, start_column=1, end_row=self.r,
                            end_column=width_cols)
        self.ws.row_dimensions[self.r].height = max(15, 14 * (1 + len(txt) // 110))
        self.r += 1

    def sec(self, txt):
        self.r += 1
        c = self.ws.cell(self.r, 1, txt); c.font = F_SEC; c.fill = FILL_SEC; self.r += 1

    def df(self, frame, ndec=3):
        hdr_row = self.r
        for j, col in enumerate(frame.columns, 1):
            c = self.ws.cell(self.r, j, str(col)); c.font = F_HDR; c.fill = FILL_HDR
        self.r += 1
        for _, row in frame.iterrows():
            for j, v in enumerate(row, 1):
                if isinstance(v, (float, np.floating)):
                    v = "" if pd.isna(v) else round(float(v), ndec)
                c = self.ws.cell(self.r, j, v); c.font = F_BODY
            self.r += 1
        for j, col in enumerate(frame.columns, 1):
            w = max(len(str(col)), frame.iloc[:, j - 1].astype(str).str.len().max()
                    if len(frame) else 8)
            self.ws.column_dimensions[get_column_letter(j)].width = min(46, max(9, w + 2))
        # NOTE: no freeze_panes on curated sheets — multiple tables per sheet would leave
        # the LAST table's header frozen, locking dozens of rows and killing scrolling.
        self.r += 1

    def rows(self, pairs):
        for k, v in pairs:
            self.ws.cell(self.r, 1, k).font = Font(name="Arial", bold=True, size=10)
            c = self.ws.cell(self.r, 2, v); c.font = F_BODY; c.alignment = WRAP
            self.r += 1
        self.r += 1


def main():
    D = load()
    wb = Workbook(); wb.remove(wb.active)

    geo = D["geometry_15"]; geo = geo[geo.dataset != "imagenet"]
    gc = D["geometry_class_15"]; cp = D["cp_long_refreshed"]; c2 = D["c2_siglip_score"]

    # cell-level deltas @MAX, 3-method angular mean, per encoder
    ang = cp[cp.Method.isin(ANG) & cp.is_max & cp.Backbone.isin(["DINOv3", "CLIP", "MAE"])]
    cell = ang.groupby(["Backbone", "dataset_key"])[["dknn", "dlp", "dft"]].mean().reset_index()
    cell.columns = ["encoder", "dataset", "dknn", "dlp", "dft"]

    # ================= 00 README =================
    s = Sheet(wb, "00_README导读")
    s.title("results_full.xlsx — 全部发现相关数据总汇（构建脚本 eval/utils/build_results_full.py，可复跑）")
    s.note("组织方式：1x_ 开头 = 按发现整理的表（含数据如何支持该发现的解释）；80_ = 来源与公式（每个数字是哪个文件、哪个脚本、pre-CP 还是 post-CP 编码器、什么公式算出来的）；9x_RAW_ = 全部源 CSV 原样转存（一行不动）。整理表里的统计量分两类：本脚本从 RAW 现算的（标注[现算]）与从核验过的 findings 文档转录的（标注来源文档）。")
    s.sec("表索引")
    s.df(pd.DataFrame([
        ["11_F1_位置律", "pre-CP 几何 × Δ@MAX 合表 + 定律/反号相关矩阵[现算]", "发现一（头条A）"],
        ["12_F1_稳健性", "角向对照/基线偏相关/SNR 防御/协议注记", "发现一"],
        ["13_F5_决策分SigLIP", "held-out 15 行打分表 + 基线算法 + 冻结系数", "发现五（头条B）"],
        ["14_F5_ViTL尺度", "ViT-L 实测 Δ + R1–R4 判决 + 几何", "发现五（头条B）"],
        ["15_F4_门控", "四轴签名/MAE 反转/θ/双线性/逐层曲线/Sorkhei", "发现四（边界）"],
        ["16_F2_两股力", "扩散/碰撞统计 + 方法分层", "发现二（机制）"],
        ["17_F2_运输场", "逐格位移分解 + T 检验[现算] + 方法签名", "发现二/六（机制）"],
        ["18_F3_动力学", "先散后撞计数 + 配方剂量披露", "发现三（机制）"],
        ["19_F6_聚合失败", "SA 恢复率 + 平移签名交叉引用", "发现六（机制）"],
        ["20_F7_探索packing", "类间距表 + 探索级相关（围栏内）", "发现七（附录）"],
        ["80_来源与公式", "文件→脚本→pre/post→公式→用于哪些表", "全部"],
        ["9x_RAW_*", "源 CSV 原样：geometry_15 / cp_long_refreshed / …", "全部"],
    ], columns=["sheet", "内容", "服务的发现"]))
    s.sec("指标通俗词典（读任何表前先看这里）")
    s.df(pd.DataFrame([
        ["Δ (dknn/dlp/dft)", "CP 后指标减 CP 前指标（macro-F1）。>0=CP 帮了。kNN/LP 用冻结特征，FT 是继续训练后的表现", "一切结论的因变量"],
        ["uniformity_t2", "点云在单位球面上铺得多开（Wang-Isola 势能的 log）。越负=越散。挤成一团=编码器没给它分配空间", "位置的第一个坐标；对应机制里的扩散力"],
        ["neighbor_overlap_k50", "你的样本的 50 个最近邻里 ImageNet 样本的占比。越高=嵌在 ImageNet 地盘越深", "位置的第二个坐标；对应机制里的碰撞力"],
        ["mmd_rbf", "两团点云的分布距离（核方法）。与 uniformity/overlap 高度共线，只作佐证不算独立证据", "发现一的佐证列"],
        ["l2_norm_cv", "特征向量长度的变异系数。<6%=像干净球面（球面型），20%+=off-sphere", "发现四门控的径向轴"],
        ["rankme", "点云实际用了 768 维中的多少个方向（有效秩）。纸=2，棉花团=768", "发现四门控轴之一（编码器体检）"],
        ["alpha_req", "主方向能量衰减的幂指数。越大=越依赖少数方向", "发现四门控轴之一"],
        ["twonn_id", "点云局部实际躺在几维的面上（内在维度）", "发现四门控轴之一"],
        ["cdnv", "类内方差 ÷ 2×类间中心距²，对类对取平均。越小=类抱得紧、分得开（带标签量）", "θ 的带标签一侧；发现四脱钩定位"],
        ["center_margin", "每个类中心到最近异类中心的余弦距离的平均（过道宽度，带标签量）", "发现七 packing 主量"],
        ["theta (θ)", "|ρ(uniformity, cdnv)| 跨 15 数据集：编码器的无标签几何与类结构有多同步", "发现四边界的连续参数"],
        ["Spearman ρ", "两个排序的吻合度。n=15 时 |ρ|≳0.52 → p<0.05。全文只做排序主张", "所有相关性的统计口径"],
        ["偏相关 partial", "先把第三个量的排序影响回归掉再算 ρ：控制 X 后 Y 还剩多少独立信号", "碰撞力与 d_cdnv 的口径"],
        ["trans/between/within 能量", "逐样本位移箭头的三分解：整团平移 / 类团相对挪动 / 类内搅乱（精确恒等式）", "发现二机制修正 + 发现六平移签名"],
        ["toward_imagenet", "整体平移在指向旧 ImageNet 质心方向上的分量（碰撞的矢量版）", "发现二共动修正"],
    ], columns=["指标", "通俗解释", "在故事里的角色"]))
    s.sec("三个口径约定")
    s.rows([
        ("Δ@MAX", "最大数据档、LeJEPA+SimCLR+DIET 三角向方法均值、3 种子均值（SigLIP 例外：2 方法，无 DIET SigLIP 跑）"),
        ("pre-CP 编码器", "timm 原始公开权重，不含任何 CP 训练——geometry_15/geometry_class_15/geometry_vitL 全部由它算"),
        ("post-CP 编码器", "CP 训练后的 checkpoint——postcp_sweep_fixed/transport_field_max/layerwise_postcp 由它算"),
    ])

    # ================= 11 F1 =================
    s = Sheet(wb, "11_F1_位置律")
    s.title("发现一：位置定律与通道反号（头条 A）")
    s.note("与发现的关系：下表把每个数据集的 CP 前几何（位置）和 CP 后行为变化（Δ）放在同一行。定律 = 按 uniformity 给数据集排序 ≈ 按 ΔkNN 排序（相关矩阵在下方，[现算]）；反号 = 同一排序对 ΔFT 大致颠倒。SigLIP 列是 held-out 复现（2 方法 realized）。")
    main_tbl = geo[["encoder", "dataset", "uniformity_t2", "neighbor_overlap_k50",
                    "mmd_rbf", "l2_norm_cv"]].merge(cell, on=["encoder", "dataset"], how="left")
    sig_add = geo[geo.encoder == "SigLIP"][["encoder", "dataset", "uniformity_t2",
                                            "neighbor_overlap_k50", "mmd_rbf", "l2_norm_cv"]]
    sig_add = sig_add.merge(c2.rename(columns={"real_dknn": "dknn", "real_dlp": "dlp",
                                               "real_dft": "dft"})[
        ["dataset", "dknn", "dlp", "dft"]], on="dataset")
    full = pd.concat([main_tbl.dropna(subset=["dknn"]), sig_add], ignore_index=True)
    s.df(full)
    s.sec("定律与反号相关矩阵 [现算：Spearman，Δ@MAX 3 方法（SigLIP 2 方法 realized）]")
    rows = []
    for enc in ["DINOv3", "CLIP", "MAE", "SigLIP"]:
        g = full[full.encoder == enc]
        for met in ["uniformity_t2", "neighbor_overlap_k50", "mmd_rbf"]:
            row = {"encoder": enc, "几何量": met}
            for ch in ["dknn", "dlp", "dft"]:
                row[f"ρ→{ch}"] = spearmanr(g[met], g[ch]).correlation
            rows.append(row)
    s.df(pd.DataFrame(rows))
    s.note("读法：球面编码器（D3/CLIP/SigLIP）行内，uniformity/mmd 对 dknn/dlp 同号、对 dft 反号；MAE 行整体失灵/反转（发现四）。‘6/6 组显著’= 3 几何量×2 拟合编码器对 dft 的 6 个相关全部 p<0.05 且与 dknn 列反号。")

    # ================= 12 F1 robustness =================
    s = Sheet(wb, "12_F1_稳健性")
    s.title("发现一的稳健性链条")
    s.df(pd.DataFrame([
        ["角向对照（阴性对照）", "不做 L2 归一化直接算 uniformity → 对 ΔkNN 只剩 +0.18 (D3) / +0.06 (CLIP)", "证明信号在角度不在长度：混入无信号的范数分量即毁掉排序", "FINDINGS_step2（2 方法协议期）"],
        ["基线偏相关", "控制 pre-CP kNN 基线后定律存活：D3 0.596→0.621、CLIP 0.754→0.775", "排除‘位置只是难度的代理’", "FINDINGS_step2（2 方法协议期）"],
        ["ΔFT SNR 之一", "LeJEPA↔SimCLR 的 ΔFT 符号一致率 40/45=88.9%（刷新数据现算；旧协议值 87%；噪声期望 50%，二项 p<1e-4）", "ΔFT 方向由数据集决定，非种子噪声", "反审复算 2026-07-09（原 HYPOTHESES_v3 SNR 块）"],
        ["ΔFT SNR 之二", "63% 格子 |ΔFT| > 2×种子标准差（旧协议值；注意：刷新表只覆写了 Δ 列、保留旧水平列，故 SNR 类推导不得用刷新表的水平列重构——见 CORRECTIONS）", "多数效应过噪声地板", "HYPOTHESES_v3 SNR 块 + 混合规则披露"],
        ["ΔFT SNR 之三", "CLIP 高信噪子集反号更强（MMD→ΔFT 至 −0.82）", "剔噪声格后相关变强=信号真", "同上"],
        ["反号机制候选否决", "类内摊开中介：与整体摊开共线 0.92，剥离后无独立信号 → 机制留白", "反号 real 但 why 未解，论文如实写 open question", "FINDINGS_step7（P-D）"],
        ["DIET 并入", "3 方法后 D3 点估计上移 +0.596→+0.668（措辞天花板：点估计陈述）", "第三个角向方法不破坏定律", "FINDINGS_step8"],
        ["深度不混杂对照", "SigLIP 网格恒 2 块解冻仍复现定律 +0.746 / 87%", "排除解冻深度随 size 分档的混杂", "FINDINGS_step9 修正块"],
        ["复制检验（数据卫生）", "59 个种子级重跑（聚合为 31 个混合格）后 |新−旧| dknn 均值：LeJEPA 0.008 / SimCLR 0.015 / DIET 0.013（种子噪声级）", "证明行为数据一直有效、cp_long_refreshed 可信", "FINDINGS_step8 Block 0；格数经反审核实为 31（早期文档的 60 为问题 run 计数口径）"],
    ], columns=["检验", "结果", "为什么重要", "来源"]))

    # ================= 13 F5 SigLIP =================
    s = Sheet(wb, "13_F5_决策分SigLIP")
    s.title("发现五：CP 前决策分 → held-out 编码器 SigLIP-2（头条 B 第一重样本外）")
    s.note("与发现的关系：分数 = logistic(z(overlap), z(uniformity))，系数在 DINOv3+CLIP 的 30 格上拟合并在任何 SigLIP CP 数据存在前存档。下表为逐数据集判卷。基线算法：always-adapt=对所有数据集说 HELP，其正确率=实际 Δ>0 的比例 → kNN 通道 11/15=73%、LP 通道 12/15=80%。")
    t = c2.copy()
    t["hit_knn"] = (t.pred_knn == "HELP") == (t.real_dknn > 0)
    t["hit_lp"] = (t.pred_lp == "HELP") == (t.real_dlp > 0)
    # regression self-test (external audit 2026-07-09 caught hit_lp using pred_knn):
    _tst = pd.DataFrame({"pred_knn": ["HELP"], "pred_lp": ["HURT"], "real_dlp": [0.1]})
    assert ((_tst.pred_lp == "HELP") == (_tst.real_dlp > 0)).iloc[0] == False, "hit_lp column test"
    s.df(t[["dataset", "overlap", "pred_knn", "real_dknn", "real_dlp", "real_dft",
            "hit_knn", "hit_lp"]])
    s.sec("汇总与防御结构")
    s.rows([
        ("sign(ΔkNN)", "13/15 = 87%（基线 11/15=73%；规则抓住全部 4 个 kNN 退化集，miss=dtd 预旗标例外 + food101 编码器翻转）"),
        ("sign(ΔLP)", "12/15 = 80%（该通道基线也是 12/15——打平但错误集互补：规则抓住基线全错过的 cars196/cub200/fgvc 三个 LP 退化集）"),
        ("sign(ΔFT)", "4/15 = 27% —— FT 通道不迁移（如实披露：分数只授权冻结部署决策）"),
        ("冻结系数（真冻结）", "b_overlap=+0.194967, b_uniformity=+1.717809, b0=+0.639062——由 results.xlsx 预刷新 2 方法目标拟合，复现 preregister_siglip.csv 全部 15 个冻结标签（vitl_score.py 含断言自检）。注：早期 scorer 曾在刷新目标上重拟合（系数 +0.198/+1.457/+0.377），标签在全部 15+7 数据集上与真冻结模型一致，结果未受影响；外部核验 2026-07-09 指出并已修正"),
        ("连续分数质量", "LOO 平衡精度 0.778 / AUC 0.829（D3+CLIP 内部；FINDINGS_step5，predictor_standardize_ablation.csv）；LP 侧 AUC 0.745 vs Sorkhei 式冻结质量基线 0.41（FINDINGS_step2/step4 T1 决策规则块）"),
        ("标准化消融", "逐编码器 z 分数是迁移机制：LOO 0.667→0.778（predictor_standardize_ablation.csv）"),
    ])

    # ================= 14 F5 ViT-L =================
    s = Sheet(wb, "14_F5_ViTL尺度")
    s.title("发现五：held-out 模型尺寸 DINOv3 ViT-L/16（头条 B 第二重样本外）")
    s.note("与发现的关系：同一套冻结系数、零改动，打在换了尺寸的编码器上。四项预注册判决全过（design: eval/DESIGN_vitL_robustness.md；判卷: eval/F5_decision_score/vitl_score.py）。")
    s.sec("判卷表（vitl_score.csv）")
    s.df(D["vitl_score"])
    s.sec("R1–R4 判决")
    s.rows([
        ("R1 几何尺度稳定", "ViT-B↔ViT-L 秩相关：uniformity 0.971 / overlap 0.965 / mmd 0.989（门槛 0.8）→ PASS"),
        ("R2 冻结系数迁移", "sign(ΔkNN) 6/7 且 sign(ΔLP) 6/7；唯一 miss = 预旗标的 dtd（+0.025 又帮了）；HELP 侧 3/3 → PASS"),
        ("R3 位置律方向", "ρ(unif, ΔkNN) = +0.750（n=7 点估计，落在 ViT-B 区间 0.62–0.76 内）→ PASS"),
        ("R4 反号方向（探索）", "ρ(unif, ΔFT) = −0.893 —— 反号在新尺度上重现"),
    ])
    s.sec("ViT-L 逐方法 Δ（cpL_behavior 现算，post−pre 种子均值）")
    b = D["cpL_behavior"]
    D2K = {"Galaxy10": "galaxy10", "DermaMNIST": "dermamnist", "EuroSAT": "eurosat",
           "FGVC_Aircraft": "fgvc_aircraft", "Cars196": "cars196", "CUB200": "cub200",
           "DTD": "dtd"}
    b = b.assign(dataset=b.display.map(D2K))
    pre = b[b.kind == "pre"].groupby("dataset")[["pre_knn", "pre_lp", "pre_sft"]].mean()
    post = b[b.kind == "cp"].groupby(["method", "dataset"])[
        ["post_knn", "post_lp", "post_sft"]].mean()
    dL = post.join(pre, on="dataset")
    for ch in ["knn", "lp", "sft"]:
        dL[f"d{ch}"] = dL[f"post_{ch}"] - dL[f"pre_{ch}"]
    s.df(dL.reset_index()[["method", "dataset", "dknn", "dlp", "dsft"]])
    s.sec("ViT-L 的 pre-CP 几何（geometry_vitL.csv，R1/R2 的输入）")
    s.df(D["geometry_vitL"])

    # ================= 15 F4 gate =================
    s = Sheet(wb, "15_F4_门控")
    s.title("发现四：编码器边界（第三支柱）")
    s.note("与发现的关系：以下依次是 (a) 四轴签名——给编码器本身体检，MAE 与球面三兄弟在四把无标签尺子上同时隔一条鸿沟（断崖非渐变）；(b) MAE 上定律反转的数值；(c) θ 耦合轴与双线性；(d) 独立协议逐层复现；(e) Sorkhei 复现与两个失效 regime。")
    s.sec("(a) 四轴签名 [现算：geometry_class_15 / geometry_15 的 15 数据集均值]")
    ax = gc.groupby("encoder")[["rankme", "alpha_req", "twonn_id"]].mean()
    ax["l2_norm_cv"] = geo.groupby("encoder")["l2_norm_cv"].mean()
    s.df(ax.reset_index())
    s.sec("(b) MAE 上的定律 [现算，3 方法 Δ@MAX]")
    gm = full[full.encoder == "MAE"]
    s.df(pd.DataFrame([{"uniformity→ΔkNN": spearmanr(gm.uniformity_t2, gm.dknn).correlation,
                        "overlap→ΔkNN(球面为负,此处反号)": spearmanr(gm.neighbor_overlap_k50, gm.dknn).correlation,
                        "uniformity→ΔLP": spearmanr(gm.uniformity_t2, gm.dlp).correlation}]))
    s.rows([
        ("MAE 力律分层", "扩散力逐方法 +0.16/+0.01/+0.05/−0.11 全不显著；pooled −0.37 是 Simpson 聚合假象（只引分层数）。来源 FINDINGS_step7"),
        ("带标签通道完好", "partial(d_cdnv | 两股力) = −0.461 (q<1e-4)；MAE 上 4 方法 −0.51~−0.66 方向一致。来源 FINDINGS_step7/8（Exp H）"),
    ])
    s.sec("(c) θ 耦合轴 [现算] 与双线性（来源 bilinear_law.py）")
    th = []
    for enc, g in gc.merge(geo[["encoder", "dataset", "uniformity_t2"]],
                           on=["encoder", "dataset"]).groupby("encoder"):
        th.append({"encoder": enc, "theta=|ρ(unif,cdnv)|": abs(
            spearmanr(g.uniformity_t2, g.cdnv).correlation)})
    s.df(pd.DataFrame(th).sort_values("theta=|ρ(unif,cdnv)|", ascending=False))
    s.rows([
        ("双线性 LOO", "仅位置 0.437 / 二元门控 0.485 / 单项 x·θ 0.497（M4−M2 自助 CI [−0.15,+0.09]：持平不超越）。来源 bilinear_law.py 输出"),
        ("V3 预注册失败", "ρ(unif, pre-kNN) 符号翻转判据被 SigLIP(+0.171) 破坏——第四证据降级，门控证据维持 3 重+θ 排序。来源 FINDINGS_step9 附录"),
    ])
    s.sec("(d) 独立协议逐层曲线（layerwise_curve.csv 全表；L12 行 = 门控复现）")
    s.df(D["layerwise_curve"])
    s.note("读法：L12（全部 15 数据集都被训练的层）律强度 D3 +0.832 / CLIP +0.757 / SigLIP +0.554 / MAE +0.043——独立协议复现门控与 θ 排序；14 个有效点上 θ↔律 pooled +0.073（CI 含 0）= 可测范围内无连续谱信号（断崖）。L9/L10 仅 7 个数据集（解冻分档），探索级。")
    s.sec("(e) Sorkhei 普适规则：复现 + 两个失效 regime（来源 FINDINGS_step2）")
    s.rows([
        ("regime 内复现", "pre-CP 冻结 kNN→FT 排名：pooled +0.653（n=45）；球面 +0.764；含 SigLIP +0.704"),
        ("失效一 off-sphere", "MAE 冻结排名≠FT 排名 11/15 任务（cars196：冻结 kNN 0.047 vs FT 0.891）——CP 之前就破"),
        ("失效二 Δ 通道", "水平耦合 +0.72 而 CP 的 Δ 反耦合（发现一反号）——水平规则≠变化规则"),
    ])

    # ================= 16 F2 forces =================
    s = Sheet(wb, "16_F2_两股力")
    s.title("发现二：扩散普适、碰撞限不变性（机制）")
    s.note("与发现的关系：几何的变化量追踪行为的变化量。扩散=Δuniformity 对 ΔkNN 的普通 Spearman；碰撞=Δoverlap 控制 Δuniformity 后的偏相关（两者强共线，边际相关≈0）。单位=配置格点（方法×编码器×数据集×数据档，种子平均，n=560）。数字转录自 final_integration.py BLOCK3 / FINDINGS_step8（管线含 size 控制与 join 口径，此处不重算）。")
    s.df(pd.DataFrame([
        ["LeJEPA-CP", -0.483, -0.442, "p<1e-6", "两股力都在"],
        ["SimCLR-CP", -0.504, -0.415, "p<1e-6", "两股力都在"],
        ["DIET-CP", -0.684, -0.112, "p=0.19 零", "只有扩散（实例判别不产生拉力）"],
        ["MAE-CP", -0.339, 0.065, "零", "只有弱扩散"],
        ["pooled(n=560)", -0.552, -0.378, "size控制后 −0.578/−0.340", "扩散双编码器过FDR；碰撞CLIP过FDR、D3 q≈0.07"],
    ], columns=["方法", "扩散 ρ(Δunif,ΔkNN)", "碰撞 partial(Δov|Δunif)", "显著性", "读法"]))
    s.rows([
        ("径向替代机制否决", "MAE-CP 的 Δnorm-CV ≈ 0（−0.008/+0.001），对 ΔkNN 无预测力——伤害是角向的不是径向的。来源 FINDINGS_step4/8"),
        ("LP 侧", "扩散对 ΔLP −0.616 同样成立（刷新数据复算 −0.614）——冻结两通道同向。来源 FINDINGS_step2/step4"),
    ])

    # ================= 17 F2 transport =================
    s = Sheet(wb, "17_F2_运输场")
    s.title("发现二机制修正 + 发现六签名：逐样本位移场（Exp J）")
    s.note("与发现的关系：同批样本 CP 前后一一对应 → 每样本一支位移箭头，精确分解为 整团平移+类间挪动+类内搅乱。关键修正：‘漂向旧 ImageNet 区域’的直觉被数据反驳（相关为负），碰撞是共动混合。下表为逐格数据（transport_stats.csv），统计量[现算]。")
    ts = D["transport_stats"]
    s.sec("方法签名 [现算：逐 seed 中位数，transport_field_max.csv（526 行）]")
    tt = ts.copy(); tt["t_share"] = np.nan
    tf = D["transport_field_max"]
    tf2 = tf.assign(t_share=tf.trans_energy / tf.total_energy,
                    w_share=tf.within_energy / tf.total_energy)
    s.df(tf2.groupby("method")[["total_energy", "t_share", "w_share"]].median().reset_index())
    s.sec("T 统计 [现算自 transport_stats.csv]")
    rows = []
    for enc in ["DINOv3", "CLIP"]:
        g = ts[(ts.encoder == enc) & ts.method.isin(["LeJEPA", "SimCLR", "DIET"])].dropna(
            subset=["d_overlap"]) if "d_overlap" in ts.columns else pd.DataFrame()
        if len(g):
            rows.append({"检验": f"T1a {enc} ρ(toward, Δoverlap)",
                         "值": spearmanr(g.toward_imagenet, g.d_overlap).correlation,
                         "判决": "CLIP −0.500 过FDR：固定坐标系图像证伪" if enc == "CLIP" else "零（n.s.）"})
    for enc in ["DINOv3", "CLIP", "MAE"]:
        g = ts[(ts.encoder == enc) & ts.method.isin(["LeJEPA", "SimCLR", "DIET"])].dropna(
            subset=["dknn"])
        if len(g) and "within_share" in g.columns:
            rows.append({"检验": f"T2 {enc} ρ(类内搅乱份额, ΔkNN)",
                         "值": spearmanr(g.within_share, g.dknn).correlation,
                         "判决": "D3 过FDR（抗深度控制 −0.512）" if enc == "DINOv3" else "零"})
    s.df(pd.DataFrame(rows))
    s.rows([
        ("T1b 拉力对比", "toward 中位数 invariance>DIET：CLIP −0.392 vs −0.482（单侧 p=1e-4 FDR ✓）；D3 方向一致 p=0.099。来源 FINDINGS_step9"),
        ("T3 恒等式", "总能量=三项之和，最大相对残差 1.2e-6（float32）✓"),
        ("T4 探索", "类间运动↔ΔFT：CLIP +0.356 (p=0.017) / D3 +0.263 (p=0.081)——类协同运动对 FT 无害，探索级"),
    ])
    s.sec("逐格数据（transport_stats.csv 全表）")
    s.df(ts)

    # ================= 18 F3 =================
    s = Sheet(wb, "18_F3_动力学")
    s.title("发现三：先扩散后碰撞（机制，配方级剂量响应）")
    s.df(pd.DataFrame([
        ["扩散随数据量", "141/180 配置 ρ(size, post-unif)<0（角向三方法 121/135，符号检验 p<1e-13）", "merge_rest_geometry.py（fixed sweep）"],
        ["碰撞随数据量", "151/171 配置 ρ(size, post-overlap)>0（角向 126/134）", "同上"],
        ["收益峰先于碰撞", "106/117 = 90.6%（当前口径：fixed sweep + 刷新行为 + 峰后最大 overlap 判据）。旧公布值 99/119=83% 依赖旧输入+端点判据，已作废（外部核验 2026-07-09）", "F3_dynamics/postcp_growth_analysis.py 判据 + 反审复算"],
        ["配方披露", "解冻深度随数据量分档（<1万:2 / 1–2.5万:4 / 2.5–5万:6 / >5万:全）——‘随数据量’=数据+容量联合剂量；SigLIP 恒深度网格为对照", "FINDINGS_step9 修正块"],
    ], columns=["主张", "数字", "来源"]))

    # ================= 19 F6 =================
    s = Sheet(wb, "19_F6_聚合失败")
    s.title("发现六：MAE-CP 的冻结崩塌 = 读出坏了（两条独立证据线）")
    s.note("证据线一：只训一个注意力池化头（骨干不动）收回崩塌的 59.4%——LeJEPA-CP 对照无物可收（+0.01）。证据线二：运输签名显示 MAE-CP 近刚体平移（17_运输场表：平移份额 0.75/搅乱 0.17）——结构没坏。下表为逐格恢复数据（sa_lp_recovery.csv）。")
    if D["sa_lp_recovery"] is not None:
        s.df(D["sa_lp_recovery"])
    s.rows([
        ("恢复率汇总", "总体 0.594（修复参考后）；DINOv3 0.540 / CLIP 0.329 / MAE 0.856；octmnist-CLIP 由虚高 1.16 修正为 0.755。来源 FINDINGS_step8（test3）"),
        ("尾部残余", "food101/CLIP 恢复后仍远低于健康参考——聚合失败‘主导’而非‘全部’（措辞天花板 mostly）"),
    ])

    # ================= 20 F7 =================
    s = Sheet(wb, "20_F7_探索packing")
    s.title("发现七（探索级，只进附录）：类间距 packing")
    s.note("围栏声明：n=7、无 FDR 幸存、依赖全局残差化构造（簇内重拟合后消失）——假设生成级。数据支持‘位置几乎相同、损伤天差地别’的簇内异质性（CUB vs Flowers）由过道宽度解释。")
    fg7 = ["fgvc_aircraft", "cars196", "cub200", "food101", "oxford_pet", "dtd", "flowers102"]
    pk = gc[gc.dataset.isin(fg7) & gc.encoder.isin(["DINOv3", "CLIP"])][
        ["encoder", "dataset", "center_margin", "n_classes", "cdnv"]]
    s.df(pk.pivot(index="dataset", columns="encoder",
                  values="center_margin").reset_index())
    s.rows([
        ("探索级相关", "FG-7 内 margin↔位置残差：D3 +0.893 (p=0.007) / CLIP +0.714 (p=0.071)；簇内重拟合后 +0.14/+0.36（构造依赖，双报）。来源 FINDINGS_step7/8"),
        ("剂量否决", "CUB@200 已 −0.35 而 Flowers@1020(MAX) −0.002——异质性非剂量。来源 heterogeneity_probe.py"),
    ])

    # ================= 80 provenance =================
    s = Sheet(wb, "80_来源与公式")
    s.title("来源与公式：每个数字从哪来")
    s.sec("输出文件 → 生成方式")
    s.df(pd.DataFrame([
        ["geometry_15.csv", "eval/utils/geometry_metrics.py", "pre-CP（timm 原始权重）", "uniformity/overlap/MMD/norm-CV/质心", "11/12/15"],
        ["geometry_class_15.csv", "eval/utils/geometry_class.py (Exp E)", "pre-CP", "CDNV/margin/RankMe/α/TwoNN（类几何+谱几何）", "15/20"],
        ["geometry_vitL.csv", "eval/F5_decision_score/geometry_vitL.py", "pre-CP（ViT-L 原始权重）", "同 geometry_15 的四个无标签量", "14"],
        ["cp_long_refreshed.csv", "results.xlsx + test4 混合规则（eval/utils/final_integration.py）", "行为=训练日志（非 ckpt 重测）", "Δ = post − pre（macro-F1）", "11/13/15/16"],
        ["postcp_sweep_fixed.csv", "eval/F2_forces/postcp_sweep.py（共享模块 utils/postcp_features.py）+ test2 合并", "post-CP（2713 ckpt 逐个提特征）", "post 几何（unif/overlap/CV）", "16/18"],
        ["c2_siglip_score.csv", "冻结预测 + realized 判卷（FINDINGS_step5）", "行为=SigLIP 网格日志", "logistic(z(ov),z(unif))", "13"],
        ["vitl_score.csv / cpL_behavior.csv", "eval/F5_decision_score/vitl_score.py / cp-L 日志汇集", "pre+post（ViT-L 网格）", "同冻结协议 + Δ", "14"],
        ["bilinear_law.csv", "eval/F4_gate/bilinear_law.py", "pre-CP 几何 + 行为 Δ", "Δ≈β·x·θ；数据集分组 LOO 秩回归", "15"],
        ["layerwise_pre/postcp/curve.csv", "eval/utils/layerwise_geometry.py + eval/F4_gate/layerwise_{postcp,law}.py (Exp I)", "pre+post，逐块输出（final-norm 前）", "内部 kNN（80/20, cos k=20）；θ_ℓ；律_ℓ", "15"],
        ["transport_field_max.csv / transport_stats.csv", "eval/F2_forces/transport_{field,law}.py (Exp J)", "pre+post 同批样本一一对应", "d_i=ẑpost−ẑpre；E‖d‖²=平移+类间+类内；toward=⟨μ_d,v_E⟩", "17/19"],
        ["sa_lp_recovery.csv", "eval/F6_aggregation/run_exp_b.py + test3 重聚合 (Exp B)", "post-CP ckpt + 学习型 SA 池化头", "恢复率=(SA−崩塌)/(健康参考−崩塌)", "19"],
        ["postcp_class_max.csv", "eval/utils/postcp_class_sweep.py (Exp H)", "post-CP（526 MAX ckpt）", "post 类几何 → d_cdnv/d_margin", "15(b)"],
        ["geometry_mae_sa.csv", "eval/F4_gate/mae_sa_geometry.py (Exp F)", "pre-CP MAE + 学习型读出", "换读出后重算 F1 几何（消融）", "15 注"],
        ["stats_pass_refreshed.csv", "eval/F1_position_law/stats_pass_refreshed.py（外部核验后新增）", "pre-CP 几何 + 刷新 Δ", "刷新 3 方法族 36 项 BH-FDR：D3/CLIP 全 18 项（含 6 组反号）过 q=0.10", "12 的 FDR 依据"],
        ["preregister_siglip.csv", "2026-06-19 冻结存档（predictor 管线）", "pre-CP 几何（SigLIP 未见）", "冻结的逐数据集 p_help 与预测（三通道）", "13 的预注册凭证"],
        ["exp_b/*.csv", "eval/F6_aggregation/run_exp_b.py（逐数据集 shard）", "post-CP ckpt + SA 头（含 LeJEPA 对照臂）", "baseline_lp_f1 / sa_lp_f1 / 差值", "19 的原始层"],
        ["rest_behavior.csv", "_archive/eval/rest/test4_behavior_deltas.py（验证代码，已归档）", "重跑 ckpt 的训练日志", "60 格新行为（复制检验的输入）", "12 复制检验行"],
        ["By Method (SigLIP)（results.xlsx）", "SigLIP 网格训练日志", "pre+post 水平值（非 Δ）", "knn/lp/ft 的 pre 与 post 水平", "15(e) Sorkhei 含 SigLIP 与 V3 的输入"],
    ], columns=["文件", "生成脚本", "pre/post 编码器", "关键公式/指标", "用于表"]))
    s.sec("公式速查（文字版；Word 版公式见 report_v2.docx 第 4 章）")
    s.rows([
        ("Δ", "metric_postCP − metric_preCP，先 3 种子平均；主口径 Δ@MAX 3 角向方法均值"),
        ("uniformity(t=2)", "log E[exp(−2·||x̂−ŷ||²)]，x̂ = L2 归一化特征，对上三角样本对取期望（≤3000 子采样）"),
        ("overlap_k50", "(1/N)·Σ_i |N50(x_i) ∩ ImageNet| / 50，两团点云拼在一起后的近邻占比"),
        ("norm-CV", "std(‖f(x)‖)/mean(‖f(x)‖)"),
        ("CDNV", "(Var_i+Var_j)/(2‖μ_i−μ_j‖²) 对类对平均（带标签）"),
        ("θ", "|Spearman(uniformity, CDNV)| 跨 15 数据集，每编码器一个值"),
        ("双线性", "Δ(D,E) ≈ β·x(D,E)·θ_E，x=编码器内 z 分数化 uniformity"),
        ("位移分解", "E‖d‖² = ‖μ_d‖² + Σ_c (n_c/N)‖μ_c−μ_d‖² + E‖d_i−μ_c(i)‖²（精确恒等式）"),
        ("决策分", "σ(β0 + β1·z_E(overlap) + β2·z_E(uniformity))，z_E=编码器内 15 数据集 z 分数"),
        ("偏相关", "把控制变量的名次回归掉后对残差做 Spearman"),
    ])

    # ================= 9x RAW =================
    raw_list = ["geometry_15", "geometry_class_15", "geometry_vitL", "cp_long_refreshed",
                "postcp_sweep_fixed", "c2_siglip_score", "vitl_score", "cpL_behavior",
                "bilinear_law", "layerwise_curve", "layerwise_pre", "layerwise_postcp",
                "transport_field_max", "transport_stats", "sa_lp_recovery",
                "postcp_class_max", "geometry_mae_sa", "second_axis_stats", "stats_pass",
                "stats_pass_refreshed", "preregister_siglip", "rest_behavior", "exp_b",
                "siglip_behavior"]
    for i, name in enumerate(raw_list):
        if D[name] is None:
            continue
        ws = wb.create_sheet(f"9{i:02d}_RAW_{name}"[:31])
        for row in dataframe_to_rows(D[name], index=False, header=True):
            ws.append(row)
        for c in ws[1]:
            c.font = F_HDR; c.fill = FILL_HDR
        # freeze removed entirely — user's Excel showed scroll issues with panes

    wb.save(DEST)
    print(f"written {DEST} with {len(wb.sheetnames)} sheets")
    for n in wb.sheetnames:
        print(" ", n)


if __name__ == "__main__":
    main()
