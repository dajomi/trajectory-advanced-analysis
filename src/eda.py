"""
탐색적 데이터 분석 (EDA) - MovingPandas 기반
모든 시각화는 outputs/figures/ 에 PNG로 저장
"""

import warnings
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import koreanize_matplotlib
import movingpandas as mpd
from pathlib import Path

warnings.filterwarnings("ignore")

from geohash_utils import (
    build_flow_matrix,
    build_time_congestion,
    build_time_mode_matrix,
    geohash_center_coords,
    compute_trajectory_stats,
)

FIGURE_DIR = Path(__file__).parent.parent / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TRANSPORT_PALETTE = {
    "walk":    "#2ECC71",
    "bike":    "#3498DB",
    "bus":     "#E67E22",
    "car":     "#E74C3C",
    "taxi":    "#9B59B6",
    "subway":  "#1ABC9C",
    "train":   "#34495E",
    "airplane":"#F39C12",
    "boat":    "#16A085",
    "run":     "#27AE60",
    "unknown": "#BDC3C7",
}

TIME_SLOT_ORDER = [
    "심야(0-6)", "출근(6-9)", "오전(9-12)", "점심(12-14)",
    "오후(14-18)", "저녁(18-21)", "야간(21-24)",
]


def save_fig(name: str, dpi: int = 150):
    path = FIGURE_DIR / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  저장: {path}")
    plt.close()


# ────────────────────────────────────────────
# 1. 기본 요약
# ────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("=" * 55)
    print("[ 데이터 기본 요약 ]")
    print(f"  총 포인트 수    : {len(df):>12,}")
    print(f"  사용자 수       : {df['user_id'].nunique():>12,}")
    print(f"  기간            : {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"  위도 범위       : {df['lat'].min():.4f} ~ {df['lat'].max():.4f}")
    print(f"  경도 범위       : {df['lon'].min():.4f} ~ {df['lon'].max():.4f}")
    print(f"  교통수단 종류   : {sorted(df['transport_mode'].unique())}")
    labeled = (df["transport_mode"] != "unknown").sum()
    print(f"  레이블 포인트   : {labeled:>12,} ({labeled/len(df)*100:.1f}%)")
    print("=" * 55)


def plot_data_overview(df: pd.DataFrame):
    """사용자별 포인트 수 + 교통수단 분포"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("데이터 개요", fontsize=14, fontweight="bold")

    # 사용자별 포인트 수
    user_counts = df.groupby("user_id").size().sort_values(ascending=False)
    axes[0].bar(range(len(user_counts)), user_counts.values, color="#3498DB", alpha=0.75)
    axes[0].set_title("사용자별 GPS 포인트 수")
    axes[0].set_xlabel("사용자 (내림차순 정렬)")
    axes[0].set_ylabel("포인트 수")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # 교통수단 분포 (unknown 제외)
    mode_counts = (
        df[df["transport_mode"] != "unknown"]["transport_mode"]
        .value_counts()
    )
    colors = [TRANSPORT_PALETTE.get(m, "#95A5A6") for m in mode_counts.index]
    axes[1].barh(mode_counts.index, mode_counts.values, color=colors, alpha=0.85)
    axes[1].set_title("교통수단별 GPS 포인트 수")
    axes[1].set_xlabel("포인트 수")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    save_fig("01_data_overview")


# ────────────────────────────────────────────
# 2. MovingPandas 궤적 통계
# ────────────────────────────────────────────

def plot_trajectory_stats(tc: mpd.TrajectoryCollection):
    """MovingPandas 기반 궤적 통계 시각화"""
    print("  궤적 통계 계산 중 (MovingPandas)...")
    stats = compute_trajectory_stats(tc)
    if stats.empty:
        print("  궤적 통계 없음")
        return stats

    labeled = stats[stats["transport_mode"] != "unknown"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("MovingPandas 궤적 통계", fontsize=14, fontweight="bold")

    # (1) 교통수단별 평균 속도 분포 (boxplot)
    if not labeled.empty:
        modes = labeled["transport_mode"].unique()
        data_bp = [labeled[labeled["transport_mode"] == m]["avg_speed_kmh"].dropna().values
                   for m in modes]
        colors = [TRANSPORT_PALETTE.get(m, "#95A5A6") for m in modes]
        bp = axes[0].boxplot(data_bp, patch_artist=True, labels=modes)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axes[0].set_title("교통수단별 평균 속도 분포")
        axes[0].set_ylabel("평균 속도 (km/h)")
        axes[0].set_xticklabels(modes, rotation=30, ha="right")

    # (2) 교통수단별 평균 이동 거리
    if not labeled.empty:
        dist_mean = labeled.groupby("transport_mode")["length_m"].mean().sort_values(ascending=False)
        colors2 = [TRANSPORT_PALETTE.get(m, "#95A5A6") for m in dist_mean.index]
        axes[1].bar(dist_mean.index, dist_mean.values / 1000, color=colors2, alpha=0.85)
        axes[1].set_title("교통수단별 평균 이동 거리")
        axes[1].set_ylabel("평균 거리 (km)")
        axes[1].set_xticklabels(dist_mean.index, rotation=30, ha="right")
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))

    # (3) 교통수단별 평균 이동 시간
    if not labeled.empty:
        dur_mean = labeled.groupby("transport_mode")["duration_s"].mean().sort_values(ascending=False)
        colors3 = [TRANSPORT_PALETTE.get(m, "#95A5A6") for m in dur_mean.index]
        axes[2].bar(dur_mean.index, dur_mean.values / 60, color=colors3, alpha=0.85)
        axes[2].set_title("교통수단별 평균 이동 시간")
        axes[2].set_ylabel("평균 시간 (분)")
        axes[2].set_xticklabels(dur_mean.index, rotation=30, ha="right")

    plt.tight_layout()
    save_fig("02_trajectory_stats")
    return stats


# ────────────────────────────────────────────
# 3. Geohash별 교통수단 흐름
# ────────────────────────────────────────────

def plot_geohash_flow(df: pd.DataFrame, top_n: int = 30):
    """상위 N 개 Geohash의 교통수단별 흐름 (스택 바)"""
    labeled = df[df["transport_mode"] != "unknown"]
    if labeled.empty:
        print("  레이블 데이터 없음")
        return

    flow = build_flow_matrix(labeled)
    top_gh = flow.nlargest(top_n, "total")
    mode_cols = [c for c in top_gh.columns if c not in ["total", "unknown"]]
    colors = [TRANSPORT_PALETTE.get(m, "#95A5A6") for m in mode_cols]

    fig, ax = plt.subplots(figsize=(16, 6))
    top_gh[mode_cols].plot(kind="bar", stacked=True, ax=ax, color=colors, alpha=0.85)
    ax.set_title(f"Geohash별 교통수단 흐름 (상위 {top_n}개)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Geohash")
    ax.set_ylabel("GPS 포인트 수")
    ax.legend(title="교통수단", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    save_fig("03_geohash_flow_by_mode")


def plot_geohash_density(df: pd.DataFrame):
    """Geohash 격자 밀도 산점도 히트맵"""
    density = df.groupby("geohash").size().reset_index(name="count")
    coords = geohash_center_coords(density["geohash"])
    density = pd.concat([density, coords], axis=1).dropna()

    # 교통수단별 대표 색상 오버레이
    labeled = df[df["transport_mode"] != "unknown"]
    mode_dominant = (
        labeled.groupby("geohash")["transport_mode"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    density = density.merge(mode_dominant, on="geohash", how="left")
    density["transport_mode"] = density["transport_mode"].fillna("unknown")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Geohash 격자 밀도 히트맵", fontsize=14, fontweight="bold")

    # 왼쪽: 밀도 (단색)
    q95 = density["count"].quantile(0.95)
    sc = axes[0].scatter(
        density["lon"], density["lat"],
        c=density["count"].clip(upper=q95),
        cmap="YlOrRd", s=8, alpha=0.7, edgecolors="none"
    )
    plt.colorbar(sc, ax=axes[0], label="GPS 포인트 수")
    axes[0].set_title("전체 밀도")
    axes[0].set_xlabel("경도")
    axes[0].set_ylabel("위도")

    # 오른쪽: 교통수단별 색상
    for mode, group in density.groupby("transport_mode"):
        color = TRANSPORT_PALETTE.get(mode, "#95A5A6")
        axes[1].scatter(group["lon"], group["lat"],
                        c=color, s=8, alpha=0.6, label=mode, edgecolors="none")
    axes[1].set_title("교통수단별 분포 (격자 지배 수단)")
    axes[1].set_xlabel("경도")
    axes[1].set_ylabel("위도")
    axes[1].legend(title="교통수단", bbox_to_anchor=(1.01, 1), loc="upper left",
                   markerscale=2, fontsize=8)

    plt.tight_layout()
    save_fig("04_geohash_density_heatmap")


# ────────────────────────────────────────────
# 4. 시간대별 혼잡도
# ────────────────────────────────────────────

def plot_hourly_congestion(df: pd.DataFrame):
    """24시간 + 시간 슬롯별 혼잡도"""
    hourly = build_time_congestion(df, "hour")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("시간대별 혼잡도", fontsize=14, fontweight="bold")

    axes[0].bar(hourly["hour"], hourly["count"], color="#3498DB", alpha=0.8, edgecolor="white")
    axes[0].set_title("시간대별 GPS 포인트 수")
    axes[0].set_xlabel("시간 (시)")
    axes[0].set_ylabel("포인트 수")
    axes[0].set_xticks(range(0, 24))
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    slot_counts = df["time_slot"].value_counts().reindex(TIME_SLOT_ORDER).fillna(0)
    slot_colors = ["#2C3E50", "#E74C3C", "#F39C12", "#27AE60",
                   "#3498DB", "#9B59B6", "#1ABC9C"]
    axes[1].bar(slot_counts.index, slot_counts.values, color=slot_colors, alpha=0.85)
    axes[1].set_title("시간 슬롯별 GPS 포인트 수")
    axes[1].set_xlabel("시간 슬롯")
    axes[1].set_ylabel("포인트 수")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    save_fig("05_hourly_congestion")


def plot_weekday_congestion(df: pd.DataFrame):
    """요일별 혼잡도"""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    labels_kr = ["월", "화", "수", "목", "금", "토", "일"]
    wd = df["weekday_name"].value_counts().reindex(order).fillna(0)
    colors = ["#3498DB"] * 5 + ["#E74C3C"] * 2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels_kr, wd.values, color=colors, alpha=0.85, edgecolor="white")
    ax.set_title("요일별 혼잡도", fontsize=13, fontweight="bold")
    ax.set_xlabel("요일")
    ax.set_ylabel("GPS 포인트 수")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(handles=[
        mpatches.Patch(color="#3498DB", alpha=0.85, label="평일"),
        mpatches.Patch(color="#E74C3C", alpha=0.85, label="주말"),
    ], loc="upper right")
    plt.tight_layout()
    save_fig("06_weekday_congestion")


# ────────────────────────────────────────────
# 5. 시간대 × 교통수단 혼잡도
# ────────────────────────────────────────────

def plot_time_mode_heatmap(df: pd.DataFrame):
    """시간 슬롯 × 교통수단 히트맵"""
    labeled = df[df["transport_mode"] != "unknown"]
    if labeled.empty:
        return

    matrix = build_time_mode_matrix(labeled)
    matrix = matrix.reindex([s for s in TIME_SLOT_ORDER if s in matrix.index])
    matrix = matrix.astype(int)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(matrix, ax=ax, cmap="YlOrRd", annot=True, fmt=",d",
                linewidths=0.5, cbar_kws={"label": "GPS 포인트 수"})
    ax.set_title("시간대 × 교통수단 혼잡도 히트맵", fontsize=13, fontweight="bold")
    ax.set_xlabel("교통수단")
    ax.set_ylabel("시간 슬롯")
    plt.tight_layout()
    save_fig("07_time_mode_heatmap")


def plot_time_mode_line(df: pd.DataFrame):
    """교통수단별 시간대(0~23) 흐름 라인차트"""
    labeled = df[df["transport_mode"] != "unknown"]
    if labeled.empty:
        return

    hourly_mode = (
        labeled.groupby(["hour", "transport_mode"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    for mode in sorted(hourly_mode["transport_mode"].unique()):
        sub = hourly_mode[hourly_mode["transport_mode"] == mode]
        ax.plot(sub["hour"], sub["count"], marker="o", markersize=4,
                label=mode, color=TRANSPORT_PALETTE.get(mode, "#95A5A6"),
                linewidth=2, alpha=0.85)

    ax.set_title("교통수단별 시간대(0~23) 흐름", fontsize=13, fontweight="bold")
    ax.set_xlabel("시간 (시)")
    ax.set_ylabel("GPS 포인트 수")
    ax.set_xticks(range(0, 24))
    ax.legend(title="교통수단", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig("08_time_mode_line")


def plot_mode_ratio_by_slot(df: pd.DataFrame):
    """시간 슬롯별 교통수단 비율 (100% 스택 바)"""
    labeled = df[df["transport_mode"] != "unknown"]
    if labeled.empty:
        return

    matrix = build_time_mode_matrix(labeled)
    matrix = matrix.reindex([s for s in TIME_SLOT_ORDER if s in matrix.index])
    ratio = matrix.div(matrix.sum(axis=1), axis=0) * 100
    colors = [TRANSPORT_PALETTE.get(m, "#95A5A6") for m in ratio.columns]

    fig, ax = plt.subplots(figsize=(12, 5))
    ratio.plot(kind="bar", stacked=True, ax=ax, color=colors, alpha=0.85, edgecolor="white")
    ax.set_title("시간 슬롯별 교통수단 비율", fontsize=13, fontweight="bold")
    ax.set_xlabel("시간 슬롯")
    ax.set_ylabel("비율 (%)")
    ax.legend(title="교통수단", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    save_fig("09_mode_ratio_by_slot")


# ────────────────────────────────────────────
# 6. MovingPandas 속도 기반 혼잡도
# ────────────────────────────────────────────

def plot_speed_by_time(tc: mpd.TrajectoryCollection):
    """
    MovingPandas: 각 Trajectory에 속도를 추가하고
    시간대별 평균 속도 변화 시각화
    """
    print("  속도 계산 중 (MovingPandas)...")
    all_dfs = []
    for traj in tc.trajectories:
        try:
            traj.add_speed(overwrite=True)
            gdf = traj.df.copy()
            if "speed" in gdf.columns and "transport_mode" in gdf.columns:
                gdf["hour"] = gdf.index.hour
                all_dfs.append(gdf[["hour", "speed", "transport_mode"]])
        except Exception:
            continue

    if not all_dfs:
        print("  속도 데이터 없음")
        return

    speed_df = pd.concat(all_dfs, ignore_index=True)
    speed_df = speed_df[speed_df["transport_mode"] != "unknown"]
    speed_df["speed_kmh"] = speed_df["speed"] * 3.6

    # 이상치 제거 (99th percentile 이하)
    q99 = speed_df["speed_kmh"].quantile(0.99)
    speed_df = speed_df[speed_df["speed_kmh"] <= q99]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("MovingPandas 속도 기반 분석", fontsize=14, fontweight="bold")

    # (1) 시간대별 평균 속도
    hourly_speed = speed_df.groupby("hour")["speed_kmh"].mean()
    axes[0].plot(hourly_speed.index, hourly_speed.values, marker="o",
                 color="#E74C3C", linewidth=2.5)
    axes[0].fill_between(hourly_speed.index, hourly_speed.values, alpha=0.15, color="#E74C3C")
    axes[0].set_title("시간대별 평균 속도")
    axes[0].set_xlabel("시간 (시)")
    axes[0].set_ylabel("평균 속도 (km/h)")
    axes[0].set_xticks(range(0, 24))
    axes[0].grid(axis="y", alpha=0.3)

    # (2) 교통수단별 시간대 평균 속도
    for mode in sorted(speed_df["transport_mode"].unique()):
        sub = speed_df[speed_df["transport_mode"] == mode].groupby("hour")["speed_kmh"].mean()
        axes[1].plot(sub.index, sub.values, marker=".", markersize=5,
                     label=mode, color=TRANSPORT_PALETTE.get(mode, "#95A5A6"),
                     linewidth=1.8, alpha=0.85)
    axes[1].set_title("교통수단별 시간대 평균 속도")
    axes[1].set_xlabel("시간 (시)")
    axes[1].set_ylabel("평균 속도 (km/h)")
    axes[1].set_xticks(range(0, 24))
    axes[1].legend(title="교통수단", bbox_to_anchor=(1.01, 1), loc="upper left")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig("10_speed_by_time")
