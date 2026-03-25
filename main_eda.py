"""
EDA 실행 진입점 - MovingPandas 기반
Usage: python main_eda.py

생성 파일 목록 (outputs/figures/):
  01_data_overview.png
  02_trajectory_stats.png
  03_geohash_flow_by_mode.png
  04_geohash_density_heatmap.png
  05_hourly_congestion.png
  06_weekday_congestion.png
  07_time_mode_heatmap.png
  08_time_mode_line.png
  09_mode_ratio_by_slot.png
  10_speed_by_time.png
"""

import sys
import warnings
import matplotlib
matplotlib.use("Agg")  # GUI 없이 파일로 저장
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_all_users, add_time_features, df_to_trajectory_collection
from geohash_utils import add_geohash
from eda import (
    print_summary,
    plot_data_overview,
    plot_trajectory_stats,
    plot_geohash_flow,
    plot_geohash_density,
    plot_hourly_congestion,
    plot_weekday_congestion,
    plot_time_mode_heatmap,
    plot_time_mode_line,
    plot_mode_ratio_by_slot,
    plot_speed_by_time,
)

DATA_DIR        = Path(__file__).parent / "data"
GEOHASH_PREC    = 6        # ~1.2km × 0.6km 격자
MAX_USERS       = None     # None=전체
SAMPLE_EVERY_N  = 5        # EDA용 샘플링: 5포인트 중 1개 (속도 향상)
# TrajectoryCollection은 레이블 있는 사용자만 사용 (메모리 절약)
TC_LABELED_ONLY = True


def main():
    # ── 1. 데이터 로드 ─────────────────────────────
    print("\n[1/5] 데이터 로드 중...")
    df = load_all_users(str(DATA_DIR), labeled_only=False, max_users=MAX_USERS)

    # ── 2. 전처리 ──────────────────────────────────
    print("\n[2/5] 전처리 중...")
    # EDA용 샘플링 (패턴 분석에 충분)
    if SAMPLE_EVERY_N > 1:
        df = df.iloc[::SAMPLE_EVERY_N].reset_index(drop=True)
        print(f"  샘플링 (1/{SAMPLE_EVERY_N}): {len(df):,}개 포인트")

    df = add_time_features(df)
    df = add_geohash(df, precision=GEOHASH_PREC)

    # 이상치 제거: 베이징 권역 외 좌표 제거
    before = len(df)
    df = df[(df["lat"].between(30, 55)) & (df["lon"].between(100, 135))]
    print(f"  이상치 제거: {before:,} -> {len(df):,}개 포인트")

    print_summary(df)

    # ── 3. MovingPandas TrajectoryCollection 생성 (레이블 사용자만) ──
    print("\n[3/5] MovingPandas TrajectoryCollection 생성 중...")
    if TC_LABELED_ONLY:
        df_tc = df[df["transport_mode"] != "unknown"]
        print(f"  레이블 데이터만 사용: {len(df_tc):,}개 포인트")
    else:
        df_tc = df
    tc = df_to_trajectory_collection(df_tc)
    print(f"  총 {len(tc.trajectories)}개 궤적 생성")

    # ── 4. EDA 시각화 ──────────────────────────────
    print("\n[4/5] EDA 시각화 생성 중...")

    print("\n  [개요]")
    plot_data_overview(df)

    print("\n  [궤적 통계 - MovingPandas]")
    stats = plot_trajectory_stats(tc)

    print("\n  [Geohash 흐름]")
    plot_geohash_flow(df, top_n=30)
    plot_geohash_density(df)

    print("\n  [시간대별 혼잡도]")
    plot_hourly_congestion(df)
    plot_weekday_congestion(df)

    print("\n  [시간대 × 교통수단 혼잡도]")
    plot_time_mode_heatmap(df)
    plot_time_mode_line(df)
    plot_mode_ratio_by_slot(df)

    print("\n  [속도 기반 분석 - MovingPandas]")
    plot_speed_by_time(tc)

    # ── 5. 완료 ────────────────────────────────────
    print("\n[5/5] 완료!")
    print(f"  시각화 저장 위치: {Path('outputs/figures').resolve()}")

    return df, tc, stats


if __name__ == "__main__":
    df, tc, stats = main()
