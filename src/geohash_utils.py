"""
Geohash 인코딩 유틸리티 (MovingPandas 연동)
- 위도/경도 → Geohash 변환
- TrajectoryCollection에 Geohash 부여
- 집계 행렬 생성
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import movingpandas as mpd
import pygeohash as pgh
from typing import Tuple


GEOHASH_PRECISION_SIZE = {
    5: "4.9km × 4.9km",
    6: "1.2km × 0.6km",
    7: "152m × 152m",
    8: "38m × 19m",
}


def encode_geohash(lat: float, lon: float, precision: int = 6) -> str:
    try:
        return pgh.encode(lat, lon, precision=precision)
    except Exception:
        return None


def decode_geohash(geohash: str) -> Tuple[float, float]:
    try:
        lat, lon = pgh.decode(geohash)
        return float(lat), float(lon)
    except Exception:
        return None, None


def add_geohash(df: pd.DataFrame, precision: int = 6) -> pd.DataFrame:
    """DataFrame에 Geohash 컬럼 추가 (numpy vectorize 활용)"""
    df = df.copy()
    _vec = np.vectorize(lambda lat, lon: encode_geohash(lat, lon, precision))
    df["geohash"] = _vec(df["lat"].values, df["lon"].values)
    df["geohash_precision"] = precision
    return df


def add_geohash_to_tc(tc: mpd.TrajectoryCollection, precision: int = 6) -> mpd.TrajectoryCollection:
    """
    TrajectoryCollection의 각 Trajectory에 Geohash 컬럼 추가
    MovingPandas Trajectory 내부 GeoDataFrame을 직접 수정
    """
    for traj in tc.trajectories:
        gdf = traj.df
        gdf["geohash"] = gdf.apply(
            lambda r: encode_geohash(r.geometry.y, r.geometry.x, precision), axis=1
        )
    return tc


def build_flow_matrix(df: pd.DataFrame,
                      geohash_col: str = "geohash",
                      mode_col: str = "transport_mode") -> pd.DataFrame:
    """Geohash × 교통수단별 포인트 수 집계 (피벗)"""
    flow = df.groupby([geohash_col, mode_col]).size().reset_index(name="count")
    pivot = flow.pivot_table(
        index=geohash_col, columns=mode_col, values="count", fill_value=0
    )
    pivot["total"] = pivot.sum(axis=1)
    return pivot.sort_values("total", ascending=False)


def build_time_congestion(df: pd.DataFrame, time_col: str = "hour") -> pd.DataFrame:
    """시간대별 포인트 수"""
    return (
        df.groupby(time_col).size()
        .reset_index(name="count")
        .sort_values(time_col)
    )


def build_time_mode_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """시간 슬롯 × 교통수단 교차 집계"""
    matrix = (
        df.groupby(["time_slot", "transport_mode"])
        .size()
        .reset_index(name="count")
    )
    return matrix.pivot_table(
        index="time_slot", columns="transport_mode", values="count", fill_value=0
    )


def geohash_center_coords(geohashes: pd.Series) -> pd.DataFrame:
    """Geohash 시리즈 → 중심 위도/경도 DataFrame"""
    return geohashes.apply(
        lambda h: pd.Series(decode_geohash(h), index=["lat", "lon"])
    )


def compute_trajectory_stats(tc: mpd.TrajectoryCollection) -> pd.DataFrame:
    """
    TrajectoryCollection에서 궤적별 통계 추출 (MovingPandas 활용)
    - 총 거리, 총 시간, 평균 속도
    """
    records = []
    for traj in tc.trajectories:
        try:
            length_m = traj.get_length()          # meters
            duration_s = traj.get_duration().total_seconds()
            speed = length_m / duration_s if duration_s > 0 else 0
            gdf = traj.df
            mode = gdf["transport_mode"].mode()[0] if "transport_mode" in gdf.columns else "unknown"
            user = gdf["user_id"].iloc[0] if "user_id" in gdf.columns else "unknown"
            records.append({
                "traj_id": traj.id,
                "user_id": user,
                "transport_mode": mode,
                "length_m": length_m,
                "duration_s": duration_s,
                "avg_speed_mps": speed,
                "avg_speed_kmh": speed * 3.6,
                "n_points": len(gdf),
                "start_time": traj.get_start_time(),
                "end_time": traj.get_end_time(),
            })
        except Exception:
            continue
    return pd.DataFrame(records)
