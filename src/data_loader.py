"""
GeoLife 데이터셋 로더 (MovingPandas 기반)
- PLT 파일 파싱
- labels.txt 연동 (교통수단 레이블)
- TrajectoryCollection 생성
"""

import os
import glob
import warnings
import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from pathlib import Path
from shapely.geometry import Point

warnings.filterwarnings("ignore")

PLT_COLS = ["lat", "lon", "zero", "altitude", "date_num", "date", "time"]
WGS84 = "EPSG:4326"


def parse_plt(filepath: str) -> pd.DataFrame:
    """단일 PLT 파일 파싱 (헤더 6줄 스킵)"""
    df = pd.read_csv(
        filepath,
        skiprows=6,
        header=None,
        names=PLT_COLS,
        dtype={"lat": float, "lon": float, "altitude": float,
               "date_num": float, "date": str, "time": str},
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.drop(columns=["zero", "date_num", "date", "time"])
    df["altitude_m"] = df["altitude"] * 0.3048  # feet → meters
    df["file"] = Path(filepath).stem
    return df


def parse_labels(filepath: str) -> pd.DataFrame:
    """labels.txt 파싱"""
    df = pd.read_csv(filepath, sep="\t")
    df.columns = ["start_time", "end_time", "transport_mode"]
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])
    return df


def assign_transport_mode(traj_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """궤적 포인트에 교통수단 레이블 부여 (구간 매칭)"""
    traj_df = traj_df.copy()
    traj_df["transport_mode"] = "unknown"
    for _, row in labels_df.iterrows():
        mask = (traj_df["datetime"] >= row["start_time"]) & \
               (traj_df["datetime"] <= row["end_time"])
        traj_df.loc[mask, "transport_mode"] = row["transport_mode"]
    return traj_df


def load_user_df(user_dir: str) -> pd.DataFrame:
    """단일 사용자 전체 궤적을 DataFrame으로 로드 + 레이블 부여"""
    user_dir = Path(user_dir)
    user_id = user_dir.name

    plt_files = sorted(glob.glob(str(user_dir / "Trajectory" / "*.plt")))
    if not plt_files:
        return pd.DataFrame()

    dfs = []
    for f in plt_files:
        try:
            dfs.append(parse_plt(f))
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    traj = pd.concat(dfs, ignore_index=True)
    traj["user_id"] = user_id

    labels_path = user_dir / "labels.txt"
    if labels_path.exists():
        labels = parse_labels(str(labels_path))
        traj = assign_transport_mode(traj, labels)
    else:
        traj["transport_mode"] = "unknown"

    return traj


def df_to_trajectory_collection(df: pd.DataFrame) -> mpd.TrajectoryCollection:
    """
    DataFrame → MovingPandas TrajectoryCollection 변환
    사용자(user_id) × 파일(file) 단위로 궤적 분리
    """
    df = df.copy()
    df = df.sort_values("datetime")

    # GeoDataFrame 생성
    geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=WGS84)
    gdf = gdf.set_index("datetime")

    # traj_id = user_id + "_" + file
    gdf["traj_id"] = gdf["user_id"] + "_" + gdf["file"]

    tc = mpd.TrajectoryCollection(gdf, traj_id_col="traj_id", min_length=10)
    return tc


def load_all_users(data_dir: str, labeled_only: bool = False,
                   max_users: int = None) -> pd.DataFrame:
    """
    전체 사용자 DataFrame 로드

    Args:
        data_dir: data 폴더 경로
        labeled_only: labels.txt 있는 사용자만 로드
        max_users: 최대 로드 사용자 수 (None=전체)
    """
    data_dir = Path(data_dir)
    user_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if labeled_only:
        user_dirs = [d for d in user_dirs if (d / "labels.txt").exists()]
    if max_users:
        user_dirs = user_dirs[:max_users]

    print(f"로딩 중: {len(user_dirs)}명의 사용자 데이터...")
    all_dfs = []
    for i, user_dir in enumerate(user_dirs):
        df = load_user_df(str(user_dir))
        if not df.empty:
            all_dfs.append(df)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(user_dirs)} 완료")

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    print(f"총 {len(result):,}개 포인트 로드 완료")
    return result


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 파생 변수 추가"""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["weekday_name"] = df["datetime"].dt.day_name()
    df["is_weekend"] = df["weekday"] >= 5
    df["time_slot"] = pd.cut(
        df["hour"],
        bins=[0, 6, 9, 12, 14, 18, 21, 24],
        labels=["심야(0-6)", "출근(6-9)", "오전(9-12)", "점심(12-14)",
                "오후(14-18)", "저녁(18-21)", "야간(21-24)"],
        right=False,
        include_lowest=True,
    )
    return df
