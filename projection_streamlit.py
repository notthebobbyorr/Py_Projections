from __future__ import annotations

import json
import os
import posixpath
import runpy
import re
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colors

LEVEL_LABELS = {
    1: "MLB",
    11: "Triple-A",
    14: "Low-A",
    16: "Low Minors",
}
BP_SOURCE_LEVEL_LABELS = {
    "1": "MLB",
    "2": "Triple-A",
    "3": "Double-A",
    "4": "High-A",
    "5": "Low-A",
}
DEFAULT_NO_FORMAT_COLS = {
    "source_season",
    "target_season",
    "level_id_source",
    "age",
    "age_used",
    "batter_mlbid",
    "pitcher_mlbid",
    "mlbid",
    "ADP",
}
HIGHER_IS_WORSE_TOKENS = (
    "strikeout",
    "k%",
    "cs",
    "chase",
    "hittable_pitches_taken",
    "secondary_whiff",
    "whiffs_vs_95",
    "la_lte_0",
    "ball",
    "era",
    "whip",
    "hr/9",
    "hr/bbe",
)
TEAM_COL_CANDIDATES = ("team", "team_abbreviations", "team_abbreviation")
POSITION_FILTER_COLS = ["UT", "C", "X1B", "X2B", "X3B", "SS", "OF", "P"]
POSITION_FILTER_LABELS = {
    "UT": "UT",
    "C": "C",
    "X1B": "1B",
    "X2B": "2B",
    "X3B": "3B",
    "SS": "SS",
    "OF": "OF",
    "P": "P",
}
POSITION_COUNT_THRESHOLD = 20
COUNTING_ONE_DEC_BASES = {
    "pa",
    "ab",
    "bbe",
    "tbf",
    "h",
    "hr",
    "g",
    "gs",
    "so",
    "w",
    "sv",
    "bb",
    "hbp",
    "sb",
    "cs",
    "sba",
    "r",
    "runs",
    "rbi",
    "g_marcel",
    "gs_marcel",
    "tbf_marcel",
    "points",
}
TWO_DEC_BASES = {"era", "whip", "points/pa", "points/ip", "sgp", "sgp_hit", "sgp_pitch"}
SGP_DENOMINATORS = {
    "R": 16.856119,
    "HR": 7.491698,
    "RBI": 18.370355,
    "SB": 6.551795,
    "AVG": 0.001353,
    "W": 2.589256,
    "SV": 4.629408,
    "SO": 30.233919,
    "ERA": 0.070897,
    "WHIP": 0.011483,
}
SGP_BASELINES = {
    "AB0": 7085.24,
    "H0": 1786.51,
    "AVG0": 0.2521448705170637,
    "IP0": 1251.01,
    "ER0": 535.28,
    "BBH0": 1522.27,
    "ERA0": 3.8508695225771272,
    "WHIP0": 1.216830493419272,
}
ONE_DEC_PERCENT_BASES = {"k%", "bb%", "k-bb%", "k_pct", "bb_pct", "ip", "hr/9", "hr/bbe%", "gb%", "whiff%", "swstr%", "ip_marcel"}
FLIPPED_PERCENTILE_BASES = {"K%", "CS"}
PITCHING_FLIPPED_PERCENTILE_BASES = {"ERA", "BB%", "BABIP", "WHIP"}
THREE_DEC_SLASH_TOKENS = ("avg", "obp", "slg", "ops", "babip_recalc_rate", "babip")
PITCHER_KPI_ZERO_DEC_BASES = {"stuff_raw", "stuff_raw_reg", "bb_rpm_reg", "bb_rpm"}
PITCHER_KPI_ONE_DEC_BASES = {
    "fastball_velo",
    "fastball_velo_reg",
    "max_velo",
    "max_velo_reg",
    "fastball_vaa",
    "fastball_vaa_reg",
    "fa_pct",
    "fa_pct_reg",
    "swstr",
    "swstr_reg",
    "zone",
    "zone_reg",
    "ball_pct",
    "ball_pct_reg",
    "z_contact",
    "z_contact_reg",
    "chase",
    "chase_reg",
    "csw",
    "csw_reg",
    "la_lte_0",
    "la_lte_0_reg",
    "la_gte_20",
    "la_gte_20_reg",
    "rel_z",
    "rel_z_reg",
    "rel_x",
    "rel_x_reg",
    "ext",
    "ext_reg",
}
KPI_ONE_DEC_TOKENS = (
    "damage",
    "ev",
    "seager",
    "pull_",
    "la_gte_20",
    "la_lte_0",
    "selection_skill",
    "hittable_pitches_taken",
    "chase",
    "z_con",
    "secondary_whiff",
    "whiffs_vs_95",
    "contact_vs_avg",
    "sweet_spot",
    "barrel",
    "hard_hit",
    "xwoba",
)
COMPONENT_SLASH_PREDICTIONS_PATH = Path(
    "projection_outputs/component_slash_models/component_slash_hitter_predictions.parquet"
)
COMPONENT_SLASH_KPI_PROJECTIONS_PATH = Path(
    "projection_outputs/component_slash_models/component_slash_from_kpi_projections_2026.parquet"
)
COMPONENT_METRICS = {
    "AVG": "AVG",
    "OBP": "OBP",
    "SLG": "SLG",
    "OPS": "OPS",
    "K%": "K_pct",
    "BB%": "BB_pct",
}
COMPONENT_KPI_METRICS = {
    "AVG": "component_AVG",
    "OBP": "component_OBP",
    "SLG": "component_SLG",
    "OPS": "component_OPS",
    "K%": "component_K_pct",
    "BB%": "component_BB_pct",
}
KPI_COMPARE_BASES = {
    "AVG": "AVG",
    "OBP": "OBP",
    "SLG": "SLG",
    "OPS": "OPS",
    "K%": "SO_per_PA_pct",
    "BB%": "BB_per_PA_pct",
}
P50_DISPLAY_ALIASES = {
    "PA": "PA",
    "SGP": "SGP",
    "SGP_hit": "SGP Hit",
    "SGP_pitch": "SGP Pitch",
    "AB": "AB",
    "H": "H",
    "HR": "HR",
    "Runs": "R",
    "RBI": "RBI",
    "SB": "SB",
    "CS": "CS",
    "K%": "K%",
    "BB%": "BB%",
    "AVG": "AVG",
    "OBP": "OBP",
    "SLG": "SLG",
    "OPS": "OPS",
    "babip_recalc_rate_mlb_eq_non_ar_delta": "BABIP",
    "ISO": "ISO",
    "G_marcel": "G",
    "GS_marcel": "GS",
    "IP_marcel": "IP",
    "TBF_marcel": "TBF",
    "stuff_raw": "Pitch Grade",
    "stuff_raw_reg": "Pitch Grade",
    "fastball_velo_reg": "FA mph",
    "max_velo_reg": "Max FA mph",
    "fastball_vaa_reg": "FA VAA",
    "FA_pct_reg": "FA Usage (%)",
    "BB_rpm_reg": "BB Spin",
    "SwStr_reg": "SwStr (%)",
    "Zone_reg": "Zone (%)",
    "Ball_pct_reg": "Ball (%)",
    "Z_Contact_reg": "Z-Contact (%)",
    "Chase_reg": "Chase (%)",
    "CSW_reg": "CSW (%)",
    "LA_gte_20_reg": "LA>=20%",
    "LA_lte_0_reg": "LA<=0%",
    "rel_z_reg": "Vertical Release (ft.)",
    "rel_x_reg": "Horizontal Release (ft.)",
    "ext_reg": "Extension (ft.)",
}
DISPLAY_COL_ALIASES = {
    "age_used": "age",
    "opening_day_status_40man": "Opening Day Status",
    "is_mlb_ip_source": "MLB IP Source",
    "is_mlb_pa_source": "MLB PA Source",
    "is_mlb_sb_source": "MLB SB Source",
}
DISPLAY_ROUNDING_BASE_OVERRIDES = {
    "pitch grade": "stuff_raw",
    "bb spin": "bb_rpm_reg",
    "fa mph": "fastball_velo_reg",
    "max fa mph": "max_velo_reg",
    "fa vaa": "fastball_vaa_reg",
    "fa usage (%)": "fa_pct_reg",
    "fa usage": "fa_pct_reg",
    "swstr (%)": "swstr_reg",
    "swstr": "swstr_reg",
    "zone (%)": "zone_reg",
    "zone": "zone_reg",
    "ball (%)": "ball_pct_reg",
    "ball": "ball_pct_reg",
    "z-contact (%)": "z_contact_reg",
    "z-contact": "z_contact_reg",
    "chase (%)": "chase_reg",
    "chase": "chase_reg",
    "csw (%)": "csw_reg",
    "csw": "csw_reg",
    "la<=0%": "la_lte_0_reg",
    "la<=0": "la_lte_0_reg",
    "la>=20%": "la_gte_20_reg",
    "la>=20": "la_gte_20_reg",
    "vertical release (ft.)": "rel_z_reg",
    "vertical release": "rel_z_reg",
    "horizontal release (ft.)": "rel_x_reg",
    "horizontal release": "rel_x_reg",
    "extension (ft.)": "ext_reg",
    "extension": "ext_reg",
}
MAX_STYLED_CELLS = 25_000
MAX_STYLED_ROWS = 300
MLB_PA_REFERENCE_MIN = 200.0
FORTY_MAN_ROSTER_PATH = Path("40man.xlsx")
FORTY_MAN_ROSTER_CSV_FALLBACK_PATH = Path("rr_roster.csv")
FORTY_MAN_ROLE_COL = "Projected Opening Day Status"
FORTY_MAN_TEAM_COL = "Team"
FORTY_MAN_ID_COL = "MLBAMID"
FORTY_MAN_POS_COL = "Pos"
FORTY_MAN_PITCHER_POS_TOKENS = {"P", "SP", "RP"}
ROLE_SAVE_ELIGIBLE = {"CLOSER", "SETUP MAN", "MIDDLE RELIEVER"}
ROLE_CLOSER = "CLOSER"
ROLE_SETUP = "SETUP MAN"
ROLE_MIDDLE = "MIDDLE RELIEVER"
ROLE_LONG = "LONG RELIEVER"
TEAM_IP_MIN = 1400.0
TEAM_IP_MAX = 1470.0
TEAM_IP_TARGET_MU = 1435.0
TEAM_IP_TARGET_SD = 17.5
TEAM_GS_TARGET = 162.0
TEAM_PA_MIN = 5900.0
TEAM_PA_MAX = 6300.0
TEAM_PA_TARGET_MU = 6100.0
TEAM_PA_TARGET_SD = 100.0
TEAM_SV_MEAN = 40.7
TEAM_SV_STD = 6.7
MLB_SB_TARGET_2025 = 3440.0
MLB_SBA_TARGET_2025 = 4429.0
SV_QUANTILE_Z = {
    "p20": -0.8416212335729143,
    "p25": -0.6744897501960817,
    "p50": 0.0,
    "p75": 0.6744897501960817,
    "p80": 0.8416212335729143,
}
BP_HITTING_MLB_PA_LOOKUP_PATH = Path(
    "projection_outputs/bp_hitting_api/bp_hitting_table_with_level_id.parquet"
)
HITTER_POSITION_SOURCE_PARQUET_PATH = Path("damage_pos_2015_2025.parquet")
HITTER_POSITION_SOURCE_CSV_PATH = Path("damage_pos_2015_2025.csv")
ZSPACE_REFERENCE_MLB_PA_GTE_100_PATH = Path(
    "projection_outputs/sandbox/zspace_reference_mu_sigma_by_metric_season_mlb_pa_gte_100.csv"
)
BP_PITCHING_MLB_LOOKUP_PATH = Path(
    "projection_outputs/bp_pitching_api/bp_pitching_table_with_level_id.parquet"
)
MLB_IP_REFERENCE_MIN = 20.0
ERA_FROM_ERIP_FACTOR = 8.28  # 9 * 0.92
STUFF_GRADE_REF_SEASON = 2025
STUFF_GRADE_REF_LEVEL_ID = 1
STUFF_GRADE_REF_PATHS = (
    Path("projection_outputs/lb2_refresh/pitcher_stuff_new_with_lb2_meta.parquet"),
    Path("pitcher_stuff_new.parquet"),
)
ADP_PATH = Path("ADP.tsv")
ADP_DRAFT_TEAMS = 15
TEAM_ABBREV_NORMALIZATION = {
    "ARI": "ARI",
    "ARZ": "ARI",
    "AZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC",
    "CHW": "CWS",
    "CWS": "CWS",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KCR": "KC",
    "KC": "KC",
    "LAA": "LAA",
    "ANA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA",
    "FLA": "MIA",
    "MIL": "MIL",
    "MLW": "MIL",
    "MIN": "MIN",
    "NYM": "NYM",
    "NYY": "NYY",
    "ATH": "ATH",
    "OAK": "ATH",
    "PHI": "PHI",
    "PIT": "PIT",
    "SDP": "SD",
    "SD": "SD",
    "SEA": "SEA",
    "SFG": "SF",
    "SF": "SF",
    "STL": "STL",
    "TBR": "TB",
    "TB": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSN": "WSH",
    "WAS": "WSH",
    "WSH": "WSH",
    "FA": "FA",
}
POSITION_TOKEN_NORMALIZATION = {
    "X1B": "1B",
    "X2B": "2B",
    "X3B": "3B",
    "LF": "OF",
    "CF": "OF",
    "RF": "OF",
    "SP": "P",
    "RP": "P",
    "DH": "UT",
}
ROSTER_MANAGER_SAVE_PATH = (
    Path(__file__).resolve().parent
    / "projection_outputs"
    / "sandbox"
    / "roster_manager_saved_teams.json"
)
ROSTER_MANAGER_SCHEMA_VERSION = 2
ROSTER_PERCENTILE_OPTIONS = ("p25", "p50", "p75")
ROSTER_PLACEHOLDER_HITTER_POSITIONS = ("C", "1B", "2B", "3B", "SS", "OF")
ROSTER_PLACEHOLDER_PITCHER_ROLES = ("SP", "RP")
ROSTER_HITTER_STARTER_SLOTS = [
    "C1",
    "C2",
    "1B",
    "2B",
    "SS",
    "3B",
    "OF1",
    "OF2",
    "OF3",
    "OF4",
    "OF5",
    "MI",
    "CI",
    "UT",
]
ROSTER_PITCHER_STARTER_SLOTS = [
    "SP1",
    "SP2",
    "SP3",
    "SP4",
    "SP5",
    "SP6",
    "SP7/RP3",
    "RP1",
    "RP2",
]
ROSTER_ALL_STARTER_SLOTS = [*ROSTER_HITTER_STARTER_SLOTS, *ROSTER_PITCHER_STARTER_SLOTS]
ROSTER_RESERVE_SLOTS = 15  # fixed number of reserve rows shown in the UI
ROSTER_HITTER_STATS = ["HR", "R", "RBI", "SB", "AVG", "AB", "H"]
ROSTER_PITCHER_STATS = ["W", "K", "SV", "WHIP", "ERA", "IP", "H", "BB", "ER"]
ROSTER_HITTER_MANUAL_COUNT_STATS = ["HR", "R", "RBI", "SB", "AB", "H"]
ROSTER_PITCHER_MANUAL_COUNT_STATS = ["W", "K", "SV", "IP", "H", "BB", "ER"]
ROSTER_HITTER_BENCHMARK_TOTAL = {
    "HR": 303.0,
    "R": 1043.0,
    "RBI": 1021.0,
    "SB": 181.0,
    "AVG": 0.257,
    "AB": 7441.0,
}
ROSTER_HITTER_BENCHMARK_PER_SPOT = {
    "HR": 20.0,
    "R": 73.0,
    "RBI": 71.0,
    "SB": 15.0,
    "AVG": 0.257,
}
ROSTER_PITCHER_BENCHMARK_TOTAL = {
    "W": 89.0,
    "K": 1345.0,
    "SV": 66.0,
    "WHIP": 1.173,
    "ERA": 3.572,
    "IP": 1359.33,
    "H": 1222.0,
    "BB": 444.0,
    "ER": 589.0,
}
ROSTER_PITCHER_BENCHMARK_PER_SPOT = {
    "W": 12.0,
    "K": 150.0,
    "SV": 36.0,
    "WHIP": 1.173,
    "ERA": 3.572,
}
ROSTER_HITTER_TO_PROJ_BASE = {
    "HR": "HR",
    "R": "Runs",
    "RBI": "RBI",
    "SB": "SB",
    "AVG": "AVG",
    "AB": "AB",
    "H": "H",
}
ROSTER_PITCHER_TO_PROJ_BASE = {
    "W": "W",
    "K": "SO",
    "SV": "SV",
    "WHIP": "WHIP",
    "ERA": "ERA",
    "IP": "IP",
    "H": "H",
    "BB": "BB",
}


def load_projection(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def _load_stuff_grade_reference_2025_mlb() -> tuple[float, float]:
    for path in STUFF_GRADE_REF_PATHS:
        if not path.exists():
            continue
        try:
            raw = pd.read_parquet(path, columns=["season", "level_id", "stuff_raw"])
        except Exception:
            continue
        if raw.empty:
            continue
        season = pd.to_numeric(raw.get("season"), errors="coerce")
        level = pd.to_numeric(raw.get("level_id"), errors="coerce")
        stuff = pd.to_numeric(raw.get("stuff_raw"), errors="coerce")
        mask = (
            season.eq(float(STUFF_GRADE_REF_SEASON))
            & level.eq(float(STUFF_GRADE_REF_LEVEL_ID))
            & stuff.notna()
            & np.isfinite(stuff)
        )
        ref = stuff.loc[mask].dropna()
        if len(ref) < 20:
            continue
        p01 = float(ref.quantile(0.01))
        p99 = float(ref.quantile(0.99))
        if np.isfinite(p01) and np.isfinite(p99) and (abs(p99 - p01) > 1e-12):
            return p01, p99
    return np.nan, np.nan


def _excel_col_to_idx(col: str) -> int:
    out = 0
    for ch in col:
        out = (out * 26) + (ord(ch) - 64)
    return out - 1


def _xlsx_member_path(target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return posixpath.normpath(f"xl/{target}")


def _load_xlsx_first_sheet_fallback(path: Path) -> pd.DataFrame:
    xml_ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    cell_ref_re = re.compile(r"([A-Z]+)[0-9]+")
    with ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib.get("Id"): rel.attrib.get("Target") for rel in rels}
        sheets = workbook.find("x:sheets", xml_ns)
        if sheets is None or not list(sheets):
            return pd.DataFrame()
        first_sheet = list(sheets)[0]
        rel_id = first_sheet.attrib.get(f"{rel_ns}id")
        if not rel_id:
            return pd.DataFrame()
        target = rel_map.get(rel_id, "")
        if not target:
            return pd.DataFrame()
        sheet_member = _xlsx_member_path(target)

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst.findall("x:si", xml_ns):
                shared_strings.append(
                    "".join(node.text or "" for node in si.findall(".//x:t", xml_ns))
                )

        sheet = ET.fromstring(zf.read(sheet_member))
        rows = sheet.findall(".//x:sheetData/x:row", xml_ns)
        if not rows:
            return pd.DataFrame()

        row_maps: list[dict[int, str]] = []
        for row in rows:
            row_map: dict[int, str] = {}
            for cell in row.findall("x:c", xml_ns):
                ref = cell.attrib.get("r", "")
                match = cell_ref_re.match(ref)
                if not match:
                    continue
                col_idx = _excel_col_to_idx(match.group(1))
                cell_type = cell.attrib.get("t")
                val_node = cell.find("x:v", xml_ns)
                if val_node is not None:
                    raw = val_node.text or ""
                    if cell_type == "s":
                        try:
                            s_idx = int(raw)
                        except ValueError:
                            val = raw
                        else:
                            val = (
                                shared_strings[s_idx]
                                if 0 <= s_idx < len(shared_strings)
                                else raw
                            )
                    else:
                        val = raw
                else:
                    inline = cell.find("x:is", xml_ns)
                    if inline is not None:
                        val = "".join(
                            node.text or "" for node in inline.findall(".//x:t", xml_ns)
                        )
                    else:
                        val = ""
                row_map[col_idx] = val
            if row_map:
                row_maps.append(row_map)
    if not row_maps:
        return pd.DataFrame()
    width = max(row_maps[0].keys()) + 1
    headers = [str(row_maps[0].get(i, "")).strip() for i in range(width)]
    headers = [h if h else f"column_{i+1}" for i, h in enumerate(headers)]
    records = [[row_map.get(i, "") for i in range(width)] for row_map in row_maps[1:]]
    return pd.DataFrame(records, columns=headers)


@st.cache_resource(show_spinner=False)
def _load_40man_roster_raw() -> pd.DataFrame:
    if FORTY_MAN_ROSTER_PATH.exists():
        try:
            return pd.read_excel(FORTY_MAN_ROSTER_PATH)
        except Exception:
            try:
                return _load_xlsx_first_sheet_fallback(FORTY_MAN_ROSTER_PATH)
            except Exception:
                pass
    if FORTY_MAN_ROSTER_CSV_FALLBACK_PATH.exists():
        try:
            return pd.read_csv(FORTY_MAN_ROSTER_CSV_FALLBACK_PATH)
        except Exception:
            pass
    return pd.DataFrame()


def _is_pitcher_pos(pos: object) -> bool:
    text = str(pos or "").strip().upper()
    if not text:
        return False
    toks = [t.strip() for t in text.replace("-", "/").split("/") if t.strip()]
    return any(tok in FORTY_MAN_PITCHER_POS_TOKENS for tok in toks)


@st.cache_resource(show_spinner=False)
def _load_40man_lookup(*, for_pitchers: bool) -> pd.DataFrame:
    roster = _load_40man_roster_raw()
    required_cols = {FORTY_MAN_ID_COL, FORTY_MAN_TEAM_COL, FORTY_MAN_ROLE_COL}
    if roster.empty or not required_cols.issubset(set(roster.columns)):
        return pd.DataFrame(columns=["mlbid", "team_40man", "opening_day_status_40man"])

    work = roster.copy()
    work["mlbid"] = pd.to_numeric(work[FORTY_MAN_ID_COL], errors="coerce").astype("Int64")
    work["team_40man"] = (
        work[FORTY_MAN_TEAM_COL].fillna("").astype(str).str.strip().str.upper()
    )
    work["opening_day_status_40man"] = (
        work[FORTY_MAN_ROLE_COL].fillna("").astype(str).str.strip()
    )
    pos_series = (
        work[FORTY_MAN_POS_COL]
        if FORTY_MAN_POS_COL in work.columns
        else pd.Series("", index=work.index)
    )
    pref_pitcher = pos_series.map(_is_pitcher_pos)
    work["_pref"] = pref_pitcher if for_pitchers else (~pref_pitcher)
    work["_score"] = (
        work["_pref"].astype("int64") * 10
        + work["opening_day_status_40man"].ne("").astype("int64")
        + work["team_40man"].ne("").astype("int64")
    )
    work = work[work["mlbid"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["mlbid", "team_40man", "opening_day_status_40man"])
    work = work.sort_values(["mlbid", "_score"], ascending=[True, False])
    out = work.drop_duplicates(subset=["mlbid"], keep="first")[
        ["mlbid", "team_40man", "opening_day_status_40man"]
    ].copy()
    return out


def _attach_40man_context(df: pd.DataFrame, *, for_pitchers: bool) -> pd.DataFrame:
    if df.empty:
        return df
    id_col = None
    if "mlbid" in df.columns:
        id_col = "mlbid"
    elif for_pitchers and "pitcher_mlbid" in df.columns:
        id_col = "pitcher_mlbid"
    elif (not for_pitchers) and "batter_mlbid" in df.columns:
        id_col = "batter_mlbid"
    elif "pitcher_mlbid" in df.columns:
        id_col = "pitcher_mlbid"
    elif "batter_mlbid" in df.columns:
        id_col = "batter_mlbid"
    if id_col is None:
        return df
    lookup = _load_40man_lookup(for_pitchers=for_pitchers)
    if lookup.empty:
        return df
    out = df.copy()
    out["_mlbid_join"] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
    out = out.merge(
        lookup.rename(columns={"mlbid": "_mlbid_join"}),
        on="_mlbid_join",
        how="left",
    )
    for team_col in TEAM_COL_CANDIDATES:
        if team_col in out.columns:
            cur = out[team_col]
            out[team_col] = np.where(
                out["team_40man"].notna() & out["team_40man"].ne(""),
                out["team_40man"],
                cur,
            )
            break
    out = out.drop(columns=["_mlbid_join"], errors="ignore")
    return out


def _apply_pitcher_role_overrides_edit_mode(
    df: pd.DataFrame,
    *,
    key_prefix: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    if out.empty:
        return out, {"overrides_active": 0, "rows_overridden": 0}
    id_col = "pitcher_mlbid" if "pitcher_mlbid" in out.columns else ("mlbid" if "mlbid" in out.columns else None)
    if id_col is None:
        st.info("Edit mode unavailable: missing pitcher id column.")
        return out, {"overrides_active": 0, "rows_overridden": 0}
    if "opening_day_status_40man" not in out.columns:
        out = out.assign(opening_day_status_40man="")

    editor_key = f"{key_prefix}_pitcher_role_overrides"
    existing = st.session_state.get(editor_key, {})
    if not isinstance(existing, dict):
        existing = {}
    overrides: dict[str, str] = {
        str(k): str(v).strip()
        for k, v in existing.items()
        if str(k).strip() and str(v).strip()
    }

    name_col = "name" if "name" in out.columns else ("player_name" if "player_name" in out.columns else None)
    team_col = _team_col(out)

    candidates = out.copy()
    candidates["id_int"] = pd.to_numeric(candidates[id_col], errors="coerce").astype("Int64")
    candidates = candidates[candidates["id_int"].notna()].copy()
    if candidates.empty:
        st.info("Edit mode unavailable: no pitchers with valid ids.")
        return out, {"overrides_active": len(overrides), "rows_overridden": 0}

    keep_cols = ["id_int", "opening_day_status_40man"]
    if name_col:
        keep_cols.append(name_col)
    if team_col:
        keep_cols.append(team_col)
    candidates = candidates[keep_cols].drop_duplicates(subset=["id_int"], keep="first").copy()

    role_values_df = (
        out["opening_day_status_40man"].dropna().astype(str).str.strip().tolist()
        if "opening_day_status_40man" in out.columns
        else []
    )
    lookup = _load_40man_lookup(for_pitchers=True)
    role_values_lookup = (
        lookup["opening_day_status_40man"].dropna().astype(str).str.strip().tolist()
        if not lookup.empty and "opening_day_status_40man" in lookup.columns
        else []
    )
    role_options = sorted(
        {
            v
            for v in (role_values_df + role_values_lookup)
            if v and v.lower() != "nan"
        }
    )
    if not role_options:
        role_options = ["Closer", "Setup Man", "Middle Reliever", "Long Reliever", "Starting Rotation"]

    label_by_key: dict[str, str] = {}
    current_role_by_key: dict[str, str] = {}
    for row in candidates.itertuples(index=False):
        pid = getattr(row, "id_int")
        if pd.isna(pid):
            continue
        key = str(int(pid))
        nm = str(getattr(row, name_col) if name_col else f"Pitcher {key}").strip() if name_col else f"Pitcher {key}"
        tm = str(getattr(row, team_col) if team_col else "").strip().upper() if team_col else ""
        cur_role = str(getattr(row, "opening_day_status_40man", "") or "").strip()
        current_role_by_key[key] = cur_role
        if tm:
            label_by_key[key] = f"{nm} ({tm}) [{key}]"
        else:
            label_by_key[key] = f"{nm} [{key}]"

    picker_keys = sorted(label_by_key.keys(), key=lambda k: label_by_key[k])
    if not picker_keys:
        return out, {"overrides_active": len(overrides), "rows_overridden": 0}
    picker_labels = [label_by_key[k] for k in picker_keys]
    selected_label = st.selectbox(
        "Edit mode: pitcher",
        picker_labels,
        index=0,
        key=f"{key_prefix}_edit_mode_pitcher_picker",
    )
    selected_key = picker_keys[picker_labels.index(selected_label)]

    role_placeholder = "(Clear role)"
    current_effective = overrides.get(selected_key, current_role_by_key.get(selected_key, ""))
    role_index = 0
    if current_effective in role_options:
        role_index = 1 + role_options.index(current_effective)
    selected_role = st.selectbox(
        "Edit mode: override Opening Day role",
        [role_placeholder] + role_options,
        index=role_index,
        key=f"{key_prefix}_edit_mode_pitcher_role_value",
        help="Overrides role for this pitcher in the current app session.",
    )
    edit_cols = st.columns(3)
    with edit_cols[0]:
        if st.button("Apply override", key=f"{key_prefix}_edit_mode_apply_role"):
            if selected_role == role_placeholder:
                overrides.pop(selected_key, None)
            else:
                overrides[selected_key] = selected_role
            st.session_state[editor_key] = overrides
            st.rerun()
    with edit_cols[1]:
        if st.button("Clear selected", key=f"{key_prefix}_edit_mode_clear_selected"):
            overrides.pop(selected_key, None)
            st.session_state[editor_key] = overrides
            st.rerun()
    with edit_cols[2]:
        if st.button("Clear all overrides", key=f"{key_prefix}_edit_mode_clear_all"):
            st.session_state[editor_key] = {}
            st.rerun()

    if overrides:
        rows = []
        for key, role in sorted(overrides.items(), key=lambda kv: label_by_key.get(kv[0], kv[0])):
            rows.append(
                {
                    "Pitcher": label_by_key.get(key, key),
                    "Override Opening Day Role": role,
                    "Roster Role": current_role_by_key.get(key, ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    id_keys = pd.to_numeric(out[id_col], errors="coerce").astype("Int64").astype("string")
    mapped_roles = id_keys.map(overrides)
    override_mask = mapped_roles.notna()
    if bool(override_mask.any()):
        out.loc[override_mask, "opening_day_status_40man"] = mapped_roles.loc[override_mask].astype(str)

    return out, {"overrides_active": len(overrides), "rows_overridden": int(override_mask.sum())}


def _apply_hitter_role_overrides_edit_mode(
    df: pd.DataFrame,
    *,
    key_prefix: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    if out.empty:
        return out, {"overrides_active": 0, "rows_overridden": 0}
    id_col = (
        "hitter_mlbid"
        if "hitter_mlbid" in out.columns
        else ("batter_mlbid" if "batter_mlbid" in out.columns else ("mlbid" if "mlbid" in out.columns else None))
    )
    if id_col is None:
        st.info("Edit mode unavailable: missing hitter id column.")
        return out, {"overrides_active": 0, "rows_overridden": 0}
    if "opening_day_status_40man" not in out.columns:
        out = out.assign(opening_day_status_40man="")

    editor_key = f"{key_prefix}_hitter_role_overrides"
    existing = st.session_state.get(editor_key, {})
    if not isinstance(existing, dict):
        existing = {}
    overrides: dict[str, str] = {
        str(k): str(v).strip()
        for k, v in existing.items()
        if str(k).strip() and str(v).strip()
    }

    name_col = (
        "hitter_name"
        if "hitter_name" in out.columns
        else ("player_name" if "player_name" in out.columns else ("name" if "name" in out.columns else None))
    )
    team_col = _team_col(out)

    candidates = out.copy()
    candidates["id_int"] = pd.to_numeric(candidates[id_col], errors="coerce").astype("Int64")
    candidates = candidates[candidates["id_int"].notna()].copy()
    if candidates.empty:
        st.info("Edit mode unavailable: no hitters with valid ids.")
        return out, {"overrides_active": len(overrides), "rows_overridden": 0}

    keep_cols = ["id_int", "opening_day_status_40man"]
    if name_col:
        keep_cols.append(name_col)
    if team_col:
        keep_cols.append(team_col)
    candidates = candidates[keep_cols].drop_duplicates(subset=["id_int"], keep="first").copy()

    role_values_df = (
        out["opening_day_status_40man"].dropna().astype(str).str.strip().tolist()
        if "opening_day_status_40man" in out.columns
        else []
    )
    lookup = _load_40man_lookup(for_pitchers=False)
    role_values_lookup = (
        lookup["opening_day_status_40man"].dropna().astype(str).str.strip().tolist()
        if not lookup.empty and "opening_day_status_40man" in lookup.columns
        else []
    )
    role_options = sorted(
        {
            v
            for v in (role_values_df + role_values_lookup)
            if v and v.lower() != "nan"
        }
    )
    if not role_options:
        role_options = [
            "Lineup Regular",
            "Platoon vs L",
            "Platoon vs R",
            "Bench",
            "Lineup/Bench Candidate",
            "Projected Injured List",
            "Bench Candidate",
            "Lineup Candidate",
            "Optioned",
            "Projected Injured List (MiLB)",
            "DFA",
            "Reassigned",
            "60-Day IL",
        ]

    label_by_key: dict[str, str] = {}
    current_role_by_key: dict[str, str] = {}
    for row in candidates.itertuples(index=False):
        pid = getattr(row, "id_int")
        if pd.isna(pid):
            continue
        key = str(int(pid))
        nm = str(getattr(row, name_col) if name_col else f"Hitter {key}").strip() if name_col else f"Hitter {key}"
        tm = str(getattr(row, team_col) if team_col else "").strip().upper() if team_col else ""
        cur_role = str(getattr(row, "opening_day_status_40man", "") or "").strip()
        current_role_by_key[key] = cur_role
        if tm:
            label_by_key[key] = f"{nm} ({tm}) [{key}]"
        else:
            label_by_key[key] = f"{nm} [{key}]"

    picker_keys = sorted(label_by_key.keys(), key=lambda k: label_by_key[k])
    if not picker_keys:
        return out, {"overrides_active": len(overrides), "rows_overridden": 0}
    picker_labels = [label_by_key[k] for k in picker_keys]
    selected_label = st.selectbox(
        "Edit mode: hitter",
        picker_labels,
        index=0,
        key=f"{key_prefix}_edit_mode_hitter_picker",
    )
    selected_key = picker_keys[picker_labels.index(selected_label)]

    role_placeholder = "(Clear role)"
    current_effective = overrides.get(selected_key, current_role_by_key.get(selected_key, ""))
    role_index = 0
    if current_effective in role_options:
        role_index = 1 + role_options.index(current_effective)
    selected_role = st.selectbox(
        "Edit mode: override Opening Day role",
        [role_placeholder] + role_options,
        index=role_index,
        key=f"{key_prefix}_edit_mode_hitter_role_value",
        help="Overrides role for this hitter in the current app session.",
    )
    edit_cols = st.columns(3)
    with edit_cols[0]:
        if st.button("Apply override", key=f"{key_prefix}_edit_mode_apply_hitter_role"):
            if selected_role == role_placeholder:
                overrides.pop(selected_key, None)
            else:
                overrides[selected_key] = selected_role
            st.session_state[editor_key] = overrides
            st.rerun()
    with edit_cols[1]:
        if st.button("Clear selected", key=f"{key_prefix}_edit_mode_clear_hitter_selected"):
            overrides.pop(selected_key, None)
            st.session_state[editor_key] = overrides
            st.rerun()
    with edit_cols[2]:
        if st.button("Clear all overrides", key=f"{key_prefix}_edit_mode_clear_hitter_all"):
            st.session_state[editor_key] = {}
            st.rerun()

    if overrides:
        rows = []
        for key, role in sorted(overrides.items(), key=lambda kv: label_by_key.get(kv[0], kv[0])):
            rows.append(
                {
                    "Hitter": label_by_key.get(key, key),
                    "Override Opening Day Role": role,
                    "Roster Role": current_role_by_key.get(key, ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    id_keys = pd.to_numeric(out[id_col], errors="coerce").astype("Int64").astype("string")
    mapped_roles = id_keys.map(overrides)
    override_mask = mapped_roles.notna()
    if bool(override_mask.any()):
        out.loc[override_mask, "opening_day_status_40man"] = mapped_roles.loc[override_mask].astype(str)

    return out, {"overrides_active": len(overrides), "rows_overridden": int(override_mask.sum())}


def _recompute_projection_spreads(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spread_cols = [c for c in out.columns if c.endswith("_proj_spread")]
    for spread_col in spread_cols:
        base = spread_col[: -len("_proj_spread")]
        c25 = f"{base}_proj_p25"
        c75 = f"{base}_proj_p75"
        c20 = f"{base}_proj_p20"
        c80 = f"{base}_proj_p80"
        if c25 in out.columns and c75 in out.columns:
            out[spread_col] = (
                pd.to_numeric(out[c75], errors="coerce")
                - pd.to_numeric(out[c25], errors="coerce")
            )
        elif c20 in out.columns and c80 in out.columns:
            out[spread_col] = (
                pd.to_numeric(out[c80], errors="coerce")
                - pd.to_numeric(out[c20], errors="coerce")
            )
    return out


def _blend_p50_volume_with_role_average(
    df: pd.DataFrame,
    *,
    volume_base: str,
    role_col: str = "opening_day_status_40man",
) -> tuple[pd.DataFrame, dict[str, float]]:
    vol_p50_col = f"{volume_base}_proj_p50"
    if df.empty or vol_p50_col not in df.columns or role_col not in df.columns:
        return df, {"rows_adjusted": 0, "roles_used": 0}
    out = df.copy()
    role_txt = out[role_col].fillna("").astype(str).str.strip()
    p50_vals = pd.to_numeric(out[vol_p50_col], errors="coerce")
    role_mask = role_txt.ne("") & p50_vals.notna() & np.isfinite(p50_vals) & (p50_vals > 0.0)
    if not bool(role_mask.any()):
        return out, {"rows_adjusted": 0, "roles_used": 0}

    role_means = (
        out.loc[role_mask, [role_col, vol_p50_col]]
        .assign(_v=p50_vals.loc[role_mask])
        .groupby(role_col)["_v"]
        .mean()
    )
    mapped_mean = role_txt.map(role_means)
    adj_mask = role_txt.ne("") & mapped_mean.notna() & p50_vals.notna() & np.isfinite(p50_vals)
    if not bool(adj_mask.any()):
        return out, {"rows_adjusted": 0, "roles_used": int(len(role_means))}

    new_p50 = p50_vals.copy()
    new_p50.loc[adj_mask] = 0.5 * (p50_vals.loc[adj_mask] + mapped_mean.loc[adj_mask])
    out[vol_p50_col] = new_p50

    scale = pd.Series(1.0, index=out.index, dtype="float64")
    base_nonzero = p50_vals.abs() > 1e-12
    scale_mask = adj_mask & base_nonzero
    scale.loc[scale_mask] = new_p50.loc[scale_mask] / p50_vals.loc[scale_mask]

    for tag in ("p20", "p25", "p75", "p80"):
        col = f"{volume_base}_proj_{tag}"
        if col not in out.columns:
            continue
        cur = pd.to_numeric(out[col], errors="coerce")
        out[col] = np.where(scale_mask, cur * scale, cur)

    out = _recompute_projection_spreads(out)
    return out, {"rows_adjusted": int(adj_mask.sum()), "roles_used": int(len(role_means))}


def _apply_role_playing_time_adjustments(df: pd.DataFrame, *, for_pitchers: bool) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    if out.empty:
        return out, {"rows_adjusted": 0, "roles_used": 0}

    if for_pitchers:
        before_ip = {
            tag: (
                pd.to_numeric(out[f"IP_proj_{tag}"], errors="coerce")
                if f"IP_proj_{tag}" in out.columns
                else None
            )
            for tag in ("p20", "p25", "p50", "p75", "p80")
        }
        out, summary = _blend_p50_volume_with_role_average(out, volume_base="IP")
        after_ip = {
            tag: (
                pd.to_numeric(out[f"IP_proj_{tag}"], errors="coerce")
                if f"IP_proj_{tag}" in out.columns
                else None
            )
            for tag in ("p20", "p25", "p50", "p75", "p80")
        }
        scale_by_tag: dict[str, pd.Series] = {}
        for tag in ("p20", "p25", "p50", "p75", "p80"):
            before = before_ip[tag]
            after = after_ip[tag]
            if before is None or after is None:
                continue
            denom_ok = before.abs() > 1e-12
            scale = pd.Series(1.0, index=out.index, dtype="float64")
            scale.loc[denom_ok] = after.loc[denom_ok] / before.loc[denom_ok]
            scale_by_tag[tag] = scale
        volume_bases = ["G", "GS", "TBF", "SO", "W", "SV", "BB", "H", "HR", "HBP", "ER"]
        for base in volume_bases:
            for tag, scale in scale_by_tag.items():
                col = f"{base}_proj_{tag}"
                if col not in out.columns:
                    continue
                cur = pd.to_numeric(out[col], errors="coerce")
                out[col] = cur * scale
        out = _recompute_projection_spreads(out)
        return out, summary

    out, summary = _blend_p50_volume_with_role_average(out, volume_base="PA")
    out = _recompute_counts_and_surface_from_rates(out)
    out = _recompute_projection_spreads(out)
    return out, summary


def _apply_projected_il_volume_cut(
    df: pd.DataFrame,
    *,
    cut_share: float = 0.33,
    role_col: str = "opening_day_status_40man",
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    if out.empty or role_col not in out.columns:
        return out, {"rows_adjusted": 0, "cut_share": float(cut_share)}

    cut = float(np.clip(cut_share, 0.0, 0.95))
    keep_factor = 1.0 - cut
    role_mask = out[role_col].map(_is_projected_il_role).fillna(False)
    if not bool(role_mask.any()):
        return out, {"rows_adjusted": 0, "cut_share": cut}

    vol_bases = [
        "G",
        "GS",
        "IP",
        "TBF",
        "SO",
        "W",
        "SV",
        "BB",
        "H",
        "HR",
        "HBP",
        "ER",
        "R",
        "G_marcel",
        "GS_marcel",
        "IP_marcel",
        "TBF_marcel",
        "SV_marcel",
    ]
    pct_tags = ["p20", "p25", "p50", "p75", "p80"]
    for base in vol_bases:
        for tag in pct_tags:
            col = f"{base}_proj_{tag}"
            if col not in out.columns:
                continue
            out.loc[role_mask, col] = (
                pd.to_numeric(out.loc[role_mask, col], errors="coerce").fillna(0.0)
                * keep_factor
            )

    out = _recompute_projection_spreads(out)
    return out, {"rows_adjusted": int(role_mask.sum()), "cut_share": cut}


def _pitcher_opening_day_role_priority(role: object) -> int:
    txt = str(role or "").strip().upper()
    if not txt:
        return 99
    if "STARTING ROTATION" in txt:
        return 1
    if "CLOSER" in txt:
        return 2
    if "SETUP MAN" in txt:
        return 3
    if "MIDDLE RELIEVER" in txt:
        return 4
    if "LONG RELIEVER" in txt:
        return 5
    if ("PROJECTED INJURED LIST" in txt) or ("PROJECTED IL" in txt):
        return 6
    if ("ROTATION CANDIDATE" in txt) or ("BULLPEN CANDIDATE" in txt):
        return 7
    if ("60-DAY IL" in txt) or ("60 DAY IL" in txt):
        return 8
    return 9


def _is_projected_il_role(role: object) -> bool:
    txt = str(role or "").strip().upper()
    if not txt:
        return False
    return ("PROJECTED INJURED LIST" in txt) or ("PROJECTED IL" in txt)


def _hitter_opening_day_role_priority(role: object) -> int:
    txt = str(role or "").strip().upper()
    if not txt:
        return 99
    if "LINEUP REGULAR" in txt:
        return 1
    if "PLATOON VS L" in txt:
        return 2
    if "PLATOON VS R" in txt:
        return 3
    if txt == "BENCH":
        return 4
    if "LINEUP/BENCH CANDIDATE" in txt:
        return 5
    if ("PROJECTED INJURED LIST (MILB)" in txt) or ("PROJECTED IL (MILB)" in txt):
        return 10
    if ("PROJECTED INJURED LIST" in txt) or ("PROJECTED IL" in txt):
        return 6
    if "BENCH CANDIDATE" in txt:
        return 7
    if "LINEUP CANDIDATE" in txt:
        return 8
    if "OPTIONED" in txt:
        return 9
    if "DFA" in txt:
        return 11
    if "REASSIGNED" in txt:
        return 12
    if ("60-DAY IL" in txt) or ("60 DAY IL" in txt):
        return 13
    return 99


def _allocate_with_caps(
    desired: pd.Series,
    cap: pd.Series,
    *,
    total_target: float,
    max_iter: int = 12,
) -> pd.Series:
    idx = desired.index
    out = pd.Series(0.0, index=idx, dtype="float64")
    cap_vals = pd.to_numeric(cap, errors="coerce").fillna(0.0).clip(lower=0.0)
    desired_vals = pd.to_numeric(desired, errors="coerce").fillna(0.0).clip(lower=0.0)
    out = np.minimum(desired_vals, cap_vals)
    remaining = float(max(total_target, 0.0) - float(out.sum()))
    if remaining <= 1e-9:
        return out
    for _ in range(int(max_iter)):
        room = (cap_vals - out).clip(lower=0.0)
        active = room > 1e-9
        if not bool(active.any()):
            break
        weights = desired_vals.where(active, 0.0)
        if float(weights.sum()) <= 1e-12:
            weights = room.where(active, 0.0)
        if float(weights.sum()) <= 1e-12:
            break
        add = remaining * (weights / float(weights.sum()))
        add = np.minimum(add, room)
        gained = float(add.sum())
        if gained <= 1e-9:
            break
        out = out + add
        remaining -= gained
        if remaining <= 1e-9:
            break
    return out


def _mlb_source_level_mask(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(False, index=df.index, dtype="bool")
    used_any = False

    if "appeared_in_MLB" in df.columns:
        used_any = True
        out = out | (pd.to_numeric(df["appeared_in_MLB"], errors="coerce").fillna(0.0) > 0.0)

    lvl_col = _source_levels_col(df)
    if lvl_col is not None and lvl_col in df.columns:
        used_any = True
        lvl_has_mlb = df[lvl_col].map(
            lambda v: any(tok == "1" for tok in _split_level_tokens(v))
        )
        out = out | lvl_has_mlb.fillna(False)

    if "level_id_source" in df.columns:
        used_any = True
        out = out | (
            pd.to_numeric(df["level_id_source"], errors="coerce").fillna(0.0) == 1.0
        )

    if not used_any:
        return pd.Series(True, index=df.index, dtype="bool")
    return out.fillna(False)


def _apply_team_mlb_ip_source_allocation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    if out.empty:
        out["is_mlb_ip_source"] = False
        return out, {"teams_adjusted": 0, "rows_selected": 0, "team_ip_mean": np.nan, "team_ip_sd": np.nan}

    team_col = _team_col(out)
    if team_col is None:
        out["is_mlb_ip_source"] = False
        return out, {"teams_adjusted": 0, "rows_selected": 0, "team_ip_mean": np.nan, "team_ip_sd": np.nan}

    ip50_col = "IP_proj_p50"
    g50_col = "G_proj_p50"
    gs50_col = "GS_proj_p50"
    if not all(c in out.columns for c in [ip50_col, g50_col, gs50_col]):
        out["is_mlb_ip_source"] = False
        return out, {"teams_adjusted": 0, "rows_selected": 0, "team_ip_mean": np.nan, "team_ip_sd": np.nan}

    fallback_team_key = out[team_col].map(
        lambda v: _team_tokens(v)[0] if _team_tokens(v) else ""
    )
    team_40 = (
        out["team_40man"].fillna("").astype(str).str.strip().str.upper()
        if "team_40man" in out.columns
        else pd.Series("", index=out.index, dtype="string")
    )
    team_key = team_40.where(team_40.ne(""), fallback_team_key)

    mlb_teams = set()
    lookup = _load_40man_lookup(for_pitchers=True)
    if not lookup.empty and "team_40man" in lookup.columns:
        mlb_teams = {
            str(v).strip().upper()
            for v in lookup["team_40man"].dropna().astype(str).tolist()
            if str(v).strip()
        }
    if not mlb_teams:
        mlb_teams = {t for t in team_key.dropna().astype(str).tolist() if t}

    mlb_mask = _mlb_source_level_mask(out)
    ip50 = pd.to_numeric(out[ip50_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    g50 = pd.to_numeric(out[g50_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    gs50 = pd.to_numeric(out[gs50_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    role_txt = (
        out.get("opening_day_status_40man", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    role_priority = out.get("opening_day_status_40man", pd.Series("", index=out.index)).map(
        _pitcher_opening_day_role_priority
    )

    candidate_mask = (
        team_key.ne("")
        & team_key.isin(mlb_teams)
        & mlb_mask
        & role_txt.ne("")
        & (~role_txt.map(_is_projected_il_role))
        & (ip50 > 0.0)
    )
    if not bool(candidate_mask.any()):
        out["is_mlb_ip_source"] = False
        return out, {"teams_adjusted": 0, "rows_selected": 0, "team_ip_mean": np.nan, "team_ip_sd": np.nan}

    raw_team_ip = (
        pd.DataFrame({"team": team_key, "ip": ip50, "cand": candidate_mask})
        .loc[lambda x: x["cand"]]
        .groupby("team")["ip"]
        .sum()
    )
    if raw_team_ip.empty:
        out["is_mlb_ip_source"] = False
        return out, {"teams_adjusted": 0, "rows_selected": 0, "team_ip_mean": np.nan, "team_ip_sd": np.nan}

    raw_mu = float(raw_team_ip.mean())
    raw_sd = float(raw_team_ip.std(ddof=0))
    if not np.isfinite(raw_sd) or raw_sd <= 1e-8:
        raw_sd = np.nan

    frac = pd.Series(0.0, index=out.index, dtype="float64")
    team_target_ip: dict[str, float] = {}
    teams_adjusted = 0
    for team, team_total_ip in raw_team_ip.items():
        team_idx = out.index[team_key.eq(team) & candidate_mask]
        if len(team_idx) == 0:
            continue
        block = pd.DataFrame(
            {
                "idx": team_idx,
                "priority": pd.to_numeric(role_priority.loc[team_idx], errors="coerce").fillna(99.0),
                "ip": ip50.loc[team_idx],
                "g": g50.loc[team_idx],
                "gs": gs50.loc[team_idx],
            }
        ).sort_values(
            ["priority", "ip", "gs", "g"],
            ascending=[True, False, False, False],
        )
        if block.empty:
            continue

        if np.isfinite(raw_sd):
            z = (float(team_total_ip) - raw_mu) / raw_sd
            target_ip = float(TEAM_IP_TARGET_MU + (z * TEAM_IP_TARGET_SD))
        else:
            target_ip = float(TEAM_IP_TARGET_MU)
        target_ip = float(np.clip(target_ip, TEAM_IP_MIN, TEAM_IP_MAX))
        team_target_ip[str(team)] = target_ip
        rem_ip = target_ip

        for row in block.itertuples(index=False):
            idx = int(row.idx)
            ip_i = float(max(row.ip, 0.0))
            if ip_i <= 0.0:
                continue

            take = float(np.clip(rem_ip / ip_i, 0.0, 1.0))
            if take <= 1e-9:
                continue
            frac.loc[idx] = max(frac.loc[idx], take)
            rem_ip -= take * ip_i
            if rem_ip <= 1e-6:
                break
        teams_adjusted += 1

    src_mask = frac > 1e-9
    out["is_mlb_ip_source"] = src_mask
    out["mlb_ip_source_share"] = frac

    vol_bases = [
        "G",
        "GS",
        "IP",
        "TBF",
        "SO",
        "W",
        "SV",
        "BB",
        "H",
        "HR",
        "HBP",
        "ER",
        "R",
        "G_marcel",
        "GS_marcel",
        "IP_marcel",
        "TBF_marcel",
        "SV_marcel",
    ]
    pct_tags = ["p20", "p25", "p50", "p75", "p80"]
    contrib_cols: dict[str, pd.Series] = {}
    for base in vol_bases:
        for tag in pct_tags:
            col = f"{base}_proj_{tag}"
            if col not in out.columns:
                continue
            cur = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            contrib_cols[col] = cur * frac

    # Enforce team-level p50 targets after source selection:
    # IP in [1400,1470]-aligned target by team, and GS normalized to 162.
    for team in sorted(team_target_ip.keys()):
        team_mask = team_key.eq(team) & src_mask
        if not bool(team_mask.any()):
            continue

        cur_ip = float(
            contrib_cols.get(
                "IP_proj_p50", pd.Series(0.0, index=out.index, dtype="float64")
            ).loc[team_mask]
            .sum()
        )
        tgt_ip = float(team_target_ip[team])
        ip_factor = (tgt_ip / cur_ip) if cur_ip > 1e-9 else 1.0
        if np.isfinite(ip_factor) and ip_factor > 0.0:
            ip_bases = [
                "IP",
                "TBF",
                "SO",
                "W",
                "SV",
                "BB",
                "H",
                "HR",
                "HBP",
                "ER",
                "R",
                "IP_marcel",
                "TBF_marcel",
                "SV_marcel",
            ]
            for base in ip_bases:
                for tag in pct_tags:
                    col = f"{base}_proj_{tag}"
                    if col not in contrib_cols:
                        continue
                    contrib_cols[col].loc[team_mask] = (
                        contrib_cols[col].loc[team_mask] * ip_factor
                    )

        cur_gs = float(
            contrib_cols.get(
                "GS_proj_p50", pd.Series(0.0, index=out.index, dtype="float64")
            ).loc[team_mask]
            .sum()
        )
        gs_factor = (float(TEAM_GS_TARGET) / cur_gs) if cur_gs > 1e-9 else 1.0
        if np.isfinite(gs_factor) and gs_factor > 0.0:
            for base in ["GS", "GS_marcel"]:
                for tag in pct_tags:
                    col = f"{base}_proj_{tag}"
                    if col not in contrib_cols:
                        continue
                    contrib_cols[col].loc[team_mask] = (
                        contrib_cols[col].loc[team_mask] * gs_factor
                    )

    # Preserve original player projections. Add MLB-source contribution columns
    # used for team/league aggregate accounting only.
    for col, vals in contrib_cols.items():
        out[f"{col}_mlb_source"] = vals

    team_ip_post = (
        pd.DataFrame(
            {
                "team": team_key,
                "ip": contrib_cols.get(
                    ip50_col, pd.Series(0.0, index=out.index, dtype="float64")
                ),
                "src": src_mask,
            }
        )
        .loc[lambda x: x["src"] & x["team"].ne("")]
        .groupby("team")["ip"]
        .sum()
    )
    team_g_post = (
        pd.DataFrame(
            {
                "team": team_key,
                "g": contrib_cols.get(
                    g50_col, pd.Series(0.0, index=out.index, dtype="float64")
                ),
                "src": src_mask,
            }
        )
        .loc[lambda x: x["src"] & x["team"].ne("")]
        .groupby("team")["g"]
        .sum()
    )
    team_gs_post = (
        pd.DataFrame(
            {
                "team": team_key,
                "gs": contrib_cols.get(
                    gs50_col, pd.Series(0.0, index=out.index, dtype="float64")
                ),
                "src": src_mask,
            }
        )
        .loc[lambda x: x["src"] & x["team"].ne("")]
        .groupby("team")["gs"]
        .sum()
    )
    return out, {
        "teams_adjusted": int(teams_adjusted),
        "rows_selected": int(src_mask.sum()),
        "team_ip_mean": float(team_ip_post.mean()) if not team_ip_post.empty else np.nan,
        "team_ip_sd": float(team_ip_post.std(ddof=0)) if len(team_ip_post) >= 2 else np.nan,
        "team_g_mean": float(team_g_post.mean()) if not team_g_post.empty else np.nan,
        "team_gs_mean": float(team_gs_post.mean()) if not team_gs_post.empty else np.nan,
    }


def _apply_team_mlb_pa_source_allocation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    if out.empty:
        out["is_mlb_pa_source"] = False
        out["mlb_pa_source_share"] = 0.0
        return out, {
            "teams_adjusted": 0,
            "rows_selected": 0,
            "team_pa_mean": np.nan,
            "team_pa_sd": np.nan,
            "team_pa_min": np.nan,
            "team_pa_max": np.nan,
        }

    team_col = _team_col(out)
    if team_col is None or "PA_proj_p50" not in out.columns:
        out["is_mlb_pa_source"] = False
        out["mlb_pa_source_share"] = 0.0
        return out, {
            "teams_adjusted": 0,
            "rows_selected": 0,
            "team_pa_mean": np.nan,
            "team_pa_sd": np.nan,
            "team_pa_min": np.nan,
            "team_pa_max": np.nan,
        }

    fallback_team_key = out[team_col].map(
        lambda v: _team_tokens(v)[0] if _team_tokens(v) else ""
    )
    team_40 = (
        out["team_40man"].fillna("").astype(str).str.strip().str.upper()
        if "team_40man" in out.columns
        else pd.Series("", index=out.index, dtype="string")
    )
    team_key = team_40.where(team_40.ne(""), fallback_team_key)

    mlb_teams = set()
    lookup = _load_40man_lookup(for_pitchers=False)
    if not lookup.empty and "team_40man" in lookup.columns:
        mlb_teams = {
            str(v).strip().upper()
            for v in lookup["team_40man"].dropna().astype(str).tolist()
            if str(v).strip()
        }
    if not mlb_teams:
        mlb_teams = {t for t in team_key.dropna().astype(str).tolist() if t}

    mlb_mask = _mlb_source_level_mask(out)
    pa50 = pd.to_numeric(out["PA_proj_p50"], errors="coerce").fillna(0.0).clip(lower=0.0)
    role_txt = (
        out.get("opening_day_status_40man", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    role_priority = out.get(
        "opening_day_status_40man",
        pd.Series("", index=out.index),
    ).map(_hitter_opening_day_role_priority)

    candidate_mask = (
        team_key.ne("")
        & team_key.isin(mlb_teams)
        & mlb_mask
        & role_txt.ne("")
        & (pa50 > 0.0)
    )
    if not bool(candidate_mask.any()):
        out["is_mlb_pa_source"] = False
        out["mlb_pa_source_share"] = 0.0
        return out, {
            "teams_adjusted": 0,
            "rows_selected": 0,
            "team_pa_mean": np.nan,
            "team_pa_sd": np.nan,
            "team_pa_min": np.nan,
            "team_pa_max": np.nan,
        }

    raw_team_pa = (
        pd.DataFrame({"team": team_key, "pa": pa50, "cand": candidate_mask})
        .loc[lambda x: x["cand"]]
        .groupby("team")["pa"]
        .sum()
    )
    if raw_team_pa.empty:
        out["is_mlb_pa_source"] = False
        out["mlb_pa_source_share"] = 0.0
        return out, {
            "teams_adjusted": 0,
            "rows_selected": 0,
            "team_pa_mean": np.nan,
            "team_pa_sd": np.nan,
            "team_pa_min": np.nan,
            "team_pa_max": np.nan,
        }

    raw_mu = float(raw_team_pa.mean())
    raw_sd = float(raw_team_pa.std(ddof=0))
    if not np.isfinite(raw_sd) or raw_sd <= 1e-8:
        raw_sd = np.nan

    frac = pd.Series(0.0, index=out.index, dtype="float64")
    team_target_pa: dict[str, float] = {}
    teams_adjusted = 0
    for team, team_total_pa in raw_team_pa.items():
        team_idx = out.index[team_key.eq(team) & candidate_mask]
        if len(team_idx) == 0:
            continue
        block = pd.DataFrame(
            {
                "idx": team_idx,
                "priority": pd.to_numeric(role_priority.loc[team_idx], errors="coerce").fillna(99.0),
                "pa": pa50.loc[team_idx],
            }
        ).sort_values(
            ["priority", "pa"],
            ascending=[True, False],
        )
        if block.empty:
            continue

        if np.isfinite(raw_sd):
            z = (float(team_total_pa) - raw_mu) / raw_sd
            target_pa = float(TEAM_PA_TARGET_MU + (z * TEAM_PA_TARGET_SD))
        else:
            target_pa = float(TEAM_PA_TARGET_MU)
        target_pa = float(np.clip(target_pa, TEAM_PA_MIN, TEAM_PA_MAX))
        team_target_pa[str(team)] = target_pa
        rem_pa = target_pa

        for row in block.itertuples(index=False):
            idx = int(row.idx)
            pa_i = float(max(row.pa, 0.0))
            if pa_i <= 0.0:
                continue
            take = float(np.clip(rem_pa / pa_i, 0.0, 1.0))
            if take <= 1e-9:
                continue
            frac.loc[idx] = max(frac.loc[idx], take)
            rem_pa -= take * pa_i
            if rem_pa <= 1e-6:
                break
        teams_adjusted += 1

    src_mask = frac > 1e-9
    out["is_mlb_pa_source"] = src_mask
    out["mlb_pa_source_share"] = frac

    vol_bases = [
        "PA",
        "AB",
        "H",
        "1B",
        "2B",
        "3B",
        "HR",
        "Runs",
        "RBI",
        "SB",
        "CS",
        "SBA",
        "BB",
        "SO",
        "HBP",
        "SF",
        "SH",
        "bbe",
        "BIP",
    ]
    pct_tags = ["p20", "p25", "p50", "p75", "p80", "p600"]
    contrib_cols: dict[str, pd.Series] = {}
    for base in vol_bases:
        for tag in pct_tags:
            col = f"{base}_proj_{tag}"
            if col not in out.columns:
                continue
            cur = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
            contrib_cols[col] = cur * frac

    for team in sorted(team_target_pa.keys()):
        team_mask = team_key.eq(team) & src_mask
        if not bool(team_mask.any()):
            continue
        cur_pa = float(
            contrib_cols.get(
                "PA_proj_p50",
                pd.Series(0.0, index=out.index, dtype="float64"),
            ).loc[team_mask].sum()
        )
        tgt_pa = float(team_target_pa[team])
        pa_factor = (tgt_pa / cur_pa) if cur_pa > 1e-9 else 1.0
        if np.isfinite(pa_factor) and pa_factor > 0.0:
            for base in vol_bases:
                for tag in pct_tags:
                    col = f"{base}_proj_{tag}"
                    if col not in contrib_cols:
                        continue
                    contrib_cols[col].loc[team_mask] = (
                        contrib_cols[col].loc[team_mask] * pa_factor
                    )

    for tag in pct_tags:
        sba_col = f"SBA_proj_{tag}"
        sb_col = f"SB_proj_{tag}"
        cs_col = f"CS_proj_{tag}"
        if sba_col in contrib_cols and sb_col in contrib_cols:
            contrib_cols[sb_col] = np.minimum(
                contrib_cols[sb_col],
                contrib_cols[sba_col],
            )
            if cs_col in contrib_cols:
                contrib_cols[cs_col] = (
                    contrib_cols[sba_col] - contrib_cols[sb_col]
                ).clip(lower=0.0)

    for col, vals in contrib_cols.items():
        out[f"{col}_mlb_source"] = vals

    team_pa_post = (
        pd.DataFrame(
            {
                "team": team_key,
                "pa": contrib_cols.get(
                    "PA_proj_p50",
                    pd.Series(0.0, index=out.index, dtype="float64"),
                ),
                "src": src_mask,
            }
        )
        .loc[lambda x: x["src"] & x["team"].ne("")]
        .groupby("team")["pa"]
        .sum()
    )
    return out, {
        "teams_adjusted": int(teams_adjusted),
        "rows_selected": int(src_mask.sum()),
        "team_pa_mean": float(team_pa_post.mean()) if not team_pa_post.empty else np.nan,
        "team_pa_sd": float(team_pa_post.std(ddof=0)) if len(team_pa_post) >= 2 else np.nan,
        "team_pa_min": float(team_pa_post.min()) if not team_pa_post.empty else np.nan,
        "team_pa_max": float(team_pa_post.max()) if not team_pa_post.empty else np.nan,
    }


def _apply_hitter_mlb_sb_sba_source_allocation(
    df: pd.DataFrame,
    *,
    target_sb_total: float = MLB_SB_TARGET_2025,
    target_sba_total: float = MLB_SBA_TARGET_2025,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df.copy()
    out["is_mlb_sb_source"] = False
    if out.empty:
        out["mlb_sb_source_share"] = 0.0
        return out, {"rows_selected": 0, "sb_total_p50": np.nan, "sba_total_p50": np.nan}

    team_col = _team_col(out)
    if team_col is None:
        out["mlb_sb_source_share"] = 0.0
        return out, {"rows_selected": 0, "sb_total_p50": np.nan, "sba_total_p50": np.nan}
    if "SB_proj_p50" not in out.columns or "SBA_proj_p50" not in out.columns:
        out["mlb_sb_source_share"] = 0.0
        return out, {"rows_selected": 0, "sb_total_p50": np.nan, "sba_total_p50": np.nan}

    fallback_team_key = out[team_col].map(
        lambda v: _team_tokens(v)[0] if _team_tokens(v) else ""
    )
    team_40 = (
        out["team_40man"].fillna("").astype(str).str.strip().str.upper()
        if "team_40man" in out.columns
        else pd.Series("", index=out.index, dtype="string")
    )
    team_key = team_40.where(team_40.ne(""), fallback_team_key)

    mlb_teams = set()
    lookup = _load_40man_lookup(for_pitchers=False)
    if not lookup.empty and "team_40man" in lookup.columns:
        mlb_teams = {
            str(v).strip().upper()
            for v in lookup["team_40man"].dropna().astype(str).tolist()
            if str(v).strip()
        }
    if not mlb_teams:
        mlb_teams = {t for t in team_key.dropna().astype(str).tolist() if t}

    mlb_mask = _mlb_source_level_mask(out)
    role_txt = (
        out.get("opening_day_status_40man", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    role_priority = role_txt.map(_hitter_opening_day_role_priority)
    sba50 = pd.to_numeric(out["SBA_proj_p50"], errors="coerce").fillna(0.0).clip(lower=0.0)
    sb50 = pd.to_numeric(out["SB_proj_p50"], errors="coerce").fillna(0.0).clip(lower=0.0)
    sb50 = np.minimum(sb50, sba50)

    candidate_mask = (
        team_key.ne("")
        & team_key.isin(mlb_teams)
        & mlb_mask
        & role_txt.ne("")
        & (sba50 > 0.0)
    )
    if not bool(candidate_mask.any()):
        out["mlb_sb_source_share"] = 0.0
        return out, {"rows_selected": 0, "sb_total_p50": np.nan, "sba_total_p50": np.nan}

    max_rank = 13.0
    priority_w = (max_rank + 1.0 - pd.to_numeric(role_priority, errors="coerce").fillna(99.0)).clip(lower=0.0)

    sb_target = float(max(target_sb_total, 0.0))
    sba_target = float(max(target_sba_total, sb_target))

    w_sba = (sba50 * priority_w).where(candidate_mask, 0.0)
    if float(w_sba.sum()) <= 1e-9:
        w_sba = sba50.where(candidate_mask, 0.0)
    if float(w_sba.sum()) <= 1e-9:
        out["mlb_sb_source_share"] = 0.0
        return out, {"rows_selected": 0, "sb_total_p50": np.nan, "sba_total_p50": np.nan}

    sba_alloc50 = pd.Series(0.0, index=out.index, dtype="float64")
    sba_alloc50.loc[candidate_mask] = (
        sba_target
        * (w_sba.loc[candidate_mask] / float(w_sba.loc[candidate_mask].sum()))
    )

    w_sb = (sb50 * priority_w).where(candidate_mask, 0.0)
    if float(w_sb.sum()) <= 1e-9:
        w_sb = sb50.where(candidate_mask, 0.0)
    if float(w_sb.sum()) <= 1e-9:
        w_sb = sba_alloc50.where(candidate_mask, 0.0)
    sb_desired50 = pd.Series(0.0, index=out.index, dtype="float64")
    if float(w_sb.sum()) > 1e-9:
        sb_desired50.loc[candidate_mask] = (
            sb_target
            * (w_sb.loc[candidate_mask] / float(w_sb.loc[candidate_mask].sum()))
        )
    sb_alloc50 = pd.Series(0.0, index=out.index, dtype="float64")
    sb_alloc50.loc[candidate_mask] = _allocate_with_caps(
        sb_desired50.loc[candidate_mask],
        sba_alloc50.loc[candidate_mask],
        total_target=sb_target,
    )
    no_sb_baseline = float(sb50.where(candidate_mask, 0.0).sum()) <= 1e-9

    sba_share = pd.Series(0.0, index=out.index, dtype="float64")
    sb_share = pd.Series(0.0, index=out.index, dtype="float64")
    sba_denom_ok = sba50 > 1e-12
    sb_denom_ok = sb50 > 1e-12
    sba_share.loc[sba_denom_ok] = (
        sba_alloc50.loc[sba_denom_ok] / sba50.loc[sba_denom_ok]
    )
    sb_share.loc[sb_denom_ok] = (
        sb_alloc50.loc[sb_denom_ok] / sb50.loc[sb_denom_ok]
    )
    sba_share = sba_share.where(candidate_mask, 0.0).fillna(0.0).clip(lower=0.0)
    sb_share = sb_share.where(candidate_mask, 0.0).fillna(0.0).clip(lower=0.0)
    src_mask = candidate_mask & (sba_share > 1e-9)
    out["is_mlb_sb_source"] = src_mask
    out["mlb_sb_source_share"] = sba_share

    pct_tags = ["p20", "p25", "p50", "p75", "p80", "p600"]
    contrib_cols: dict[str, pd.Series] = {}
    for tag in pct_tags:
        sba_col = f"SBA_proj_{tag}"
        sb_col = f"SB_proj_{tag}"
        cs_col = f"CS_proj_{tag}"
        if sba_col in out.columns:
            sba_cur = pd.to_numeric(out[sba_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            sba_new = (sba_cur * sba_share).where(src_mask, 0.0)
            contrib_cols[sba_col] = sba_new
        if sb_col in out.columns:
            if no_sb_baseline and sba_col in contrib_cols:
                sb_new = (
                    contrib_cols[sba_col]
                    * (sb_target / float(max(sba_target, 1e-9)))
                ).where(src_mask, 0.0)
            else:
                sb_cur = pd.to_numeric(out[sb_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                sb_new = (sb_cur * sb_share).where(src_mask, 0.0)
            if sba_col in contrib_cols:
                sb_new = np.minimum(sb_new, contrib_cols[sba_col])
            contrib_cols[sb_col] = sb_new
        if cs_col in out.columns:
            if sba_col in contrib_cols and sb_col in contrib_cols:
                cs_new = (contrib_cols[sba_col] - contrib_cols[sb_col]).clip(lower=0.0)
            else:
                cs_cur = pd.to_numeric(out[cs_col], errors="coerce").fillna(0.0).clip(lower=0.0)
                cs_new = (cs_cur * sba_share).where(src_mask, 0.0)
            contrib_cols[cs_col] = cs_new

        pa_col = f"PA_proj_{tag}"
        sba_rate_col = f"stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta_proj_{tag}"
        sb_succ_col = f"stolen_base_success_rate_mlb_eq_non_ar_delta_proj_{tag}"
        if pa_col in out.columns and sba_rate_col in out.columns and sba_col in contrib_cols:
            pa_vals = pd.to_numeric(out[pa_col], errors="coerce").fillna(0.0).clip(lower=0.0)
            out[sba_rate_col] = _safe_divide(contrib_cols[sba_col], pa_vals).clip(lower=0.0, upper=1.0)
        if sb_succ_col in out.columns and sb_col in contrib_cols and sba_col in contrib_cols:
            out[sb_succ_col] = _safe_divide(contrib_cols[sb_col], contrib_cols[sba_col]).clip(lower=0.0, upper=1.0)

    for col, vals in contrib_cols.items():
        out[col] = vals
        out[f"{col}_mlb_source"] = vals

    out = _recompute_projection_spreads(out)
    sb_total = float(pd.to_numeric(out.get("SB_proj_p50"), errors="coerce").fillna(0.0).sum())
    sba_total = float(pd.to_numeric(out.get("SBA_proj_p50"), errors="coerce").fillna(0.0).sum())
    return out, {
        "rows_selected": int(src_mask.sum()),
        "sb_total_p50": sb_total,
        "sba_total_p50": sba_total,
    }


def _redistribute_team_saves_by_role(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    if out.empty or "SV_proj_p50" not in out.columns or "opening_day_status_40man" not in out.columns:
        return out, {"teams_adjusted": 0, "players_receiving_sv": 0}
    team_col = _team_col(out)
    if not team_col:
        return out, {"teams_adjusted": 0, "players_receiving_sv": 0}

    sv_tags = [tag for tag in ("p20", "p25", "p50", "p75", "p80") if f"SV_proj_{tag}" in out.columns]
    if not sv_tags:
        return out, {"teams_adjusted": 0, "players_receiving_sv": 0}

    team_series = out[team_col].fillna("").astype(str).str.strip().str.upper()
    role_series = out["opening_day_status_40man"].fillna("").astype(str).str.strip().str.upper()
    base_rank_vals = pd.to_numeric(out["SV_proj_p50"], errors="coerce").fillna(0.0)

    target_by_tag = {
        tag: max(0.0, float(TEAM_SV_MEAN + (SV_QUANTILE_Z.get(tag, 0.0) * TEAM_SV_STD)))
        for tag in sv_tags
    }

    teams_adjusted = 0
    players_receiving_sv = 0
    for team in sorted(team_series[team_series.ne("")].unique().tolist()):
        team_idx = out.index[team_series.eq(team)]
        if len(team_idx) == 0:
            continue
        eligible = team_idx[role_series.loc[team_idx].isin(ROLE_SAVE_ELIGIBLE)]
        if len(eligible) == 0:
            continue

        team_roles = role_series.loc[eligible]
        closer_idx = eligible[team_roles.eq(ROLE_CLOSER)]
        setup_idx = eligible[team_roles.eq(ROLE_SETUP)]
        middle_idx = eligible[team_roles.eq(ROLE_MIDDLE)]

        group_map: dict[int, int] = {}
        if len(closer_idx) == 1:
            group_map[int(closer_idx[0])] = 1
        elif len(closer_idx) > 1:
            for idx in closer_idx:
                group_map[int(idx)] = 2
        for idx in setup_idx:
            group_map[int(idx)] = 3
        for idx in middle_idx:
            group_map[int(idx)] = 4
        if not group_map:
            continue

        secondary_col = "IP_proj_p50" if "IP_proj_p50" in out.columns else "TBF_proj_p50"
        name_col = "name" if "name" in out.columns else ("player_name" if "player_name" in out.columns else None)
        group_strength = {
            1: 1.00,  # singleton closer
            2: 0.78,  # multi-closer bucket
            3: 0.46,  # setup
            4: 0.07,  # big drop to middle relievers
        }
        raw_weights = pd.Series(0.0, index=eligible, dtype="float64")
        for grp in [1, 2, 3, 4]:
            grp_idx = [idx for idx in eligible.tolist() if group_map.get(int(idx)) == grp]
            if not grp_idx:
                continue
            block = out.loc[grp_idx].copy()
            block["_rank_sv"] = base_rank_vals.loc[grp_idx]
            if secondary_col in out.columns:
                block["_rank_vol"] = pd.to_numeric(
                    out.loc[grp_idx, secondary_col],
                    errors="coerce",
                ).fillna(0.0)
            else:
                block["_rank_vol"] = 0.0
            if name_col is not None:
                block["_rank_name"] = out.loc[grp_idx, name_col].fillna("").astype(str)
                block = block.sort_values(
                    ["_rank_sv", "_rank_vol", "_rank_name"],
                    ascending=[False, False, True],
                )
            else:
                block = block.sort_values(
                    ["_rank_sv", "_rank_vol"],
                    ascending=[False, False],
                )
            ordered = block.index.tolist()
            intra = np.arange(1, len(ordered) + 1, dtype="float64")
            grp_w = float(group_strength.get(grp, 0.0)) / intra
            raw_weights.loc[ordered] = grp_w

        total_w = float(raw_weights.sum())
        if total_w <= 0.0 or not np.isfinite(total_w):
            continue
        weights_s = raw_weights / total_w

        for tag, team_total in target_by_tag.items():
            col = f"SV_proj_{tag}"
            if col not in out.columns:
                continue
            out.loc[team_idx, col] = 0.0
            out.loc[eligible, col] = float(team_total) * weights_s
        teams_adjusted += 1
        players_receiving_sv += len(eligible)

    out = _recompute_projection_spreads(out)
    return out, {"teams_adjusted": teams_adjusted, "players_receiving_sv": players_receiving_sv}


@st.cache_resource(show_spinner=False)
def _load_source_season_mlb_pa_lookup(
    path: Path = BP_HITTING_MLB_PA_LOOKUP_PATH,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["mlbid", "season", "mlb_pa_source_season"])
    try:
        bp = pd.read_parquet(
            path,
            columns=["mlbid", "season", "level_id", "plate_appearances_agg"],
        )
    except Exception:
        return pd.DataFrame(columns=["mlbid", "season", "mlb_pa_source_season"])
    if bp.empty:
        return pd.DataFrame(columns=["mlbid", "season", "mlb_pa_source_season"])
    bp["mlbid"] = pd.to_numeric(bp["mlbid"], errors="coerce").astype("Int64")
    bp["season"] = pd.to_numeric(bp["season"], errors="coerce").astype("Int64")
    bp["level_id"] = pd.to_numeric(bp["level_id"], errors="coerce")
    bp["plate_appearances_agg"] = pd.to_numeric(
        bp["plate_appearances_agg"],
        errors="coerce",
    ).fillna(0.0)
    work = bp[
        bp["mlbid"].notna()
        & bp["season"].notna()
        & bp["level_id"].eq(1.0)
    ][["mlbid", "season", "plate_appearances_agg"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["mlbid", "season", "mlb_pa_source_season"])
    out = (
        work.groupby(["mlbid", "season"], as_index=False)["plate_appearances_agg"]
        .sum()
        .rename(columns={"plate_appearances_agg": "mlb_pa_source_season"})
    )
    out["mlbid"] = out["mlbid"].astype("int64")
    out["season"] = out["season"].astype("int64")
    return out


@st.cache_resource(show_spinner=False)
def _load_hitter_position_counts_lookup() -> pd.DataFrame:
    source_path = (
        HITTER_POSITION_SOURCE_PARQUET_PATH
        if HITTER_POSITION_SOURCE_PARQUET_PATH.exists()
        else HITTER_POSITION_SOURCE_CSV_PATH
    )
    if not source_path.exists():
        cols = ["mlbid", "season", "level_id", *POSITION_FILTER_COLS]
        cols.extend(f"is_{p}" for p in POSITION_FILTER_COLS)
        return pd.DataFrame(columns=cols)

    read_cols = ["batter_mlbid", "season", "level_id", *POSITION_FILTER_COLS]
    read_cols.extend(f"is_{p}" for p in POSITION_FILTER_COLS)
    try:
        if source_path.suffix.lower() == ".parquet":
            raw = pd.read_parquet(source_path, columns=read_cols)
        else:
            raw = pd.read_csv(source_path, usecols=read_cols)
    except Exception:
        cols = ["mlbid", "season", "level_id", *POSITION_FILTER_COLS]
        cols.extend(f"is_{p}" for p in POSITION_FILTER_COLS)
        return pd.DataFrame(columns=cols)

    if raw.empty:
        cols = ["mlbid", "season", "level_id", *POSITION_FILTER_COLS]
        cols.extend(f"is_{p}" for p in POSITION_FILTER_COLS)
        return pd.DataFrame(columns=cols)

    out = raw.copy()
    out = out.rename(columns={"batter_mlbid": "mlbid"})
    out["mlbid"] = pd.to_numeric(out["mlbid"], errors="coerce").astype("Int64")
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["level_id"] = pd.to_numeric(out["level_id"], errors="coerce")

    for pos in POSITION_FILTER_COLS:
        if pos in out.columns:
            out[pos] = pd.to_numeric(out[pos], errors="coerce").fillna(0.0)
        else:
            out[pos] = 0.0
        flag_col = f"is_{pos}"
        if flag_col in out.columns:
            out[flag_col] = (
                pd.to_numeric(out[flag_col], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0, upper=1.0)
            )
        else:
            out[flag_col] = (out[pos] >= float(POSITION_COUNT_THRESHOLD)).astype(float)

    work = out[
        out["mlbid"].notna()
        & out["season"].notna()
        & out["level_id"].notna()
    ][["mlbid", "season", "level_id", *POSITION_FILTER_COLS, *[f"is_{p}" for p in POSITION_FILTER_COLS]]].copy()
    if work.empty:
        cols = ["mlbid", "season", "level_id", *POSITION_FILTER_COLS]
        cols.extend(f"is_{p}" for p in POSITION_FILTER_COLS)
        return pd.DataFrame(columns=cols)

    agg_map: dict[str, str] = {pos: "sum" for pos in POSITION_FILTER_COLS}
    agg_map.update({f"is_{p}": "max" for p in POSITION_FILTER_COLS})
    grouped = work.groupby(["mlbid", "season", "level_id"], as_index=False).agg(agg_map)

    for pos in POSITION_FILTER_COLS:
        flag_col = f"is_{pos}"
        # Recompute flags from counts so duplicate-name rows remain stable.
        grouped[flag_col] = (
            pd.to_numeric(grouped[pos], errors="coerce").fillna(0.0)
            >= float(POSITION_COUNT_THRESHOLD)
        ).astype("Int64")

    grouped["mlbid"] = grouped["mlbid"].astype("int64")
    grouped["season"] = grouped["season"].astype("int64")
    grouped["level_id"] = pd.to_numeric(grouped["level_id"], errors="coerce").round(0).astype("int64")
    return grouped


def _attach_hitter_position_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    id_col = None
    if "mlbid" in df.columns:
        id_col = "mlbid"
    elif "batter_mlbid" in df.columns:
        id_col = "batter_mlbid"
    if id_col is None:
        return df

    lookup = _load_hitter_position_counts_lookup()
    if lookup.empty:
        return df

    out = df.copy()
    out["_pos_id_join"] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
    if "source_season" in out.columns:
        out["_pos_season_join"] = pd.to_numeric(out["source_season"], errors="coerce").astype("Int64")
    elif "target_season" in out.columns:
        out["_pos_season_join"] = (
            pd.to_numeric(out["target_season"], errors="coerce") - 1.0
        ).astype("Int64")
    else:
        return out.drop(columns=["_pos_id_join"], errors="ignore")

    if "level_id_source" in out.columns:
        out["_pos_level_join"] = pd.to_numeric(out["level_id_source"], errors="coerce").round(0).astype("Int64")
    elif "level_id" in out.columns:
        out["_pos_level_join"] = pd.to_numeric(out["level_id"], errors="coerce").round(0).astype("Int64")
    else:
        out["_pos_level_join"] = pd.Series(1, index=out.index, dtype="Int64")

    merged = out.merge(
        lookup.rename(
            columns={
                "mlbid": "_pos_id_join",
                "season": "_pos_season_join",
                "level_id": "_pos_level_join",
            }
        ),
        on=["_pos_id_join", "_pos_season_join", "_pos_level_join"],
        how="left",
        suffixes=("", "_pos_src"),
    )
    merged = merged.drop(
        columns=["_pos_id_join", "_pos_season_join", "_pos_level_join"],
        errors="ignore",
    )

    for col in [*POSITION_FILTER_COLS, *[f"is_{p}" for p in POSITION_FILTER_COLS]]:
        src_col = f"{col}_pos_src"
        if src_col not in merged.columns:
            continue
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").where(
                pd.to_numeric(merged[col], errors="coerce").notna(),
                pd.to_numeric(merged[src_col], errors="coerce"),
            )
        else:
            merged[col] = pd.to_numeric(merged[src_col], errors="coerce")
        merged = merged.drop(columns=[src_col], errors="ignore")
    return merged


@st.cache_resource(show_spinner=False)
def _load_weighted_ops_mu_by_season(
    path: Path = BP_HITTING_MLB_PA_LOOKUP_PATH,
) -> dict[int, float]:
    if not path.exists():
        return {}
    try:
        bp = pd.read_parquet(
            path,
            columns=["season", "level_id", "plate_appearances_agg", "ops"],
        )
    except Exception:
        return {}
    if bp.empty:
        return {}
    bp["season"] = pd.to_numeric(bp["season"], errors="coerce").astype("Int64")
    bp["level_id"] = pd.to_numeric(bp["level_id"], errors="coerce")
    bp["plate_appearances_agg"] = pd.to_numeric(
        bp["plate_appearances_agg"],
        errors="coerce",
    )
    bp["ops"] = pd.to_numeric(bp["ops"], errors="coerce")
    work = bp[
        bp["season"].notna()
        & bp["level_id"].eq(1.0)
        & bp["plate_appearances_agg"].notna()
        & bp["ops"].notna()
        & (bp["plate_appearances_agg"] > 0.0)
    ][["season", "plate_appearances_agg", "ops"]].copy()
    if work.empty:
        return {}
    weighted_num = work["ops"] * work["plate_appearances_agg"]
    grouped = (
        work.assign(_weighted_num=weighted_num)
        .groupby("season", as_index=False)[["_weighted_num", "plate_appearances_agg"]]
        .sum()
    )
    grouped = grouped[grouped["plate_appearances_agg"] > 0.0]
    if grouped.empty:
        return {}
    grouped["mu"] = grouped["_weighted_num"] / grouped["plate_appearances_agg"]
    out: dict[int, float] = {}
    for _, row in grouped.iterrows():
        season = int(row["season"])
        mu = float(row["mu"])
        if np.isfinite(mu):
            out[season] = mu
    return out


@st.cache_resource(show_spinner=False)
def _load_ops_sigma_by_season_from_reference(
    path: Path = ZSPACE_REFERENCE_MLB_PA_GTE_100_PATH,
) -> dict[int, float]:
    if not path.exists():
        return {}
    try:
        ref = pd.read_csv(path, usecols=["metric", "season", "sigma"])
    except Exception:
        return {}
    if ref.empty:
        return {}
    metric = ref["metric"].astype(str).str.strip().str.lower()
    work = ref.loc[metric.eq("ops"), ["season", "sigma"]].copy()
    if work.empty:
        return {}
    work["season"] = pd.to_numeric(work["season"], errors="coerce").astype("Int64")
    work["sigma"] = pd.to_numeric(work["sigma"], errors="coerce")
    work = work[
        work["season"].notna()
        & work["sigma"].notna()
        & np.isfinite(work["sigma"])
        & (work["sigma"] > 1e-12)
    ]
    if work.empty:
        return {}
    grouped = work.groupby("season", as_index=False)["sigma"].median()
    out: dict[int, float] = {}
    for _, row in grouped.iterrows():
        season = int(row["season"])
        sigma = float(row["sigma"])
        if np.isfinite(sigma) and sigma > 1e-12:
            out[season] = sigma
    return out


@st.cache_resource(show_spinner=False)
def _load_weighted_era_mu_by_season(
    path: Path = BP_PITCHING_MLB_LOOKUP_PATH,
) -> dict[int, float]:
    if not path.exists():
        return {}
    try:
        bp = pd.read_parquet(
            path,
            columns=[
                "season",
                "bp_level_id",
                "innings_pitched_agg",
                "earned_runs_agg",
                "runs_agg",
            ],
        )
    except Exception:
        return {}
    if bp.empty:
        return {}
    bp["season"] = pd.to_numeric(bp["season"], errors="coerce").astype("Int64")
    bp["bp_level_id"] = pd.to_numeric(bp["bp_level_id"], errors="coerce")
    bp["innings_pitched_agg"] = pd.to_numeric(bp["innings_pitched_agg"], errors="coerce")
    er = pd.to_numeric(bp.get("earned_runs_agg"), errors="coerce")
    ra = pd.to_numeric(bp.get("runs_agg"), errors="coerce")
    runs_like = er.where(er.notna(), ra)
    work = bp[
        bp["season"].notna()
        & bp["bp_level_id"].eq(1.0)
        & bp["innings_pitched_agg"].notna()
        & np.isfinite(bp["innings_pitched_agg"])
        & (bp["innings_pitched_agg"] > 0.0)
    ][["season", "innings_pitched_agg"]].copy()
    work["runs_like"] = runs_like
    work = work[
        work["runs_like"].notna()
        & np.isfinite(work["runs_like"])
        & (work["runs_like"] >= 0.0)
    ]
    if work.empty:
        return {}
    grouped = (
        work.groupby("season", as_index=False)[["runs_like", "innings_pitched_agg"]]
        .sum()
    )
    grouped = grouped[grouped["innings_pitched_agg"] > 0.0]
    if grouped.empty:
        return {}
    grouped["mu"] = (9.0 * grouped["runs_like"]) / grouped["innings_pitched_agg"]
    out: dict[int, float] = {}
    for _, row in grouped.iterrows():
        season = int(row["season"])
        mu = float(row["mu"])
        if np.isfinite(mu):
            out[season] = mu
    return out


@st.cache_resource(show_spinner=False)
def _load_weighted_whip_mu_by_season(
    path: Path = BP_PITCHING_MLB_LOOKUP_PATH,
) -> dict[int, float]:
    if not path.exists():
        return {}
    try:
        bp = pd.read_parquet(
            path,
            columns=["season", "bp_level_id", "innings_pitched_agg", "whip"],
        )
    except Exception:
        return {}
    if bp.empty:
        return {}
    bp["season"] = pd.to_numeric(bp["season"], errors="coerce").astype("Int64")
    bp["bp_level_id"] = pd.to_numeric(bp["bp_level_id"], errors="coerce")
    bp["innings_pitched_agg"] = pd.to_numeric(
        bp["innings_pitched_agg"], errors="coerce"
    )
    bp["whip"] = pd.to_numeric(bp["whip"], errors="coerce")
    work = bp[
        bp["season"].notna()
        & bp["bp_level_id"].eq(1.0)
        & bp["innings_pitched_agg"].notna()
        & np.isfinite(bp["innings_pitched_agg"])
        & (bp["innings_pitched_agg"] > 0.0)
        & bp["whip"].notna()
        & np.isfinite(bp["whip"])
        & (bp["whip"] >= 0.0)
    ][["season", "innings_pitched_agg", "whip"]].copy()
    if work.empty:
        return {}
    weighted_num = work["whip"] * work["innings_pitched_agg"]
    grouped = (
        work.assign(_weighted_num=weighted_num)
        .groupby("season", as_index=False)[["_weighted_num", "innings_pitched_agg"]]
        .sum()
    )
    grouped = grouped[grouped["innings_pitched_agg"] > 0.0]
    if grouped.empty:
        return {}
    grouped["mu"] = grouped["_weighted_num"] / grouped["innings_pitched_agg"]
    out: dict[int, float] = {}
    for _, row in grouped.iterrows():
        season = int(row["season"])
        mu = float(row["mu"])
        if np.isfinite(mu):
            out[season] = mu
    return out


def _resolve_ops_reference_season(df: pd.DataFrame) -> int | None:
    season_vals = pd.Series(dtype="float64")
    if "source_season" in df.columns:
        season_vals = pd.to_numeric(df["source_season"], errors="coerce")
    elif "target_season" in df.columns:
        season_vals = pd.to_numeric(df["target_season"], errors="coerce") - 1.0
    season_vals = season_vals.replace([np.inf, -np.inf], np.nan).dropna()
    if len(season_vals) == 0:
        return None
    try:
        return int(round(float(season_vals.median())))
    except Exception:
        return None


def _attach_source_season_mlb_pa(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "source_season" not in df.columns:
        return df
    id_col = None
    if "mlbid" in df.columns:
        id_col = "mlbid"
    elif "batter_mlbid" in df.columns:
        id_col = "batter_mlbid"
    elif "pitcher_mlbid" in df.columns:
        id_col = "pitcher_mlbid"
    if id_col is None:
        return df
    lookup = _load_source_season_mlb_pa_lookup()
    if lookup.empty:
        return df
    out = df.copy()
    out["_id_join"] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
    out["_source_join"] = pd.to_numeric(out["source_season"], errors="coerce").astype("Int64")
    out = out.merge(
        lookup.rename(columns={"mlbid": "_id_join", "season": "_source_join"}),
        on=["_id_join", "_source_join"],
        how="left",
    )
    out = out.drop(columns=["_id_join", "_source_join"], errors="ignore")
    return out


def _apply_compact_ui_chrome(compact_ui: bool) -> None:
    if not compact_ui:
        return
    st.markdown(
        """
<style>
div.block-container {
  padding-top: 0.5rem !important;
  padding-bottom: 0.5rem !important;
}
div[data-testid="stMetric"] {
  padding-top: 0.1rem !important;
  padding-bottom: 0.1rem !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    out = pd.Series(np.nan, index=n.index, dtype="float64")
    mask = n.notna() & d.notna() & np.isfinite(n) & np.isfinite(d) & (d > 0)
    out.loc[mask] = n.loc[mask] / d.loc[mask]
    return out


def _true_flag_mask(values: pd.Series) -> pd.Series:
    txt = (
        values.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    return txt.isin({"T", "TRUE", "1", "Y", "YES"})


def _ops_reference_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if "mlb_pa_source_season" in df.columns:
        pa = pd.to_numeric(df["mlb_pa_source_season"], errors="coerce").fillna(0.0)
        mask = pa >= float(MLB_PA_REFERENCE_MIN)
    elif "mlb_pa_recent_3yr" in df.columns:
        pa3 = pd.to_numeric(df["mlb_pa_recent_3yr"], errors="coerce").fillna(0.0)
        mask = pa3 >= float(MLB_PA_REFERENCE_MIN)
    if (not bool(mask.any())) and "appeared_in_MLB" in df.columns:
        fallback = _true_flag_mask(df["appeared_in_MLB"])
        mask = fallback
    if not bool(mask.any()):
        mask = pd.Series(True, index=df.index)
    return mask


def _ops_env_reference_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    if "OPS_proj_p50" not in df.columns:
        return np.nan, np.nan, np.nan
    ref_mask = _ops_reference_mask(df)
    ref_season = _resolve_ops_reference_season(df)
    weighted_mu_lookup = _load_weighted_ops_mu_by_season()
    sigma_lookup = _load_ops_sigma_by_season_from_reference()
    source_mu = np.nan
    source_sigma = np.nan
    if ref_season is not None:
        source_mu = float(weighted_mu_lookup.get(ref_season, np.nan))
        source_sigma = float(sigma_lookup.get(ref_season, np.nan))

    ref_vals = (
        pd.to_numeric(df.loc[ref_mask, "OPS_proj_p50"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if (not np.isfinite(source_mu)) and len(ref_vals) >= 2:
        source_mu = float(ref_vals.median())
    if (not np.isfinite(source_sigma) or source_sigma <= 1e-12) and len(ref_vals) >= 2:
        source_sigma = float(ref_vals.std(ddof=0))
    if not np.isfinite(source_sigma) or source_sigma <= 1e-12:
        return source_mu, np.nan, np.nan
    target_sigma = np.nan
    if "ops_lg_sd" in df.columns:
        lg_sd = (
            pd.to_numeric(df.loc[ref_mask, "ops_lg_sd"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(lg_sd) > 0:
            sd = float(lg_sd.median())
            if np.isfinite(sd) and sd > 1e-12:
                target_sigma = sd
    if not np.isfinite(target_sigma) or target_sigma <= 1e-12:
        target_sigma = source_sigma
    return source_mu, source_sigma, target_sigma


def _pitching_reference_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in ["IP_marcel_proj_p50", "IP_proj_p50"]:
        if col in df.columns:
            ip = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            mask = ip >= float(MLB_IP_REFERENCE_MIN)
            if bool(mask.any()):
                break
    if (not bool(mask.any())) and "appeared_in_MLB" in df.columns:
        mask = _true_flag_mask(df["appeared_in_MLB"])
    if not bool(mask.any()):
        mask = pd.Series(True, index=df.index)
    return mask


def _era_env_reference_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    if "ERA_proj_p50" not in df.columns:
        return np.nan, np.nan, np.nan
    ref_mask = _pitching_reference_mask(df)
    ref_season = _resolve_ops_reference_season(df)
    weighted_mu_lookup = _load_weighted_era_mu_by_season()
    source_mu = np.nan
    source_sigma = np.nan
    if ref_season is not None:
        source_mu = float(weighted_mu_lookup.get(ref_season, np.nan))

    ref_vals = (
        pd.to_numeric(df.loc[ref_mask, "ERA_proj_p50"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if (not np.isfinite(source_mu)) and len(ref_vals) >= 2:
        source_mu = float(ref_vals.median())
    if len(ref_vals) >= 2:
        source_sigma = float(ref_vals.std(ddof=0))
    if not np.isfinite(source_sigma) or source_sigma <= 1e-12:
        return source_mu, np.nan, np.nan
    target_sigma = source_sigma
    return source_mu, source_sigma, target_sigma


def _whip_env_reference_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    if "WHIP_proj_p50" not in df.columns:
        return np.nan, np.nan, np.nan
    ref_mask = _pitching_reference_mask(df)
    ref_season = _resolve_ops_reference_season(df)
    weighted_mu_lookup = _load_weighted_whip_mu_by_season()
    source_mu = np.nan
    source_sigma = np.nan
    if ref_season is not None:
        source_mu = float(weighted_mu_lookup.get(ref_season, np.nan))

    ref_vals = (
        pd.to_numeric(df.loc[ref_mask, "WHIP_proj_p50"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if (not np.isfinite(source_mu)) and len(ref_vals) >= 2:
        source_mu = float(ref_vals.median())
    if len(ref_vals) >= 2:
        source_sigma = float(ref_vals.std(ddof=0))
    if not np.isfinite(source_sigma) or source_sigma <= 1e-12:
        return source_mu, np.nan, np.nan
    target_sigma = source_sigma
    return source_mu, source_sigma, target_sigma


def _reference_mu_sigma(
    df: pd.DataFrame,
    *,
    col: str,
    ref_mask: pd.Series,
) -> tuple[float, float]:
    if col not in df.columns:
        return np.nan, np.nan
    vals = (
        pd.to_numeric(df.loc[ref_mask, col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(vals) < 2:
        return np.nan, np.nan
    mu = float(vals.median())
    sigma = float(vals.std(ddof=0))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return mu, np.nan
    return mu, sigma


def _recompute_counts_and_surface_from_rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_tags = ("p20", "p25", "p50", "p75", "p80", "p600")
    for tag in pct_tags:
        pa_col = f"PA_proj_{tag}"
        if pa_col not in out.columns:
            continue
        pa = pd.to_numeric(out[pa_col], errors="coerce")
        if pa.isna().all():
            continue

        def _n(col: str) -> pd.Series:
            if col not in out.columns:
                return pd.Series(np.nan, index=out.index, dtype="float64")
            return pd.to_numeric(out[col], errors="coerce")

        bb_rate = _n(f"walk_rate_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0, upper=1.0)
        so_rate = _n(f"strikeout_rate_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0, upper=1.0)
        hbp_rate = _n(f"hit_by_pitch_rate_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0, upper=1.0)
        bbe_rate = _n(f"batted_ball_rate_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0, upper=1.0)
        sf_rate_bbe = _n(f"sac_fly_rate_bbe_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0, upper=1.0)
        sh_rate_bbe = _n(f"sac_hit_rate_bbe_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0, upper=1.0)

        one_rate = _n(f"singles_rate_bbe_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0)
        two_rate = _n(f"doubles_rate_bbe_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0)
        three_rate = _n(f"triples_rate_bbe_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0)
        hr_rate = _n(f"home_run_rate_bbe_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0)

        rate_sum = one_rate + two_rate + three_rate + hr_rate
        scale = pd.Series(1.0, index=out.index, dtype="float64")
        gt_one = rate_sum > 1.0
        scale.loc[gt_one] = 1.0 / rate_sum.loc[gt_one]
        one_rate = one_rate * scale
        two_rate = two_rate * scale
        three_rate = three_rate * scale
        hr_rate = hr_rate * scale

        bb = (pa * bb_rate).clip(lower=0.0)
        so = (pa * so_rate).clip(lower=0.0)
        hbp = (pa * hbp_rate).clip(lower=0.0)
        bbe = (pa * bbe_rate).clip(lower=0.0)
        sf = (bbe * sf_rate_bbe).clip(lower=0.0)
        sh = (bbe * sh_rate_bbe).clip(lower=0.0)

        one_b = (bbe * one_rate).clip(lower=0.0)
        two_b = (bbe * two_rate).clip(lower=0.0)
        three_b = (bbe * three_rate).clip(lower=0.0)
        hr = (bbe * hr_rate).clip(lower=0.0)

        hits = (one_b + two_b + three_b + hr).clip(lower=0.0)
        ab = (pa - (bb + hbp + sh + sf)).clip(lower=0.0)
        tb = (one_b + 2.0 * two_b + 3.0 * three_b + 4.0 * hr).clip(lower=0.0)

        sba_rate = _n(f"stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0, upper=1.0)
        sb_succ = _n(f"stolen_base_success_rate_mlb_eq_non_ar_delta_proj_{tag}").clip(lower=0.0, upper=1.0)
        sba = (pa * sba_rate).clip(lower=0.0)
        sb = (sba * sb_succ).clip(lower=0.0)
        cs = (sba - sb).clip(lower=0.0)

        if f"BB_proj_{tag}" in out.columns:
            out[f"BB_proj_{tag}"] = bb
        if f"SO_proj_{tag}" in out.columns:
            out[f"SO_proj_{tag}"] = so
        if f"HBP_proj_{tag}" in out.columns:
            out[f"HBP_proj_{tag}"] = hbp
        if f"bbe_proj_{tag}" in out.columns:
            out[f"bbe_proj_{tag}"] = bbe
        if f"SF_proj_{tag}" in out.columns:
            out[f"SF_proj_{tag}"] = sf
        if f"SH_proj_{tag}" in out.columns:
            out[f"SH_proj_{tag}"] = sh
        if f"1B_proj_{tag}" in out.columns:
            out[f"1B_proj_{tag}"] = one_b
        if f"2B_proj_{tag}" in out.columns:
            out[f"2B_proj_{tag}"] = two_b
        if f"3B_proj_{tag}" in out.columns:
            out[f"3B_proj_{tag}"] = three_b
        if f"HR_proj_{tag}" in out.columns:
            out[f"HR_proj_{tag}"] = hr
        if f"H_proj_{tag}" in out.columns:
            out[f"H_proj_{tag}"] = hits
        if f"AB_proj_{tag}" in out.columns:
            out[f"AB_proj_{tag}"] = ab
        if f"SBA_proj_{tag}" in out.columns:
            out[f"SBA_proj_{tag}"] = sba
        if f"SB_proj_{tag}" in out.columns:
            out[f"SB_proj_{tag}"] = sb
        if f"CS_proj_{tag}" in out.columns:
            out[f"CS_proj_{tag}"] = cs

        runs_rate_col = f"runs_rate_mlb_eq_non_ar_delta_proj_{tag}"
        rbi_rate_col = f"rbi_rate_mlb_eq_non_ar_delta_proj_{tag}"
        if runs_rate_col in out.columns and f"Runs_proj_{tag}" in out.columns:
            out[f"Runs_proj_{tag}"] = (
                pa * _n(runs_rate_col).clip(lower=0.0, upper=1.0)
            ).clip(lower=0.0)
        if rbi_rate_col in out.columns and f"RBI_proj_{tag}" in out.columns:
            out[f"RBI_proj_{tag}"] = (
                pa * _n(rbi_rate_col).clip(lower=0.0, upper=1.0)
            ).clip(lower=0.0)

        avg = _safe_divide(hits, ab)
        obp = _safe_divide(hits + bb + hbp, pa)
        slg = _safe_divide(tb, ab)
        ops = obp + slg
        k_pct = _safe_divide(so, pa) * 100.0
        bb_pct = _safe_divide(bb, pa) * 100.0
        if f"AVG_proj_{tag}" in out.columns:
            out[f"AVG_proj_{tag}"] = avg
        if f"OBP_proj_{tag}" in out.columns:
            out[f"OBP_proj_{tag}"] = obp
        if f"SLG_proj_{tag}" in out.columns:
            out[f"SLG_proj_{tag}"] = slg
        if f"OPS_proj_{tag}" in out.columns:
            out[f"OPS_proj_{tag}"] = ops
        if f"K%_proj_{tag}" in out.columns:
            out[f"K%_proj_{tag}"] = k_pct
        if f"BB%_proj_{tag}" in out.columns:
            out[f"BB%_proj_{tag}"] = bb_pct

    return out


def _apply_ops_delta_to_required_rates(
    df: pd.DataFrame,
    *,
    ref_mask: pd.Series,
    delta_z_ops: float,
) -> pd.DataFrame:
    out = df.copy()
    required_rate_bases = [
        "walk_rate_mlb_eq_non_ar_delta",
        "strikeout_rate_mlb_eq_non_ar_delta",
        "hit_by_pitch_rate_mlb_eq_non_ar_delta",
        "batted_ball_rate_mlb_eq_non_ar_delta",
        "singles_rate_bbe_mlb_eq_non_ar_delta",
        "doubles_rate_bbe_mlb_eq_non_ar_delta",
        "triples_rate_bbe_mlb_eq_non_ar_delta",
        "home_run_rate_bbe_mlb_eq_non_ar_delta",
        "sac_fly_rate_bbe_mlb_eq_non_ar_delta",
        "sac_hit_rate_bbe_mlb_eq_non_ar_delta",
        "stolen_base_attempt_rate_pa_mlb_eq_non_ar_delta",
        "stolen_base_success_rate_mlb_eq_non_ar_delta",
        "runs_rate_mlb_eq_non_ar_delta",
        "rbi_rate_mlb_eq_non_ar_delta",
    ]
    pct_tags = ("p20", "p25", "p50", "p75", "p80", "p600")
    for base in required_rate_bases:
        for tag in pct_tags:
            col = f"{base}_proj_{tag}"
            if col not in out.columns:
                continue
            mu, sigma = _reference_mu_sigma(out, col=col, ref_mask=ref_mask)
            if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 1e-12:
                continue
            vals = pd.to_numeric(out[col], errors="coerce")
            z_vals = (vals - mu) / sigma
            target_mu = mu + (delta_z_ops * sigma)
            out[col] = target_mu + (z_vals * sigma)
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0.0, upper=1.0)
    return _recompute_counts_and_surface_from_rates(out)


def _apply_ops_league_environment(
    df: pd.DataFrame,
    *,
    target_ops_median: float,
    target_ops_sigma: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if "OPS_proj_p50" not in out.columns:
        return out
    source_mu, source_sigma, target_sigma_default = _ops_env_reference_stats(out)
    target_sigma = (
        float(target_ops_sigma)
        if target_ops_sigma is not None and np.isfinite(float(target_ops_sigma))
        else float(target_sigma_default)
    )
    if (
        not np.isfinite(source_mu)
        or not np.isfinite(source_sigma)
        or source_sigma <= 1e-12
        or not np.isfinite(target_sigma)
        or target_sigma <= 1e-12
    ):
        return out
    if not np.isfinite(float(target_ops_median)):
        return out
    ref_mask = _ops_reference_mask(out)
    target_mu = float(target_ops_median)
    base_mu = source_mu
    delta_z_raw = (target_mu - base_mu) / target_sigma

    if abs(delta_z_raw) <= 1e-12:
        return out

    # Calibrate delta_z so recomputed OPS median (MLB-T reference) lands on target.
    def _median_ops_p50(frame: pd.DataFrame) -> float:
        vals = pd.to_numeric(frame.loc[ref_mask, "OPS_proj_p50"], errors="coerce")
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) == 0:
            return np.nan
        return float(vals.median())

    lo = 0.0
    hi = float(delta_z_raw)
    trial_hi = _apply_ops_delta_to_required_rates(out, ref_mask=ref_mask, delta_z_ops=hi)
    med_lo = base_mu
    med_hi = _median_ops_p50(trial_hi)

    calibrated = trial_hi
    if np.isfinite(med_hi):
        target_between = (
            (med_lo <= target_mu <= med_hi) or (med_hi <= target_mu <= med_lo)
        )
        if target_between:
            best = trial_hi
            best_err = abs(med_hi - target_mu)
            lo_d, hi_d = lo, hi
            lo_m, hi_m = med_lo, med_hi
            for _ in range(10):
                mid_d = 0.5 * (lo_d + hi_d)
                mid_df = _apply_ops_delta_to_required_rates(
                    out, ref_mask=ref_mask, delta_z_ops=mid_d
                )
                mid_m = _median_ops_p50(mid_df)
                if not np.isfinite(mid_m):
                    break
                mid_err = abs(mid_m - target_mu)
                if mid_err < best_err:
                    best = mid_df
                    best_err = mid_err
                if (lo_m <= target_mu <= mid_m) or (mid_m <= target_mu <= lo_m):
                    hi_d, hi_m = mid_d, mid_m
                else:
                    lo_d, lo_m = mid_d, mid_m
            calibrated = best
    out = calibrated
    if "OPS_proj_p25" in out.columns and "OPS_proj_p75" in out.columns:
        out["OPS_proj_spread"] = (
            pd.to_numeric(out["OPS_proj_p75"], errors="coerce")
            - pd.to_numeric(out["OPS_proj_p25"], errors="coerce")
        )
    return out


def _recompute_pitching_surface_from_er_rate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_tags = ("p20", "p25", "p50", "p75", "p80", "p600")
    for tag in pct_tags:
        erip_col = f"ER_per_IP_mlb_eq_non_ar_delta_proj_{tag}"
        era_col = f"ERA_proj_{tag}"
        if erip_col not in out.columns:
            continue
        erip = pd.to_numeric(out[erip_col], errors="coerce")
        era = (erip * float(ERA_FROM_ERIP_FACTOR)).clip(lower=0.0)
        if era_col in out.columns:
            out[era_col] = era
    for base in ["ER_per_IP_mlb_eq_non_ar_delta", "ERA"]:
        c25 = f"{base}_proj_p25"
        c75 = f"{base}_proj_p75"
        spread = f"{base}_proj_spread"
        if c25 in out.columns and c75 in out.columns:
            out[spread] = (
                pd.to_numeric(out[c75], errors="coerce")
                - pd.to_numeric(out[c25], errors="coerce")
            )
    return out


def _apply_era_delta_to_er_rate(
    df: pd.DataFrame,
    *,
    ref_mask: pd.Series,
    delta_z_era: float,
) -> pd.DataFrame:
    out = df.copy()
    pct_tags = ("p20", "p25", "p50", "p75", "p80", "p600")
    for tag in pct_tags:
        col = f"ER_per_IP_mlb_eq_non_ar_delta_proj_{tag}"
        if col not in out.columns:
            continue
        mu, sigma = _reference_mu_sigma(out, col=col, ref_mask=ref_mask)
        if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 1e-12:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        z_vals = (vals - mu) / sigma
        target_mu = mu + (delta_z_era * sigma)
        out[col] = target_mu + (z_vals * sigma)
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0.0, upper=1.5)
    return _recompute_pitching_surface_from_er_rate(out)


def _apply_era_league_environment(
    df: pd.DataFrame,
    *,
    target_era_median: float,
    target_era_sigma: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if "ERA_proj_p50" not in out.columns or "ER_per_IP_mlb_eq_non_ar_delta_proj_p50" not in out.columns:
        return out
    source_mu, source_sigma, target_sigma_default = _era_env_reference_stats(out)
    target_sigma = (
        float(target_era_sigma)
        if target_era_sigma is not None and np.isfinite(float(target_era_sigma))
        else float(target_sigma_default)
    )
    if (
        not np.isfinite(source_mu)
        or not np.isfinite(source_sigma)
        or source_sigma <= 1e-12
        or not np.isfinite(target_sigma)
        or target_sigma <= 1e-12
    ):
        return out
    if not np.isfinite(float(target_era_median)):
        return out
    ref_mask = _pitching_reference_mask(out)
    target_mu = float(target_era_median)
    base_mu = source_mu
    delta_z_raw = (target_mu - base_mu) / target_sigma
    if abs(delta_z_raw) <= 1e-12:
        return out

    def _median_era_p50(frame: pd.DataFrame) -> float:
        vals = pd.to_numeric(frame.loc[ref_mask, "ERA_proj_p50"], errors="coerce")
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) == 0:
            return np.nan
        return float(vals.median())

    lo = 0.0
    hi = float(delta_z_raw)
    trial_hi = _apply_era_delta_to_er_rate(out, ref_mask=ref_mask, delta_z_era=hi)
    med_lo = base_mu
    med_hi = _median_era_p50(trial_hi)

    calibrated = trial_hi
    if np.isfinite(med_hi):
        target_between = (
            (med_lo <= target_mu <= med_hi) or (med_hi <= target_mu <= med_lo)
        )
        if target_between:
            best = trial_hi
            best_err = abs(med_hi - target_mu)
            lo_d, hi_d = lo, hi
            lo_m, hi_m = med_lo, med_hi
            for _ in range(10):
                mid_d = 0.5 * (lo_d + hi_d)
                mid_df = _apply_era_delta_to_er_rate(
                    out,
                    ref_mask=ref_mask,
                    delta_z_era=mid_d,
                )
                mid_m = _median_era_p50(mid_df)
                if not np.isfinite(mid_m):
                    break
                mid_err = abs(mid_m - target_mu)
                if mid_err < best_err:
                    best = mid_df
                    best_err = mid_err
                if (lo_m <= target_mu <= mid_m) or (mid_m <= target_mu <= lo_m):
                    hi_d, hi_m = mid_d, mid_m
                else:
                    lo_d, lo_m = mid_d, mid_m
            calibrated = best
    return calibrated


def _apply_whip_delta_to_whip_rate(
    df: pd.DataFrame,
    *,
    ref_mask: pd.Series,
    delta_z_whip: float,
) -> pd.DataFrame:
    out = df.copy()
    pct_tags = ("p20", "p25", "p50", "p75", "p80", "p600")
    bases = ("whip_mlb_eq_non_ar_delta", "WHIP")
    for base in bases:
        for tag in pct_tags:
            col = f"{base}_proj_{tag}"
            if col not in out.columns:
                continue
            mu, sigma = _reference_mu_sigma(out, col=col, ref_mask=ref_mask)
            if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 1e-12:
                continue
            vals = pd.to_numeric(out[col], errors="coerce")
            z_vals = (vals - mu) / sigma
            target_mu = mu + (delta_z_whip * sigma)
            out[col] = target_mu + (z_vals * sigma)
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=0.0, upper=4.0)
    for base in bases:
        c25 = f"{base}_proj_p25"
        c75 = f"{base}_proj_p75"
        spread = f"{base}_proj_spread"
        if c25 in out.columns and c75 in out.columns:
            out[spread] = (
                pd.to_numeric(out[c75], errors="coerce")
                - pd.to_numeric(out[c25], errors="coerce")
            )
    return out


def _apply_whip_league_environment(
    df: pd.DataFrame,
    *,
    target_whip_median: float,
    target_whip_sigma: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if "WHIP_proj_p50" not in out.columns:
        return out
    source_mu, source_sigma, target_sigma_default = _whip_env_reference_stats(out)
    target_sigma = (
        float(target_whip_sigma)
        if target_whip_sigma is not None and np.isfinite(float(target_whip_sigma))
        else float(target_sigma_default)
    )
    if (
        not np.isfinite(source_mu)
        or not np.isfinite(source_sigma)
        or source_sigma <= 1e-12
        or not np.isfinite(target_sigma)
        or target_sigma <= 1e-12
    ):
        return out
    if not np.isfinite(float(target_whip_median)):
        return out
    ref_mask = _pitching_reference_mask(out)
    target_mu = float(target_whip_median)
    base_mu = source_mu
    delta_z_raw = (target_mu - base_mu) / target_sigma
    if abs(delta_z_raw) <= 1e-12:
        return out

    def _median_whip_p50(frame: pd.DataFrame) -> float:
        vals = pd.to_numeric(frame.loc[ref_mask, "WHIP_proj_p50"], errors="coerce")
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) == 0:
            return np.nan
        return float(vals.median())

    lo = 0.0
    hi = float(delta_z_raw)
    trial_hi = _apply_whip_delta_to_whip_rate(out, ref_mask=ref_mask, delta_z_whip=hi)
    med_lo = base_mu
    med_hi = _median_whip_p50(trial_hi)

    calibrated = trial_hi
    if np.isfinite(med_hi):
        target_between = (
            (med_lo <= target_mu <= med_hi) or (med_hi <= target_mu <= med_lo)
        )
        if target_between:
            best = trial_hi
            best_err = abs(med_hi - target_mu)
            lo_d, hi_d = lo, hi
            lo_m, hi_m = med_lo, med_hi
            for _ in range(10):
                mid_d = 0.5 * (lo_d + hi_d)
                mid_df = _apply_whip_delta_to_whip_rate(
                    out,
                    ref_mask=ref_mask,
                    delta_z_whip=mid_d,
                )
                mid_m = _median_whip_p50(mid_df)
                if not np.isfinite(mid_m):
                    break
                mid_err = abs(mid_m - target_mu)
                if mid_err < best_err:
                    best = mid_df
                    best_err = mid_err
                if (lo_m <= target_mu <= mid_m) or (mid_m <= target_mu <= lo_m):
                    hi_d, hi_m = mid_d, mid_m
                else:
                    lo_d, lo_m = mid_d, mid_m
            calibrated = best
    return calibrated


def _add_derived_slashline_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _set_or_fill(col: str, vals: pd.Series) -> None:
        if col in out.columns:
            existing = pd.to_numeric(out[col], errors="coerce")
            out[col] = existing.where(existing.notna(), vals)
        else:
            out[col] = vals

    pct_tags = ["p20", "p25", "p50", "p75", "p80", "p600"]
    for tag in pct_tags:
        pa_col = f"PA_proj_{tag}"
        if pa_col not in out.columns:
            continue

        needed = {
            "bb": f"BB_proj_{tag}",
            "hbp": f"HBP_proj_{tag}",
            "sh": f"SH_proj_{tag}",
            "sf": f"SF_proj_{tag}",
            "so": f"SO_proj_{tag}",
            "s1b": f"1B_proj_{tag}",
            "s2b": f"2B_proj_{tag}",
            "s3b": f"3B_proj_{tag}",
            "hr": f"HR_proj_{tag}",
        }
        if not all(col in out.columns for col in needed.values()):
            continue

        pa = pd.to_numeric(out[pa_col], errors="coerce")
        bb = pd.to_numeric(out[needed["bb"]], errors="coerce")
        hbp = pd.to_numeric(out[needed["hbp"]], errors="coerce")
        sh = pd.to_numeric(out[needed["sh"]], errors="coerce")
        sf = pd.to_numeric(out[needed["sf"]], errors="coerce")
        so = pd.to_numeric(out[needed["so"]], errors="coerce")
        one_b = pd.to_numeric(out[needed["s1b"]], errors="coerce")
        two_b = pd.to_numeric(out[needed["s2b"]], errors="coerce")
        three_b = pd.to_numeric(out[needed["s3b"]], errors="coerce")
        hr = pd.to_numeric(out[needed["hr"]], errors="coerce")

        ab = (pa - (bb + hbp + sh + sf)).clip(lower=0.0)
        hits = (one_b + two_b + three_b + hr).clip(lower=0.0)
        total_bases = (one_b + 2.0 * two_b + 3.0 * three_b + 4.0 * hr).clip(lower=0.0)

        avg = _safe_divide(hits, ab)
        obp = _safe_divide(hits + bb + hbp, pa)
        slg = _safe_divide(total_bases, ab)
        ops = obp + slg
        k_pct = _safe_divide(so, pa) * 100.0
        bb_pct = _safe_divide(bb, pa) * 100.0

        _set_or_fill(f"AB_proj_{tag}", ab)
        _set_or_fill(f"H_proj_{tag}", hits)
        _set_or_fill(f"AVG_proj_{tag}", avg)
        _set_or_fill(f"OBP_proj_{tag}", obp)
        _set_or_fill(f"SLG_proj_{tag}", slg)
        _set_or_fill(f"OPS_proj_{tag}", ops)
        _set_or_fill(f"K%_proj_{tag}", k_pct)
        _set_or_fill(f"BB%_proj_{tag}", bb_pct)

    # Z-scores for p50 slashline using MLB-appearance subset as the reference population.
    # Use population std (ddof=0) to match the requested interpretation.
    z_targets = ["AVG", "OBP", "SLG", "OPS"]
    ref_mask = _ops_reference_mask(out)

    for stat in z_targets:
        p50_col = f"{stat}_proj_p50"
        z_col = f"{stat}_z_mlbT_proj_p50"
        if p50_col not in out.columns:
            out[z_col] = np.nan
            continue
        vals = pd.to_numeric(out[p50_col], errors="coerce")
        ref_vals = vals.loc[ref_mask].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ref_vals) < 2:
            out[z_col] = np.nan
            continue
        mu = float(ref_vals.median())
        sigma = float(ref_vals.std(ddof=0))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            out[z_col] = np.nan
            continue
        out[z_col] = (vals - mu) / sigma
    return out


def _add_derived_pitcher_kpi_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Convert raw stuff projections to 20-80 pitch grade using the same shape as data_aggregate:
    # grade = clip(80 - 60 * (raw - p01) / (p99 - p01), 20, 80), rounded to integer.
    stuff_base = None
    if "stuff_raw_proj_p50" in out.columns:
        stuff_base = "stuff_raw"
    elif "stuff_raw_reg_proj_p50" in out.columns:
        stuff_base = "stuff_raw_reg"
    if stuff_base is not None:
        src_col = f"source_{stuff_base}"
        p50_col = f"{stuff_base}_proj_p50"
        p01, p99 = _load_stuff_grade_reference_2025_mlb()
        if not (np.isfinite(p01) and np.isfinite(p99) and (abs(p99 - p01) > 1e-12)):
            # Fallback: derive from MLB rows in the current frame.
            ref_mask = _pitching_reference_mask(out)
            if "source_season" in out.columns:
                src_season = pd.to_numeric(out["source_season"], errors="coerce")
                ref_mask = ref_mask & src_season.eq(float(STUFF_GRADE_REF_SEASON))
            if "level_id_source" in out.columns:
                src_level = pd.to_numeric(out["level_id_source"], errors="coerce")
                ref_mask = ref_mask & src_level.eq(float(STUFF_GRADE_REF_LEVEL_ID))
            ref_col = src_col if src_col in out.columns else p50_col
            ref_vals = (
                pd.to_numeric(out.loc[ref_mask, ref_col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(ref_vals) >= 2:
                p01 = float(ref_vals.quantile(0.01))
                p99 = float(ref_vals.quantile(0.99))
        if np.isfinite(p01) and np.isfinite(p99) and (abs(p99 - p01) > 1e-12):
            def _stuff_grade(s: pd.Series) -> pd.Series:
                raw = pd.to_numeric(s, errors="coerce")
                grade = 80.0 - 60.0 * (raw - p01) / (p99 - p01)
                return grade.clip(lower=20.0, upper=80.0).round(0)

            for pct in ("p20", "p25", "p50", "p75", "p80", "p600"):
                c = f"{stuff_base}_proj_{pct}"
                if c in out.columns:
                    out[c] = _stuff_grade(out[c])
            if src_col in out.columns:
                out[src_col] = _stuff_grade(out[src_col])

    # Mirror damage_streamlit "Zone (%)" by deriving from Ball% when Zone isn't explicitly projected.
    for pct in ("p20", "p25", "p50", "p75", "p80"):
        zone_col = f"Zone_reg_proj_{pct}"
        ball_col = f"Ball_pct_reg_proj_{pct}"
        if zone_col in out.columns or ball_col not in out.columns:
            continue
        ball_vals = pd.to_numeric(out[ball_col], errors="coerce")
        out[zone_col] = 100.0 - ball_vals
    if "source_Zone_reg" not in out.columns and "source_Ball_pct_reg" in out.columns:
        out["source_Zone_reg"] = 100.0 - pd.to_numeric(out["source_Ball_pct_reg"], errors="coerce")
    if "n_eff_Zone_reg" not in out.columns and "n_eff_Ball_pct_reg" in out.columns:
        out["n_eff_Zone_reg"] = pd.to_numeric(out["n_eff_Ball_pct_reg"], errors="coerce")
    zone_p25 = "Zone_reg_proj_p25"
    zone_p75 = "Zone_reg_proj_p75"
    if zone_p25 in out.columns and zone_p75 in out.columns:
        out["Zone_reg_proj_spread"] = (
            pd.to_numeric(out[zone_p75], errors="coerce")
            - pd.to_numeric(out[zone_p25], errors="coerce")
        )

    # Derived pitching skill delta: K-BB% = K% - BB%.
    for pct in ("p20", "p25", "p50", "p75", "p80", "p600"):
        k_col = f"K%_proj_{pct}"
        bb_col = f"BB%_proj_{pct}"
        out_col = f"K-BB%_proj_{pct}"
        if k_col in out.columns and bb_col in out.columns:
            out[out_col] = (
                pd.to_numeric(out[k_col], errors="coerce")
                - pd.to_numeric(out[bb_col], errors="coerce")
            )
    if "K-BB%_proj_p25" in out.columns and "K-BB%_proj_p75" in out.columns:
        out["K-BB%_proj_spread"] = (
            pd.to_numeric(out["K-BB%_proj_p75"], errors="coerce")
            - pd.to_numeric(out["K-BB%_proj_p25"], errors="coerce")
        )
    return out


def _add_derived_points_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_tags = ("p20", "p25", "p50", "p75", "p80", "p600")
    sgp_den = SGP_DENOMINATORS
    sgp_base = SGP_BASELINES

    def _num(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype="float64")
        return pd.to_numeric(out[col], errors="coerce")

    def _num_any(cols: list[str]) -> pd.Series:
        for col in cols:
            if col in out.columns:
                return pd.to_numeric(out[col], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype="float64")

    hitter_inputs = ["AB", "H", "Runs", "HR", "RBI", "SB"]
    pitcher_inputs = ["IP", "H", "ER", "BB", "SO", "W", "SV", "ERA"]
    hitter_present = sum(1 for base in hitter_inputs if f"{base}_proj_p50" in out.columns)
    pitcher_present = sum(1 for base in pitcher_inputs if f"{base}_proj_p50" in out.columns)
    use_pitching_formula = pitcher_present > hitter_present

    for tag in pct_tags:
        if use_pitching_formula:
            ip = _num(f"IP_proj_{tag}")
            h_allowed = _num(f"H_proj_{tag}")
            er = _num(f"ER_proj_{tag}")
            if f"ER_proj_{tag}" not in out.columns:
                era = _num(f"ERA_proj_{tag}")
                er = (era * ip / 9.0).where(
                    ip.notna() & np.isfinite(ip) & (ip > 0.0),
                    np.nan,
                )
            bb = _num(f"BB_proj_{tag}")
            so = _num(f"SO_proj_{tag}")
            w = _num(f"W_proj_{tag}")
            sv = _num(f"SV_proj_{tag}")
            components = {
                "ip": ip,
                "h": h_allowed,
                "er": er,
                "bb": bb,
                "so": so,
                "w": w,
                "sv": sv,
            }
            if not any(
                (series.notna() & np.isfinite(series)).any()
                for series in components.values()
            ):
                continue
            points = (
                3.0 * ip.fillna(0.0)
                - 1.0 * h_allowed.fillna(0.0)
                - 2.0 * er.fillna(0.0)
                - 1.0 * bb.fillna(0.0)
                + 1.0 * so.fillna(0.0)
                + 6.0 * w.fillna(0.0)
                + 8.0 * sv.fillna(0.0)
            )
            out[f"Points_proj_{tag}"] = points
        else:
            ab = _num(f"AB_proj_{tag}")
            h = _num(f"H_proj_{tag}")
            runs = _num(f"Runs_proj_{tag}")
            hr = _num(f"HR_proj_{tag}")
            rbi = _num(f"RBI_proj_{tag}")
            sb = _num(f"SB_proj_{tag}")
            components = {
                "ab": ab,
                "h": h,
                "runs": runs,
                "hr": hr,
                "rbi": rbi,
                "sb": sb,
            }
            if not any(
                (series.notna() & np.isfinite(series)).any()
                for series in components.values()
            ):
                continue
            points = (
                -1.0 * ab.fillna(0.0)
                + 4.0 * h.fillna(0.0)
                + 2.0 * runs.fillna(0.0)
                + 6.0 * hr.fillna(0.0)
                + 2.0 * rbi.fillna(0.0)
                + 5.0 * sb.fillna(0.0)
            )
            out[f"Points_proj_{tag}"] = points

        pts_col = f"Points_proj_{tag}"
        if pts_col in out.columns:
            points_vals = pd.to_numeric(out[pts_col], errors="coerce")
            pa_col = f"PA_proj_{tag}"
            if pa_col in out.columns:
                pa_vals = pd.to_numeric(out[pa_col], errors="coerce")
                pa_den = pa_vals.where(pa_vals.abs() > 1e-12, np.nan)
                out[f"Points/PA_proj_{tag}"] = points_vals / pa_den
            ip_col = f"IP_proj_{tag}"
            ip_fallback_col = f"IP_marcel_proj_{tag}"
            if ip_col in out.columns or ip_fallback_col in out.columns:
                ip_vals = (
                    pd.to_numeric(out[ip_col], errors="coerce")
                    if ip_col in out.columns
                    else pd.to_numeric(out[ip_fallback_col], errors="coerce")
                )
                ip_den = ip_vals.where(ip_vals.abs() > 1e-12, np.nan)
                out[f"Points/IP_proj_{tag}"] = points_vals / ip_den

        # Roto SGP (15-team 5x5) calibrated from 2025 Draft Champions standings.
        runs = _num_any([f"Runs_proj_{tag}", f"R_proj_{tag}"])
        hr = _num_any([f"HR_proj_{tag}"])
        rbi = _num_any([f"RBI_proj_{tag}"])
        sb = _num_any([f"SB_proj_{tag}"])
        ab = _num_any([f"AB_proj_{tag}"])
        h = _num_any([f"H_proj_{tag}"])
        w = _num_any([f"W_proj_{tag}"])
        sv = _num_any([f"SV_proj_{tag}"])
        so = _num_any([f"SO_proj_{tag}", f"K_proj_{tag}"])
        ip = _num_any([f"IP_proj_{tag}", f"IP_marcel_proj_{tag}"])
        bb = _num_any([f"BB_proj_{tag}"])
        h_allowed = _num_any([f"H_proj_{tag}"])
        er = _num_any([f"ER_proj_{tag}"])
        if f"ER_proj_{tag}" not in out.columns:
            era = _num_any([f"ERA_proj_{tag}"])
            er = (era * ip / 9.0).where(ip.notna() & np.isfinite(ip) & (ip > 0.0), np.nan)
        if f"H_proj_{tag}" not in out.columns:
            whip = _num_any([f"WHIP_proj_{tag}"])
            h_allowed = ((whip * ip) - bb).where(ip.notna() & np.isfinite(ip) & (ip > 0.0), np.nan)

        sgp_runs = (runs / sgp_den["R"]).fillna(0.0)
        sgp_hr = (hr / sgp_den["HR"]).fillna(0.0)
        sgp_rbi = (rbi / sgp_den["RBI"]).fillna(0.0)
        sgp_sb = (sb / sgp_den["SB"]).fillna(0.0)
        ab_den = sgp_base["AB0"] + ab
        avg_after = ((sgp_base["H0"] + h) / ab_den).where(ab_den.abs() > 1e-12, np.nan)
        sgp_avg = ((avg_after - sgp_base["AVG0"]) / sgp_den["AVG"]).fillna(0.0)

        sgp_w = (w / sgp_den["W"]).fillna(0.0)
        sgp_sv = (sv / sgp_den["SV"]).fillna(0.0)
        sgp_so = (so / sgp_den["SO"]).fillna(0.0)
        ip_den = sgp_base["IP0"] + ip
        era_after = (((sgp_base["ER0"] + er) * 9.0) / ip_den).where(ip_den.abs() > 1e-12, np.nan)
        sgp_era = ((sgp_base["ERA0"] - era_after) / sgp_den["ERA"]).fillna(0.0)
        whip_after = ((sgp_base["BBH0"] + bb + h_allowed) / ip_den).where(
            ip_den.abs() > 1e-12,
            np.nan,
        )
        sgp_whip = ((sgp_base["WHIP0"] - whip_after) / sgp_den["WHIP"]).fillna(0.0)

        sgp_hit = sgp_runs + sgp_hr + sgp_rbi + sgp_sb + sgp_avg
        sgp_pitch = sgp_w + sgp_sv + sgp_so + sgp_era + sgp_whip
        out[f"SGP_hit_proj_{tag}"] = sgp_hit
        out[f"SGP_pitch_proj_{tag}"] = sgp_pitch
        out[f"SGP_proj_{tag}"] = sgp_hit + sgp_pitch

    if (
        "Points_proj_p25" in out.columns
        and "Points_proj_p75" in out.columns
    ):
        out["Points_proj_spread"] = (
            pd.to_numeric(out["Points_proj_p75"], errors="coerce")
            - pd.to_numeric(out["Points_proj_p25"], errors="coerce")
        )
    if "Points/PA_proj_p25" in out.columns and "Points/PA_proj_p75" in out.columns:
        out["Points/PA_proj_spread"] = (
            pd.to_numeric(out["Points/PA_proj_p75"], errors="coerce")
            - pd.to_numeric(out["Points/PA_proj_p25"], errors="coerce")
        )
    if "Points/IP_proj_p25" in out.columns and "Points/IP_proj_p75" in out.columns:
        out["Points/IP_proj_spread"] = (
            pd.to_numeric(out["Points/IP_proj_p75"], errors="coerce")
            - pd.to_numeric(out["Points/IP_proj_p25"], errors="coerce")
        )
    if "SGP_proj_p25" in out.columns and "SGP_proj_p75" in out.columns:
        out["SGP_proj_spread"] = (
            pd.to_numeric(out["SGP_proj_p75"], errors="coerce")
            - pd.to_numeric(out["SGP_proj_p25"], errors="coerce")
        )
    if "SGP_hit_proj_p25" in out.columns and "SGP_hit_proj_p75" in out.columns:
        out["SGP_hit_proj_spread"] = (
            pd.to_numeric(out["SGP_hit_proj_p75"], errors="coerce")
            - pd.to_numeric(out["SGP_hit_proj_p25"], errors="coerce")
        )
    if "SGP_pitch_proj_p25" in out.columns and "SGP_pitch_proj_p75" in out.columns:
        out["SGP_pitch_proj_spread"] = (
            pd.to_numeric(out["SGP_pitch_proj_p75"], errors="coerce")
            - pd.to_numeric(out["SGP_pitch_proj_p25"], errors="coerce")
        )
    return out


def _apply_fixed_volume_projection_view(
    df: pd.DataFrame,
    *,
    target_pa: float | None = None,
    target_ip: float | None = None,
) -> pd.DataFrame:
    out = df.copy()

    def _num(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype="float64")
        return pd.to_numeric(out[col], errors="coerce")

    pct_tags = ("p20", "p25", "p50", "p75", "p80", "p600")

    if (target_pa is not None) and (target_pa > 0.0):
        hitter_bases = [
            "PA",
            "AB",
            "H",
            "1B",
            "2B",
            "3B",
            "HR",
            "Runs",
            "RBI",
            "SB",
            "CS",
            "SBA",
            "BB",
            "SO",
            "HBP",
            "SF",
            "SH",
            "bbe",
            "BIP",
            "Points",
            "SGP",
        ]
        for tag in pct_tags:
            pa_col = f"PA_proj_{tag}"
            if pa_col not in out.columns:
                continue
            pa_vals = _num(pa_col)
            pa_den = pa_vals.where(pa_vals.abs() > 1e-12, np.nan)
            tag_scale = float(target_pa) / pa_den
            for base in hitter_bases:
                src_col = f"{base}_proj_{tag}"
                if src_col in out.columns:
                    out[src_col] = _num(src_col) * tag_scale

    if (target_ip is not None) and (target_ip > 0.0):
        pitcher_bases = [
            "IP",
            "IP_marcel",
            "TBF",
            "TBF_marcel",
            "G",
            "G_marcel",
            "GS",
            "GS_marcel",
            "W",
            "SV",
            "SO",
            "K",
            "BB",
            "H",
            "HR",
            "HBP",
            "ER",
            "Points",
            "SGP",
        ]
        for tag in pct_tags:
            ip_col = (
                f"IP_proj_{tag}"
                if f"IP_proj_{tag}" in out.columns
                else (f"IP_marcel_proj_{tag}" if f"IP_marcel_proj_{tag}" in out.columns else "")
            )
            if not ip_col:
                continue
            ip_vals = _num(ip_col)
            ip_den = ip_vals.where(ip_vals.abs() > 1e-12, np.nan)
            tag_scale = float(target_ip) / ip_den
            for base in pitcher_bases:
                src_col = f"{base}_proj_{tag}"
                if src_col in out.columns:
                    out[src_col] = _num(src_col) * tag_scale

    return _recompute_projection_spreads(out)


def _metric_base_from_proj_col(col: str) -> str:
    txt = str(col)
    txt_l = txt.lower()
    if txt_l.endswith(" p50"):
        return txt[: -len(" p50")]
    if txt_l.endswith(" p25"):
        return txt[: -len(" p25")]
    if txt_l.endswith(" p75"):
        return txt[: -len(" p75")]
    if txt_l.endswith(" p20"):
        return txt[: -len(" p20")]
    if txt_l.endswith(" p80"):
        return txt[: -len(" p80")]
    if txt_l.endswith(" p600"):
        return txt[: -len(" p600")]
    suffixes = [
        "_proj_p20",
        "_proj_p25",
        "_proj_p50",
        "_proj_p75",
        "_proj_p80",
        "_proj_p600",
        "_proj_spread",
    ]
    for suf in suffixes:
        if txt.endswith(suf):
            return txt[: -len(suf)]
    return txt


def _display_label_for_projection_col(col: str) -> str:
    text = str(col)
    if text in DISPLAY_COL_ALIASES:
        return DISPLAY_COL_ALIASES[text]
    for pct in ("p25", "p50", "p75"):
        suffix = f"_proj_{pct}"
        if text.endswith(suffix):
            base = text[: -len(suffix)]
            label = P50_DISPLAY_ALIASES.get(base, base)
            return f"{label} {pct}"
    return text


def _flip_percentile_values_for_bases(
    df: pd.DataFrame,
    *,
    bases: set[str],
) -> pd.DataFrame:
    out = df.copy()
    for base in bases:
        for suffix in ("", "_after", "_before"):
            for low_pct, high_pct in (("p25", "p75"), ("p20", "p80")):
                c_lo = f"{base}_proj_{low_pct}{suffix}"
                c_hi = f"{base}_proj_{high_pct}{suffix}"
                if c_lo in out.columns and c_hi in out.columns:
                    lo_vals = out[c_lo].copy()
                    out[c_lo] = out[c_hi]
                    out[c_hi] = lo_vals
    return out


def _team_tokens(v: object) -> list[str]:
    text = str(v or "").strip().upper()
    if not text:
        return []
    raw = text.replace(",", "|").replace("/", "|").replace(";", "|")
    return [t.strip() for t in raw.split("|") if t.strip()]


def _team_col(df: pd.DataFrame) -> str | None:
    for col in TEAM_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _normalize_ascii_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_team_token(token: object) -> str:
    text = str(token or "").strip().upper()
    if not text:
        return ""
    return TEAM_ABBREV_NORMALIZATION.get(text, text)


def _normalized_team_tokens(value: object) -> tuple[str, ...]:
    raw = str(value or "").strip().upper()
    if not raw:
        return tuple()
    split_raw = raw.replace(",", "|").replace("/", "|").replace(";", "|")
    out = {
        _normalize_team_token(tok)
        for tok in (t.strip() for t in split_raw.split("|"))
        if tok
    }
    return tuple(sorted(t for t in out if t))


def _normalize_position_token(token: object) -> str:
    text = str(token or "").strip().upper()
    if not text:
        return ""
    return POSITION_TOKEN_NORMALIZATION.get(text, text)


def _position_tokens_from_text(value: object) -> tuple[str, ...]:
    raw = str(value or "").strip().upper()
    if not raw:
        return tuple()
    split_raw = raw.replace("/", ",").replace("|", ",").replace(";", ",")
    out = {
        _normalize_position_token(tok)
        for tok in (t.strip() for t in split_raw.split(","))
        if tok
    }
    return tuple(sorted(t for t in out if t))


def _position_tokens_from_row(row: pd.Series, *, for_pitchers: bool) -> tuple[str, ...]:
    out: set[str] = set()
    if "position" in row.index:
        out.update(_position_tokens_from_text(row.get("position")))

    for pos in POSITION_FILTER_COLS:
        for col in (f"is_{pos}", pos):
            if col not in row.index:
                continue
            val = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
            if pd.notna(val) and float(val) >= 1.0:
                out.add(_normalize_position_token(pos))
                break

    if for_pitchers and not out:
        out.add("P")
    return tuple(sorted(t for t in out if t))


def _tokens_intersect(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if not left or not right:
        return False
    return bool(set(left).intersection(right))


def _adp_to_pick_notation(value: object, *, teams: int = ADP_DRAFT_TEAMS) -> str:
    adp_val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(adp_val) or (not np.isfinite(float(adp_val))) or float(adp_val) <= 0.0:
        return ""
    overall_pick = int(np.floor(float(adp_val) + 0.5))
    round_num = ((overall_pick - 1) // int(teams)) + 1
    pick_in_round = ((overall_pick - 1) % int(teams)) + 1
    return f"{int(round_num)}.{int(pick_in_round)}"


@st.cache_resource(show_spinner=False)
def _load_adp_lookup() -> pd.DataFrame:
    if not ADP_PATH.exists():
        return pd.DataFrame()
    try:
        raw = pd.read_csv(ADP_PATH, sep="\t")
    except Exception:
        return pd.DataFrame()
    if raw.empty or ("Player" not in raw.columns) or ("ADP" not in raw.columns):
        return pd.DataFrame()

    work = raw.copy()
    player_series = work["Player"].fillna("").astype(str)
    split_names = player_series.str.split(",", n=1, expand=True)
    if split_names.shape[1] >= 2:
        work["adp_last_name"] = split_names[0].fillna("").astype(str).str.strip()
        work["adp_first_name"] = split_names[1].fillna("").astype(str).str.strip()
    else:
        work["adp_last_name"] = player_series.str.strip()
        work["adp_first_name"] = ""
    full_name = (
        work["adp_first_name"].fillna("").astype(str).str.strip()
        + " "
        + work["adp_last_name"].fillna("").astype(str).str.strip()
    ).str.strip()
    work["adp_name_key"] = full_name.map(_normalize_ascii_text)
    fallback_mask = work["adp_name_key"].eq("")
    if bool(fallback_mask.any()):
        work.loc[fallback_mask, "adp_name_key"] = player_series.loc[fallback_mask].map(
            _normalize_ascii_text
        )

    if "Team" in work.columns:
        team_series = work["Team"]
    else:
        team_series = pd.Series("", index=work.index)
    if "Position(s)" in work.columns:
        pos_series = work["Position(s)"]
    else:
        pos_series = pd.Series("", index=work.index)

    work["adp_team_token"] = team_series.fillna("").map(_normalize_team_token)
    work["adp_position_tokens"] = pos_series.fillna("").map(_position_tokens_from_text)
    work["ADP"] = pd.to_numeric(work["ADP"], errors="coerce")
    work["adp_rank"] = pd.to_numeric(work.get("Rank"), errors="coerce")
    work = work[work["adp_name_key"].ne("")].copy()

    keep_cols = [
        "adp_name_key",
        "adp_first_name",
        "adp_last_name",
        "adp_team_token",
        "adp_position_tokens",
        "adp_rank",
        "ADP",
    ]
    return work[keep_cols]


def _attach_adp_context(df: pd.DataFrame, *, for_pitchers: bool) -> pd.DataFrame:
    if df.empty:
        return df
    adp = _load_adp_lookup()
    if adp.empty:
        return df

    name_candidates = (
        ["name", "player_name", "hitter_name"]
        if for_pitchers
        else ["hitter_name", "player_name", "name"]
    )
    name_col = next((c for c in name_candidates if c in df.columns), None)
    if name_col is None:
        return df

    out = df.copy()
    out = out.drop(columns=["ADP"], errors="ignore")
    out["_adp_row_id"] = np.arange(len(out), dtype="int64")
    out["adp_name_key"] = out[name_col].map(_normalize_ascii_text)

    team_col = _team_col(out)
    if team_col is None and "team_40man" in out.columns:
        team_col = "team_40man"
    if team_col is None:
        out["_adp_team_tokens"] = [tuple()] * len(out)
    else:
        out["_adp_team_tokens"] = out[team_col].map(_normalized_team_tokens)

    out["_adp_position_tokens"] = out.apply(
        lambda row: _position_tokens_from_row(row, for_pitchers=for_pitchers),
        axis=1,
    )

    probe_cols = [
        "_adp_row_id",
        "adp_name_key",
        "_adp_team_tokens",
        "_adp_position_tokens",
    ]
    candidates = out[probe_cols].merge(adp, on="adp_name_key", how="left")
    candidates = candidates[candidates["ADP"].notna()].copy()
    if candidates.empty:
        out["ADP"] = pd.to_numeric(out.get("ADP"), errors="coerce")
        return out.drop(
            columns=[
                "_adp_row_id",
                "adp_name_key",
                "_adp_team_tokens",
                "_adp_position_tokens",
            ],
            errors="ignore",
        )

    candidates["team_match"] = candidates.apply(
        lambda row: bool(row.get("adp_team_token"))
        and (str(row.get("adp_team_token")) in set(row.get("_adp_team_tokens", tuple()))),
        axis=1,
    )
    candidates["position_match"] = candidates.apply(
        lambda row: _tokens_intersect(
            tuple(row.get("_adp_position_tokens", tuple())),
            tuple(row.get("adp_position_tokens", tuple())),
        ),
        axis=1,
    )
    candidates["adp_match_score"] = (
        candidates["team_match"].astype("int64") * 2
        + candidates["position_match"].astype("int64")
    )

    selected_rows: list[dict[str, float | int]] = []
    for row_id, grp in candidates.groupby("_adp_row_id", sort=False):
        if len(grp) == 1:
            sel = grp.iloc[0]
        else:
            top_score = int(pd.to_numeric(grp["adp_match_score"], errors="coerce").max())
            if top_score <= 0:
                continue
            best = grp[grp["adp_match_score"] == top_score]
            if len(best) != 1:
                continue
            sel = best.iloc[0]
        selected_rows.append(
            {
                "_adp_row_id": int(row_id),
                "ADP": float(sel["ADP"]),
            }
        )

    if selected_rows:
        picked = pd.DataFrame(selected_rows)
        out = out.merge(picked, on="_adp_row_id", how="left")
    else:
        out["ADP"] = np.nan

    out["ADP"] = pd.to_numeric(out["ADP"], errors="coerce")
    out["Pick"] = out["ADP"].map(_adp_to_pick_notation)
    return out.drop(
        columns=[
            "_adp_row_id",
            "adp_name_key",
            "_adp_team_tokens",
            "_adp_position_tokens",
        ],
        errors="ignore",
    )


def _roster_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _roster_parse_int(value: object) -> int | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed) or (not np.isfinite(float(parsed))):
        return None
    return int(parsed)


def _roster_parse_nonneg_float(value: object) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed) or (not np.isfinite(float(parsed))):
        return None
    out = float(parsed)
    if out < 0.0:
        return None
    return out


def _roster_default_manual_projection_counts(*, hitter: bool) -> dict[str, float]:
    keys = (
        ROSTER_HITTER_MANUAL_COUNT_STATS
        if hitter
        else ROSTER_PITCHER_MANUAL_COUNT_STATS
    )
    return {key: 0.0 for key in keys}


def _roster_normalize_placeholder_manual_stats(
    value: object,
    *,
    hitter: bool,
) -> dict[str, float]:
    out = _roster_default_manual_projection_counts(hitter=hitter)
    if not isinstance(value, dict):
        return out
    for key in out:
        parsed = _roster_parse_nonneg_float(value.get(key))
        if parsed is not None:
            out[key] = float(parsed)
    return out


def _roster_normalize_manual_stats(
    value: object,
    *,
    hitter: bool,
) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    keys = (
        ROSTER_HITTER_MANUAL_COUNT_STATS
        if hitter
        else ROSTER_PITCHER_MANUAL_COUNT_STATS
    )
    out: dict[str, float] = {}
    any_value = False
    for key in keys:
        parsed = _roster_parse_nonneg_float(value.get(key))
        if parsed is None:
            out[key] = 0.0
        else:
            out[key] = float(parsed)
            any_value = True
    if not any_value:
        return None
    return out


def _roster_tokens_from_input(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        raw = ",".join(str(item) for item in value)
    else:
        raw = str(value or "")
    return _position_tokens_from_text(raw)


def _roster_normalize_custom_player_entry(
    value: object,
    *,
    hitter: bool,
) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    player_id = _roster_parse_int(value.get("player_id"))
    if player_id is None or player_id >= 0:
        return None

    fallback_name = f"Placeholder {'Hitter' if hitter else 'Pitcher'} {abs(int(player_id))}"
    name = str(value.get("name") or "").strip() or fallback_name
    team_raw = str(value.get("team") or "").strip()
    team = str(_normalize_team_token(team_raw) or team_raw or "FA").upper().strip() or "FA"

    if hitter:
        positions = tuple(
            sorted(
                {
                    tok
                    for tok in _roster_tokens_from_input(value.get("positions"))
                    if tok in set(ROSTER_PLACEHOLDER_HITTER_POSITIONS)
                }
            )
        )
        manual_stats = _roster_normalize_placeholder_manual_stats(
            value.get("manual_stats"),
            hitter=True,
        )
        plate_appearances = _roster_parse_nonneg_float(value.get("plate_appearances"))
        if plate_appearances is None:
            plate_appearances = float(manual_stats.get("AB", 0.0))
        return {
            "player_id": int(player_id),
            "name": name,
            "team": team,
            "positions": list(positions),
            "plate_appearances": float(plate_appearances),
            "manual_stats": manual_stats,
        }

    roles = tuple(
        sorted(
            {
                tok
                for tok in _roster_tokens_from_input(value.get("roles"))
                if tok in set(ROSTER_PLACEHOLDER_PITCHER_ROLES)
            }
        )
    )
    if not roles:
        roles = ("SP",)
    return {
        "player_id": int(player_id),
        "name": name,
        "team": team,
        "roles": list(roles),
        "manual_stats": _roster_normalize_placeholder_manual_stats(
            value.get("manual_stats"),
            hitter=False,
        ),
    }


def _roster_normalize_custom_player_lists(
    raw_hitters: object,
    raw_pitchers: object,
    *,
    next_player_id_raw: object = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    used_ids: set[int] = set()

    def _clean(values: object, *, hitter: bool) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        if not isinstance(values, list):
            return out
        for item in values:
            normalized = _roster_normalize_custom_player_entry(item, hitter=hitter)
            if normalized is None:
                continue
            player_id = int(normalized["player_id"])
            if player_id in used_ids:
                continue
            out.append(normalized)
            used_ids.add(player_id)
        return out

    custom_hitters = _clean(raw_hitters, hitter=True)
    custom_pitchers = _clean(raw_pitchers, hitter=False)
    min_existing_id = min([0, *used_ids]) if used_ids else 0
    next_player_id = _roster_parse_int(next_player_id_raw)
    if next_player_id is None or next_player_id >= 0 or next_player_id >= min_existing_id:
        next_player_id = int(min_existing_id - 1)
    return custom_hitters, custom_pitchers, int(next_player_id)


def _roster_normalize_custom_players_in_state(state: dict) -> None:
    custom_hitters, custom_pitchers, next_player_id = _roster_normalize_custom_player_lists(
        state.get("custom_hitters"),
        state.get("custom_pitchers"),
        next_player_id_raw=state.get("next_custom_player_id"),
    )
    state["custom_hitters"] = custom_hitters
    state["custom_pitchers"] = custom_pitchers
    state["next_custom_player_id"] = int(next_player_id)


def _roster_add_custom_player(state: dict, *, hitter: bool) -> int:
    _roster_normalize_custom_players_in_state(state)
    next_player_id = _roster_parse_int(state.get("next_custom_player_id"))
    if next_player_id is None or next_player_id >= 0:
        next_player_id = -1
    entry: dict[str, object] = {
        "player_id": int(next_player_id),
        "name": "",
        "team": "FA",
        "manual_stats": _roster_default_manual_projection_counts(hitter=hitter),
    }
    if hitter:
        entry["positions"] = []
        entry["plate_appearances"] = 0.0
        state.setdefault("custom_hitters", []).append(entry)
    else:
        entry["roles"] = ["SP"]
        state.setdefault("custom_pitchers", []).append(entry)
    state["next_custom_player_id"] = int(next_player_id - 1)
    return int(next_player_id)


def _roster_clear_player_from_state(state: dict, player_id: int) -> None:
    for slot in ROSTER_ALL_STARTER_SLOTS:
        if _roster_parse_int(state.get("starters", {}).get(slot)) != int(player_id):
            continue
        state["starters"][slot] = None
        if isinstance(state.get("starter_custom_volume"), dict):
            state["starter_custom_volume"][slot] = None
        if isinstance(state.get("starter_manual_stats"), dict):
            state["starter_manual_stats"][slot] = None
    for reserve_key in ("hitter_reserves", "pitcher_reserves"):
        reserves = state.get(reserve_key, [])
        if not isinstance(reserves, list):
            continue
        for row in reserves:
            if not isinstance(row, dict):
                continue
            if _roster_parse_int(row.get("player_id")) != int(player_id):
                continue
            row["player_id"] = None
            row["custom_volume"] = None
            row["manual_stats"] = None


def _roster_hitter_stats_from_manual(
    manual: dict[str, float] | None,
) -> dict[str, float] | None:
    norm = _roster_normalize_manual_stats(manual, hitter=True)
    if norm is None:
        return None
    out = {k: float(norm.get(k, 0.0)) for k in ROSTER_HITTER_MANUAL_COUNT_STATS}
    ab = float(out.get("AB", 0.0))
    h = float(out.get("H", 0.0))
    out["AVG"] = (h / ab) if ab > 0.0 else float("nan")
    return out


def _roster_pitcher_stats_from_manual(
    manual: dict[str, float] | None,
) -> dict[str, float] | None:
    norm = _roster_normalize_manual_stats(manual, hitter=False)
    if norm is None:
        return None
    out = {k: float(norm.get(k, 0.0)) for k in ROSTER_PITCHER_MANUAL_COUNT_STATS}
    ip = float(out.get("IP", 0.0))
    h = float(out.get("H", 0.0))
    bb = float(out.get("BB", 0.0))
    er = float(out.get("ER", 0.0))
    out["WHIP"] = ((h + bb) / ip) if ip > 0.0 else float("nan")
    out["ERA"] = ((9.0 * er) / ip) if ip > 0.0 else float("nan")
    return out


def _roster_norm_percentile(value: object, *, default: str = "p50") -> str:
    fallback = str(default or "p50").strip().lower()
    if fallback not in set(ROSTER_PERCENTILE_OPTIONS):
        fallback = "p50"
    pct = str(value or "").strip().lower()
    if pct not in set(ROSTER_PERCENTILE_OPTIONS):
        return fallback
    return pct


def _roster_default_state() -> dict:
    default_pct = "p50"
    return {
        "version": int(ROSTER_MANAGER_SCHEMA_VERSION),
        "default_percentile": default_pct,
        "percentile": default_pct,
        "starters": {slot: None for slot in ROSTER_ALL_STARTER_SLOTS},
        "starter_percentiles": {slot: default_pct for slot in ROSTER_ALL_STARTER_SLOTS},
        "starter_custom_volume": {slot: None for slot in ROSTER_ALL_STARTER_SLOTS},
        "starter_manual_stats": {slot: None for slot in ROSTER_ALL_STARTER_SLOTS},
        "hitter_reserves": [
            {"row_id": f"hres_{i+1}", "player_id": None, "percentile": default_pct, "custom_volume": None, "manual_stats": None}
            for i in range(ROSTER_RESERVE_SLOTS)
        ],
        "pitcher_reserves": [
            {"row_id": f"pres_{i+1}", "player_id": None, "percentile": default_pct, "custom_volume": None, "manual_stats": None}
            for i in range(ROSTER_RESERVE_SLOTS)
        ],
        "custom_hitters": [],
        "custom_pitchers": [],
        "next_custom_player_id": -1,
        "next_row_id": 1,
    }


def _roster_default_saves_payload() -> dict:
    return {
        "version": int(ROSTER_MANAGER_SCHEMA_VERSION),
        "updated_at": "",
        "teams": {},
    }


def _roster_normalize_saves_payload(raw: object) -> dict:
    default_payload = _roster_default_saves_payload()
    if not isinstance(raw, dict):
        return default_payload
    teams = raw.get("teams", {})
    if not isinstance(teams, dict):
        teams = {}
    clean_teams: dict[str, dict] = {}
    for key, val in teams.items():
        nm = str(key or "").strip()
        if not nm or (not isinstance(val, dict)):
            continue
        clean_teams[nm] = val
    return {
        "version": int(ROSTER_MANAGER_SCHEMA_VERSION),
        "updated_at": str(raw.get("updated_at", "") or ""),
        "teams": clean_teams,
    }


def _load_roster_manager_saves(path: Path) -> dict:
    default_payload = _roster_default_saves_payload()
    if not path.exists():
        return default_payload
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_payload
    return _roster_normalize_saves_payload(raw)


def _write_roster_manager_saves(path: Path, payload: dict) -> bool:
    try:
        payload = _roster_normalize_saves_payload(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return True
    except Exception:
        return False


def _roster_hitter_tokens_from_row(row: pd.Series) -> tuple[str, ...]:
    out: set[str] = set()
    mapping = [
        ("C", "is_C", "C"),
        ("1B", "is_X1B", "X1B"),
        ("2B", "is_X2B", "X2B"),
        ("3B", "is_X3B", "X3B"),
        ("SS", "is_SS", "SS"),
        ("OF", "is_OF", "OF"),
    ]
    for token, flag_col, count_col in mapping:
        flag_val = pd.to_numeric(pd.Series([row.get(flag_col)]), errors="coerce").iloc[0]
        count_val = pd.to_numeric(pd.Series([row.get(count_col)]), errors="coerce").iloc[0]
        if (pd.notna(flag_val) and float(flag_val) >= 1.0) or (
            pd.notna(count_val) and float(count_val) >= float(POSITION_COUNT_THRESHOLD)
        ):
            out.add(token)
    for pos in _position_tokens_from_text(row.get("position")):
        if pos in {"C", "1B", "2B", "3B", "SS", "OF"}:
            out.add(pos)
    out.add("UT")
    return tuple(sorted(out))


def _roster_pitcher_tokens_from_row(row: pd.Series) -> tuple[str, ...]:
    role_txt = str(row.get("opening_day_status_40man") or "").strip().upper()
    raw_pos_txt = str(row.get("position") or "").strip().upper()
    gs_val = pd.to_numeric(pd.Series([row.get("GS_proj_p50")]), errors="coerce").iloc[0]

    sp = False
    rp = False
    if "ROTATION" in role_txt or "STARTER" in role_txt:
        sp = True
    if any(tok in role_txt for tok in ["BULLPEN", "CLOSER", "SETUP", "MIDDLE RELIEVER", "LONG RELIEVER", "RELIEVER"]):
        rp = True
    if "SP" in raw_pos_txt:
        sp = True
    if "RP" in raw_pos_txt:
        rp = True
    if not sp and not rp:
        if pd.notna(gs_val) and np.isfinite(float(gs_val)) and float(gs_val) >= 1.0:
            sp = True
        else:
            rp = True
    out: set[str] = set()
    if sp:
        out.add("SP")
    if rp:
        out.add("RP")
    return tuple(sorted(out))


def _build_roster_player_pool(df: pd.DataFrame, *, for_pitchers: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["mlbid", "name", "team", "label", "elig_tokens"])
    id_col = (
        "mlbid"
        if "mlbid" in df.columns
        else ("pitcher_mlbid" if for_pitchers and "pitcher_mlbid" in df.columns else "batter_mlbid")
    )
    if id_col not in df.columns:
        return pd.DataFrame(columns=["mlbid", "name", "team", "label", "elig_tokens"])
    name_candidates = (
        ["name", "player_name", "hitter_name"]
        if for_pitchers
        else ["hitter_name", "player_name", "name"]
    )
    name_col = next((c for c in name_candidates if c in df.columns), None)
    if name_col is None:
        return pd.DataFrame(columns=["mlbid", "name", "team", "label", "elig_tokens"])

    work = df.copy()
    work["mlbid"] = pd.to_numeric(work[id_col], errors="coerce").astype("Int64")
    work = work[work["mlbid"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["mlbid", "name", "team", "label", "elig_tokens"])
    work["mlbid"] = work["mlbid"].astype("int64")
    work = work.sort_values("mlbid").drop_duplicates(subset=["mlbid"], keep="first").copy()
    work["name"] = work[name_col].fillna("").astype(str).str.strip()

    team_col = _team_col(work)
    if team_col is None and "team_40man" in work.columns:
        team_col = "team_40man"
    if team_col is None:
        work["team"] = ""
    else:
        work["team"] = (
            work[team_col].fillna("").astype(str).map(lambda v: _normalize_team_token(str(v).split("|")[0]))
        )
    work["team"] = work["team"].fillna("").astype(str).str.upper().str.strip()
    work["label"] = work.apply(
        lambda row: f"{row['name'] or 'Unknown'} - {row['team'] or 'FA'} - {int(row['mlbid'])}",
        axis=1,
    )
    if for_pitchers:
        work["elig_tokens"] = work.apply(_roster_pitcher_tokens_from_row, axis=1)
    else:
        work["elig_tokens"] = work.apply(_roster_hitter_tokens_from_row, axis=1)
    return work


def _build_roster_custom_player_pool(
    custom_players: object,
    *,
    for_pitchers: bool,
) -> pd.DataFrame:
    if not isinstance(custom_players, list) or not custom_players:
        return pd.DataFrame(columns=["mlbid", "name", "team", "label", "elig_tokens"])

    rows: list[dict[str, object]] = []
    for item in custom_players:
        normalized = _roster_normalize_custom_player_entry(item, hitter=(not for_pitchers))
        if normalized is None:
            continue
        player_id = int(normalized["player_id"])
        name = str(normalized.get("name") or "").strip() or (
            f"Placeholder {'Pitcher' if for_pitchers else 'Hitter'} {abs(player_id)}"
        )
        team = str(normalized.get("team") or "FA").upper().strip() or "FA"

        if for_pitchers:
            roles = tuple(
                tok
                for tok in normalized.get("roles", [])
                if tok in set(ROSTER_PLACEHOLDER_PITCHER_ROLES)
            )
            if not roles:
                roles = ("SP",)
            manual_stats = _roster_normalize_placeholder_manual_stats(
                normalized.get("manual_stats"),
                hitter=False,
            )
            ip = float(manual_stats.get("IP", 0.0))
            h = float(manual_stats.get("H", 0.0))
            bb = float(manual_stats.get("BB", 0.0))
            er = float(manual_stats.get("ER", 0.0))
            whip = ((h + bb) / ip) if ip > 0.0 else float("nan")
            era = ((9.0 * er) / ip) if ip > 0.0 else float("nan")
            row: dict[str, object] = {
                "mlbid": player_id,
                "name": name,
                "team": team,
                "label": f"{name} - {team} - {'/'.join(roles)} [manual]",
                "elig_tokens": tuple(sorted(set(roles))),
                "position": "/".join(roles),
                "is_placeholder": True,
            }
            for pct in ROSTER_PERCENTILE_OPTIONS:
                row[f"W_proj_{pct}"] = float(manual_stats.get("W", 0.0))
                row[f"SO_proj_{pct}"] = float(manual_stats.get("K", 0.0))
                row[f"SV_proj_{pct}"] = float(manual_stats.get("SV", 0.0))
                row[f"WHIP_proj_{pct}"] = whip
                row[f"ERA_proj_{pct}"] = era
                row[f"IP_proj_{pct}"] = ip
                row[f"H_proj_{pct}"] = h
                row[f"BB_proj_{pct}"] = bb
            rows.append(row)
            continue

        positions = tuple(
            tok
            for tok in normalized.get("positions", [])
            if tok in set(ROSTER_PLACEHOLDER_HITTER_POSITIONS)
        )
        manual_stats = _roster_normalize_placeholder_manual_stats(
            normalized.get("manual_stats"),
            hitter=True,
        )
        ab = float(manual_stats.get("AB", 0.0))
        h = float(manual_stats.get("H", 0.0))
        avg = (h / ab) if ab > 0.0 else float("nan")
        pa = _roster_parse_nonneg_float(normalized.get("plate_appearances"))
        if pa is None:
            pa = ab
        elig_tokens = tuple(sorted({*positions, "UT"}))
        display_positions = positions or ("UT",)
        row = {
            "mlbid": player_id,
            "name": name,
            "team": team,
            "label": f"{name} - {team} - {'/'.join(display_positions)} [manual]",
            "elig_tokens": elig_tokens,
            "position": "/".join(display_positions),
            "is_placeholder": True,
        }
        for pct in ROSTER_PERCENTILE_OPTIONS:
            row[f"PA_proj_{pct}"] = float(pa)
            row[f"HR_proj_{pct}"] = float(manual_stats.get("HR", 0.0))
            row[f"Runs_proj_{pct}"] = float(manual_stats.get("R", 0.0))
            row[f"RBI_proj_{pct}"] = float(manual_stats.get("RBI", 0.0))
            row[f"SB_proj_{pct}"] = float(manual_stats.get("SB", 0.0))
            row[f"AVG_proj_{pct}"] = avg
            row[f"AB_proj_{pct}"] = ab
            row[f"H_proj_{pct}"] = h
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["mlbid", "name", "team", "label", "elig_tokens"])
    return pd.DataFrame(rows)


def _combine_roster_player_pools(
    base_pool: pd.DataFrame,
    custom_players: object,
    *,
    for_pitchers: bool,
) -> pd.DataFrame:
    custom_pool = _build_roster_custom_player_pool(custom_players, for_pitchers=for_pitchers)
    if base_pool.empty:
        out = custom_pool.copy()
    elif custom_pool.empty:
        out = base_pool.copy()
    else:
        out = pd.concat([base_pool, custom_pool], ignore_index=True, sort=False)
    if out.empty:
        return out
    out["mlbid"] = pd.to_numeric(out.get("mlbid"), errors="coerce").astype("Int64")
    out = out[out["mlbid"].notna()].copy()
    out["mlbid"] = out["mlbid"].astype("int64")
    out = out.sort_values(["mlbid", "label"]).drop_duplicates(subset=["mlbid"], keep="last").copy()
    if "label" in out.columns:
        out = out.sort_values("label").reset_index(drop=True)
    return out


def _roster_hitter_slot_is_eligible(slot: str, tokens: tuple[str, ...]) -> bool:
    tok_set = set(tokens)
    if slot == "UT":
        return True
    if slot in {"C1", "C2"}:
        return "C" in tok_set
    if slot == "MI":
        return bool({"2B", "SS"}.intersection(tok_set))
    if slot == "CI":
        return bool({"1B", "3B"}.intersection(tok_set))
    if slot.startswith("OF"):
        return "OF" in tok_set
    return slot in tok_set


def _roster_pitcher_slot_is_eligible(slot: str, tokens: tuple[str, ...]) -> bool:
    tok_set = set(tokens)
    if slot == "SP7/RP3":
        return bool({"SP", "RP"}.intersection(tok_set))
    if slot.startswith("SP"):
        return "SP" in tok_set
    if slot.startswith("RP"):
        return "RP" in tok_set
    return False


def _roster_selected_player_ids(state: dict) -> set[int]:
    out: set[int] = set()
    starters = state.get("starters", {})
    if isinstance(starters, dict):
        for v in starters.values():
            parsed = _roster_parse_int(v)
            if parsed is not None:
                out.add(parsed)
    for reserve_key in ("hitter_reserves", "pitcher_reserves"):
        reserves = state.get(reserve_key, [])
        if not isinstance(reserves, list):
            continue
        for item in reserves:
            if isinstance(item, dict):
                parsed = _roster_parse_int(item.get("player_id"))
            else:
                parsed = _roster_parse_int(item)
            if parsed is not None:
                out.add(parsed)
    return out


def _roster_add_reserve_row(
    state: dict,
    *,
    hitter: bool,
    percentile: str | None = None,
    custom_volume: float | None = None,
    manual_stats: dict[str, float] | None = None,
) -> None:
    next_row_id = _roster_parse_int(state.get("next_row_id"))
    if next_row_id is None or next_row_id < 1:
        next_row_id = 1
    prefix = "hres" if hitter else "pres"
    key = "hitter_reserves" if hitter else "pitcher_reserves"
    if key not in state or (not isinstance(state.get(key), list)):
        state[key] = []
    pct = _roster_norm_percentile(
        percentile,
        default=_roster_norm_percentile(state.get("default_percentile", "p50")),
    )
    state[key].append(
        {
            "row_id": f"{prefix}_{next_row_id}",
            "player_id": None,
            "percentile": pct,
            "custom_volume": _roster_parse_nonneg_float(custom_volume),
            "manual_stats": _roster_normalize_manual_stats(manual_stats, hitter=hitter),
        }
    )
    state["next_row_id"] = int(next_row_id + 1)


def _serialize_roster_state(state: dict) -> dict:
    custom_hitters, custom_pitchers, next_custom_player_id = _roster_normalize_custom_player_lists(
        state.get("custom_hitters"),
        state.get("custom_pitchers"),
        next_player_id_raw=state.get("next_custom_player_id"),
    )
    out_starters: dict[str, int | None] = {}
    out_starter_pct: dict[str, str] = {}
    out_starter_custom_volume: dict[str, float | None] = {}
    out_starter_manual_stats: dict[str, dict[str, float] | None] = {}
    starters = state.get("starters", {})
    starter_percentiles = state.get("starter_percentiles", {})
    starter_custom_volume = state.get("starter_custom_volume", {})
    starter_manual_stats = state.get("starter_manual_stats", {})
    default_pct = _roster_norm_percentile(
        state.get("default_percentile", state.get("percentile", "p50"))
    )
    for slot in ROSTER_ALL_STARTER_SLOTS:
        parsed = _roster_parse_int(starters.get(slot) if isinstance(starters, dict) else None)
        out_starters[slot] = parsed
        src_pct = starter_percentiles.get(slot) if isinstance(starter_percentiles, dict) else None
        out_starter_pct[slot] = _roster_norm_percentile(src_pct, default=default_pct)
        src_custom = starter_custom_volume.get(slot) if isinstance(starter_custom_volume, dict) else None
        out_starter_custom_volume[slot] = _roster_parse_nonneg_float(src_custom)
        src_manual = starter_manual_stats.get(slot) if isinstance(starter_manual_stats, dict) else None
        out_starter_manual_stats[slot] = _roster_normalize_manual_stats(
            src_manual,
            hitter=(slot in ROSTER_HITTER_STARTER_SLOTS),
        )

    def _serialize_reserves(key: str) -> list[dict[str, object]]:
        rows = state.get(key, [])
        if not isinstance(rows, list):
            return []
        out_rows: list[dict[str, object]] = []
        for idx, item in enumerate(rows):
            item_dict = item if isinstance(item, dict) else {"player_id": item}
            parsed = _roster_parse_int(item_dict.get("player_id"))
            row_id = str(item_dict.get("row_id") or f"{key}_{idx+1}").strip()
            if not row_id:
                row_id = f"{key}_{idx+1}"
            row_pct = _roster_norm_percentile(item_dict.get("percentile"), default=default_pct)
            row_custom = _roster_parse_nonneg_float(item_dict.get("custom_volume"))
            row_manual = _roster_normalize_manual_stats(
                item_dict.get("manual_stats"),
                hitter=(key == "hitter_reserves"),
            )
            out_rows.append(
                {
                    "row_id": row_id,
                    "player_id": parsed,
                    "percentile": row_pct,
                    "custom_volume": row_custom,
                    "manual_stats": row_manual,
                }
            )
        return out_rows

    return {
        "version": int(ROSTER_MANAGER_SCHEMA_VERSION),
        "saved_at": _roster_now_iso(),
        "default_percentile": default_pct,
        "percentile": default_pct,
        "starters": out_starters,
        "starter_percentiles": out_starter_pct,
        "starter_custom_volume": out_starter_custom_volume,
        "starter_manual_stats": out_starter_manual_stats,
        "hitter_reserves": _serialize_reserves("hitter_reserves"),
        "pitcher_reserves": _serialize_reserves("pitcher_reserves"),
        "custom_hitters": custom_hitters,
        "custom_pitchers": custom_pitchers,
        "next_custom_player_id": next_custom_player_id,
    }


def _deserialize_roster_state(raw_state: object, hitter_pool: pd.DataFrame, pitcher_pool: pd.DataFrame) -> tuple[dict, int]:
    state = _roster_default_state()
    if not isinstance(raw_state, dict):
        return state, 0

    custom_hitters, custom_pitchers, next_custom_player_id = _roster_normalize_custom_player_lists(
        raw_state.get("custom_hitters"),
        raw_state.get("custom_pitchers"),
        next_player_id_raw=raw_state.get("next_custom_player_id"),
    )
    state["custom_hitters"] = custom_hitters
    state["custom_pitchers"] = custom_pitchers
    state["next_custom_player_id"] = int(next_custom_player_id)

    hitter_pool_full = _combine_roster_player_pools(
        hitter_pool,
        custom_hitters,
        for_pitchers=False,
    )
    pitcher_pool_full = _combine_roster_player_pools(
        pitcher_pool,
        custom_pitchers,
        for_pitchers=True,
    )

    hitter_ids = set(
        pd.to_numeric(hitter_pool_full.get("mlbid"), errors="coerce").dropna().astype("int64").tolist()
    )
    pitcher_ids = set(
        pd.to_numeric(pitcher_pool_full.get("mlbid"), errors="coerce").dropna().astype("int64").tolist()
    )
    hitter_tokens = hitter_pool_full.set_index("mlbid")["elig_tokens"].to_dict() if not hitter_pool_full.empty else {}
    pitcher_tokens = pitcher_pool_full.set_index("mlbid")["elig_tokens"].to_dict() if not pitcher_pool_full.empty else {}

    default_pct = _roster_norm_percentile(
        raw_state.get("default_percentile", raw_state.get("percentile", "p50"))
    )
    state["default_percentile"] = default_pct
    state["percentile"] = default_pct
    state["starter_percentiles"] = {slot: default_pct for slot in ROSTER_ALL_STARTER_SLOTS}
    state["starter_custom_volume"] = {slot: None for slot in ROSTER_ALL_STARTER_SLOTS}
    state["starter_manual_stats"] = {slot: None for slot in ROSTER_ALL_STARTER_SLOTS}
    starter_pct_raw = raw_state.get("starter_percentiles", {})
    if isinstance(starter_pct_raw, dict):
        for slot in ROSTER_ALL_STARTER_SLOTS:
            state["starter_percentiles"][slot] = _roster_norm_percentile(
                starter_pct_raw.get(slot),
                default=default_pct,
            )
    starter_custom_raw = raw_state.get("starter_custom_volume", {})
    if isinstance(starter_custom_raw, dict):
        for slot in ROSTER_ALL_STARTER_SLOTS:
            state["starter_custom_volume"][slot] = _roster_parse_nonneg_float(
                starter_custom_raw.get(slot)
            )
    starter_manual_raw = raw_state.get("starter_manual_stats", {})
    if isinstance(starter_manual_raw, dict):
        for slot in ROSTER_ALL_STARTER_SLOTS:
            state["starter_manual_stats"][slot] = _roster_normalize_manual_stats(
                starter_manual_raw.get(slot),
                hitter=(slot in ROSTER_HITTER_STARTER_SLOTS),
            )

    used: set[int] = set()
    dropped = 0

    starters = raw_state.get("starters", {})
    if isinstance(starters, dict):
        for slot in ROSTER_HITTER_STARTER_SLOTS:
            pid = _roster_parse_int(starters.get(slot))
            if pid is None:
                continue
            toks = tuple(hitter_tokens.get(pid, tuple()))
            if (pid in hitter_ids) and (pid not in used) and _roster_hitter_slot_is_eligible(slot, toks):
                state["starters"][slot] = pid
                used.add(pid)
            else:
                dropped += 1
                state["starter_custom_volume"][slot] = None
                state["starter_manual_stats"][slot] = None
        for slot in ROSTER_PITCHER_STARTER_SLOTS:
            pid = _roster_parse_int(starters.get(slot))
            if pid is None:
                continue
            toks = tuple(pitcher_tokens.get(pid, tuple()))
            if (pid in pitcher_ids) and (pid not in used) and _roster_pitcher_slot_is_eligible(slot, toks):
                state["starters"][slot] = pid
                used.add(pid)
            else:
                dropped += 1
                state["starter_custom_volume"][slot] = None
                state["starter_manual_stats"][slot] = None

    def _load_reserves(raw_values: object, *, hitter: bool) -> None:
        nonlocal dropped
        if not isinstance(raw_values, list):
            return
        valid_pool = hitter_ids if hitter else pitcher_ids
        key = "hitter_reserves" if hitter else "pitcher_reserves"
        for item in raw_values:
            item_dict = item if isinstance(item, dict) else {"player_id": item}
            row_pct = _roster_norm_percentile(item_dict.get("percentile"), default=default_pct)
            row_custom = _roster_parse_nonneg_float(item_dict.get("custom_volume"))
            row_manual = _roster_normalize_manual_stats(
                item_dict.get("manual_stats"),
                hitter=hitter,
            )
            pid = _roster_parse_int(item_dict.get("player_id"))
            _roster_add_reserve_row(
                state,
                hitter=hitter,
                percentile=row_pct,
                custom_volume=row_custom,
                manual_stats=row_manual,
            )
            if pid is None:
                continue
            if pid in valid_pool and pid not in used:
                state[key][-1]["player_id"] = int(pid)
                used.add(pid)
            else:
                dropped += 1

    # Clear the pre-populated default rows before loading saved data so _load_reserves
    # appends only the saved players (not stacked on top of the 15 empty defaults).
    state["hitter_reserves"] = []
    state["pitcher_reserves"] = []
    _load_reserves(raw_state.get("hitter_reserves"), hitter=True)
    _load_reserves(raw_state.get("pitcher_reserves"), hitter=False)
    # Always ensure exactly ROSTER_RESERVE_SLOTS entries in each reserve list.
    for _rkey in ("hitter_reserves", "pitcher_reserves"):
        while len(state[_rkey]) < ROSTER_RESERVE_SLOTS:
            state[_rkey].append({"row_id": None, "player_id": None, "percentile": default_pct, "custom_volume": None, "manual_stats": None})
        if len(state[_rkey]) > ROSTER_RESERVE_SLOTS:
            state[_rkey] = state[_rkey][:ROSTER_RESERVE_SLOTS]
    return state, dropped


def _roster_display_pick_label(player_id: object, labels_by_id: dict[int, str], *, empty_label: str) -> str:
    pid = _roster_parse_int(player_id)
    if pid is None:
        return empty_label
    return str(labels_by_id.get(pid, f"Unknown - FA - {pid}"))


def _roster_reset_ui_widget_state(key_prefix: str) -> None:
    # Streamlit widget keys persist across navigation/reruns; clear roster row widgets
    # when loading/clearing rosters so pickers re-hydrate from saved state.
    reset_prefixes = (
        f"{key_prefix}_starter_",
        f"{key_prefix}_hres_",
        f"{key_prefix}_pres_",
        f"{key_prefix}_custom_",
    )
    for key in list(st.session_state.keys()):
        if any(str(key).startswith(prefix) for prefix in reset_prefixes):
            st.session_state.pop(key, None)


def _roster_prepare_live_state(raw_state: object) -> dict:
    live_state = raw_state if isinstance(raw_state, dict) else _roster_default_state()
    if not isinstance(live_state.get("starters"), dict):
        live_state["starters"] = {slot: None for slot in ROSTER_ALL_STARTER_SLOTS}
    for slot in ROSTER_ALL_STARTER_SLOTS:
        if slot not in live_state["starters"]:
            live_state["starters"][slot] = None
    _empty_hres = {"row_id": None, "player_id": None, "percentile": "p50", "custom_volume": None, "manual_stats": None}
    _empty_pres = {"row_id": None, "player_id": None, "percentile": "p50", "custom_volume": None, "manual_stats": None}
    for reserve_key, empty_row in (("hitter_reserves", _empty_hres), ("pitcher_reserves", _empty_pres)):
        if not isinstance(live_state.get(reserve_key), list):
            live_state[reserve_key] = []
        # Pad to ROSTER_RESERVE_SLOTS; trim extras beyond that
        while len(live_state[reserve_key]) < ROSTER_RESERVE_SLOTS:
            live_state[reserve_key].append(dict(empty_row))
        if len(live_state[reserve_key]) > ROSTER_RESERVE_SLOTS:
            live_state[reserve_key] = live_state[reserve_key][:ROSTER_RESERVE_SLOTS]
    next_row_id = _roster_parse_int(live_state.get("next_row_id"))
    live_state["next_row_id"] = int(next_row_id if (next_row_id is not None and next_row_id >= 1) else 1)
    default_pct = _roster_norm_percentile(
        live_state.get("default_percentile", live_state.get("percentile", "p50"))
    )
    live_state["default_percentile"] = default_pct
    live_state["percentile"] = default_pct
    if not isinstance(live_state.get("starter_percentiles"), dict):
        live_state["starter_percentiles"] = {}
    if not isinstance(live_state.get("starter_custom_volume"), dict):
        live_state["starter_custom_volume"] = {}
    if not isinstance(live_state.get("starter_manual_stats"), dict):
        live_state["starter_manual_stats"] = {}
    for slot in ROSTER_ALL_STARTER_SLOTS:
        live_state["starter_percentiles"][slot] = _roster_norm_percentile(
            live_state["starter_percentiles"].get(slot),
            default=default_pct,
        )
        live_state["starter_custom_volume"][slot] = _roster_parse_nonneg_float(
            live_state["starter_custom_volume"].get(slot)
        )
        live_state["starter_manual_stats"][slot] = _roster_normalize_manual_stats(
            live_state["starter_manual_stats"].get(slot),
            hitter=(slot in ROSTER_HITTER_STARTER_SLOTS),
        )
    _roster_normalize_custom_players_in_state(live_state)
    return live_state


def _roster_validate_live_state(
    live_state: dict,
    *,
    hitter_pool_by_id: pd.DataFrame,
    pitcher_pool_by_id: pd.DataFrame,
    hitter_tokens_by_id: dict[int, tuple[str, ...]],
    pitcher_tokens_by_id: dict[int, tuple[str, ...]],
) -> int:
    dropped_from_state = 0
    used_ids: set[int] = set()

    for slot in ROSTER_HITTER_STARTER_SLOTS:
        pid = _roster_parse_int(live_state["starters"].get(slot))
        if pid is None:
            live_state["starters"][slot] = None
            live_state["starter_custom_volume"][slot] = None
            live_state["starter_manual_stats"][slot] = None
            continue
        if (
            pid in hitter_pool_by_id.index
            and pid not in used_ids
            and _roster_hitter_slot_is_eligible(slot, tuple(hitter_tokens_by_id.get(pid, tuple())))
        ):
            live_state["starters"][slot] = pid
            used_ids.add(pid)
        else:
            live_state["starters"][slot] = None
            live_state["starter_custom_volume"][slot] = None
            live_state["starter_manual_stats"][slot] = None
            dropped_from_state += 1

    for slot in ROSTER_PITCHER_STARTER_SLOTS:
        pid = _roster_parse_int(live_state["starters"].get(slot))
        if pid is None:
            live_state["starters"][slot] = None
            live_state["starter_custom_volume"][slot] = None
            live_state["starter_manual_stats"][slot] = None
            continue
        if (
            pid in pitcher_pool_by_id.index
            and pid not in used_ids
            and _roster_pitcher_slot_is_eligible(slot, tuple(pitcher_tokens_by_id.get(pid, tuple())))
        ):
            live_state["starters"][slot] = pid
            used_ids.add(pid)
        else:
            live_state["starters"][slot] = None
            live_state["starter_custom_volume"][slot] = None
            live_state["starter_manual_stats"][slot] = None
            dropped_from_state += 1

    for reserve_key, pool_by_id, hitter_flag in (
        ("hitter_reserves", hitter_pool_by_id, True),
        ("pitcher_reserves", pitcher_pool_by_id, False),
    ):
        cleaned_rows: list[dict[str, object]] = []
        for idx, row in enumerate(live_state.get(reserve_key, [])):
            row_dict = row if isinstance(row, dict) else {"player_id": row}
            row_id = str(row_dict.get("row_id") or f"{reserve_key}_{idx+1}").strip()
            if not row_id:
                row_id = f"{reserve_key}_{idx+1}"
            row_pct = _roster_norm_percentile(
                row_dict.get("percentile"),
                default=live_state.get("default_percentile", "p50"),
            )
            row_custom = _roster_parse_nonneg_float(row_dict.get("custom_volume"))
            row_manual = _roster_normalize_manual_stats(
                row_dict.get("manual_stats"),
                hitter=hitter_flag,
            )
            pid = _roster_parse_int(row_dict.get("player_id"))
            if pid is None:
                cleaned_rows.append(
                    {
                        "row_id": row_id,
                        "player_id": None,
                        "percentile": row_pct,
                        "custom_volume": row_custom,
                        "manual_stats": None,
                    }
                )
                continue
            if pid in pool_by_id.index and pid not in used_ids:
                cleaned_rows.append(
                    {
                        "row_id": row_id,
                        "player_id": pid,
                        "percentile": row_pct,
                        "custom_volume": row_custom,
                        "manual_stats": row_manual,
                    }
                )
                used_ids.add(pid)
            else:
                cleaned_rows.append(
                    {
                        "row_id": row_id,
                        "player_id": None,
                        "percentile": row_pct,
                        "custom_volume": row_custom,
                        "manual_stats": None,
                    }
                )
                dropped_from_state += 1
        live_state[reserve_key] = cleaned_rows

    return int(dropped_from_state)


def _roster_numeric_from_row(row: pd.Series, col: str) -> float:
    val = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
    if pd.isna(val) or (not np.isfinite(float(val))):
        return float("nan")
    return float(val)


def _roster_apply_int_display(
    df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").round(0).astype("Int64")
    return out


def _roster_style_needed_cells(
    df: pd.DataFrame,
    *,
    fail_cols: set[str],
) -> pd.DataFrame | object:
    if df.empty or (not fail_cols) or ("Row" not in df.columns):
        return df
    needed_mask = df["Row"].astype(str).eq("Needed")
    if not bool(needed_mask.any()):
        return df
    needed_idx = list(df.index[needed_mask])

    def _style_block(frame: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for col in fail_cols:
            if col not in styles.columns:
                continue
            for idx in needed_idx:
                styles.at[idx, col] = (
                    "background-color: rgba(220,53,69,0.25); "
                    "color: #7f1d1d; font-weight: 600;"
                )
        return styles

    styler = df.style.apply(_style_block, axis=None)
    fmt = _float_format_map(df)
    if fmt:
        styler = styler.format(fmt)
    return styler


def _source_levels_col(df: pd.DataFrame) -> str | None:
    cols = [c for c in df.columns if str(c).startswith("levels_played_")]
    if not cols:
        return None
    # Prefer highest suffix year if present.
    def _year_key(col: str) -> int:
        tail = str(col).replace("levels_played_", "")
        yr = pd.to_numeric(tail, errors="coerce")
        return int(yr) if pd.notna(yr) else -1

    cols = sorted(cols, key=_year_key, reverse=True)
    return cols[0]


def _split_level_tokens(v: object) -> list[str]:
    raw = str(v or "").strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split("|") if t.strip()]


def _apply_levels_played_filter(
    df: pd.DataFrame,
    *,
    label: str,
    key: str,
) -> pd.DataFrame:
    level_col = _source_levels_col(df)
    if not level_col:
        return df
    token_set = sorted({tok for v in df[level_col].dropna().tolist() for tok in _split_level_tokens(v)})
    token_set = [t for t in token_set if t in BP_SOURCE_LEVEL_LABELS]
    if not token_set:
        return df
    opts = ["All"] + [BP_SOURCE_LEVEL_LABELS[t] for t in token_set]
    choice = st.selectbox(label, opts, index=0, key=key)
    if choice == "All":
        return df
    token = next((k for k, v in BP_SOURCE_LEVEL_LABELS.items() if v == choice), None)
    if token is None:
        return df
    mask = df[level_col].map(lambda v: token in _split_level_tokens(v))
    return df[mask]


def _apply_team_filter(
    df: pd.DataFrame,
    *,
    label: str,
    key: str,
) -> pd.DataFrame:
    team_col = _team_col(df)
    if not team_col:
        return df
    teams = sorted({tok for v in df[team_col].dropna().tolist() for tok in _team_tokens(v)})
    team = st.selectbox(label, ["All"] + teams, index=0, key=key)
    if team == "All":
        return df
    mask = df[team_col].map(lambda v: team in _team_tokens(v))
    return df[mask]


def _apply_opening_day_status_filter(
    df: pd.DataFrame,
    *,
    label: str,
    key: str,
) -> pd.DataFrame:
    status_col = "opening_day_status_40man"
    if status_col not in df.columns:
        return df
    vals = (
        df[status_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    statuses = sorted([s for s in vals.unique().tolist() if s])
    if not statuses:
        return df
    selected = st.multiselect(
        label,
        ["All"] + statuses,
        default=["All"],
        key=key,
    )
    selected_clean = [s for s in selected if s != "All"]
    if not selected_clean:
        return df
    return df[vals.isin(set(selected_clean))]


def _prefer_marcel_volume_bases(metric_bases: list[str]) -> list[str]:
    base_set = set(metric_bases)
    out = list(metric_bases)
    replace_map = {
        "G": "G_marcel",
        "GS": "GS_marcel",
        "IP": "IP_marcel",
        "TBF": "TBF_marcel",
    }
    for base, marcel in replace_map.items():
        if marcel in base_set and base in out:
            out = [marcel if m == base else m for m in out]
    return out


def _position_options(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["All"]
    opts = [
        pos
        for pos in POSITION_FILTER_COLS
        if pos in df.columns or f"is_{pos}" in df.columns
    ]
    return ["All"] + opts if opts else ["All"]


def _filter_by_positions(
    df: pd.DataFrame,
    positions: list[str] | tuple[str, ...] | str | None,
    *,
    min_count: int = POSITION_COUNT_THRESHOLD,
) -> pd.DataFrame:
    if df.empty or positions is None:
        return df
    if isinstance(positions, (str, bytes)):
        positions = [positions]
    selected = [str(p) for p in positions if str(p) != "All"]
    if not selected:
        return df
    mask = pd.Series(False, index=df.index)
    for pos in selected:
        flag_col = f"is_{pos}"
        if flag_col in df.columns:
            values = pd.to_numeric(df[flag_col], errors="coerce").fillna(0.0)
            mask |= values >= 1.0
            continue
        if pos not in df.columns:
            continue
        values = pd.to_numeric(df[pos], errors="coerce").fillna(0.0)
        threshold = 1 if float(values.max(skipna=True)) <= 1.0 else float(min_count)
        mask |= values >= threshold
    if not bool(mask.any()):
        return df.iloc[0:0]
    return df[mask]


def _apply_column_filters(
    df: pd.DataFrame,
    *,
    key_prefix: str,
    expander_label: str = "Column filters",
) -> pd.DataFrame:
    if df.empty:
        return df
    with st.expander(expander_label, expanded=False):
        filtered = df
        for col in df.columns:
            if str(col).startswith("__"):
                continue
            col_key = f"{key_prefix}_{col}"
            if pd.api.types.is_numeric_dtype(df[col]):
                op = st.selectbox(
                    f"{col} filter",
                    options=["(no filter)", "=", "<", "<=", ">", ">=", "between"],
                    key=f"{col_key}_op",
                )
                if op == "(no filter)":
                    continue
                if op == "between":
                    low = st.number_input(f"{col} min", key=f"{col_key}_min", value=0.0)
                    high = st.number_input(f"{col} max", key=f"{col_key}_max", value=0.0)
                    filtered = filtered[(filtered[col] >= low) & (filtered[col] <= high)]
                else:
                    value = st.number_input(f"{col} value", key=f"{col_key}_val", value=0.0)
                    if op == "=":
                        filtered = filtered[filtered[col] == value]
                    elif op == "<":
                        filtered = filtered[filtered[col] < value]
                    elif op == "<=":
                        filtered = filtered[filtered[col] <= value]
                    elif op == ">":
                        filtered = filtered[filtered[col] > value]
                    elif op == ">=":
                        filtered = filtered[filtered[col] >= value]
            else:
                op = st.selectbox(
                    f"{col} filter",
                    options=["(no filter)", "=", "contains"],
                    key=f"{col_key}_op",
                )
                if op == "(no filter)":
                    continue
                value = st.text_input(f"{col} value", key=f"{col_key}_val", value="")
                if value:
                    if op == "=":
                        filtered = filtered[filtered[col] == value]
                    else:
                        filtered = filtered[
                            filtered[col]
                            .astype(str)
                            .str.contains(value, case=False, na=False)
                        ]
        return filtered


def _rounding_metric_base(col: str) -> str:
    base = str(col).strip().lower()
    for suffix in ("_after", "_before"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    for prefix in (
        "delta_",
        "actual_",
        "pred_",
        "model_",
        "kpi_",
        "two_stage_resid_pred_",
        "two_stage_resid_applied_",
        "model_minus_kpi_",
        "delta_model_minus_kpi_",
    ):
        if base.startswith(prefix):
            base = base[len(prefix) :]
    base = _metric_base_from_proj_col(base)
    if base in DISPLAY_ROUNDING_BASE_OVERRIDES:
        return DISPLAY_ROUNDING_BASE_OVERRIDES[base]
    base_compact = base.replace("%", "").replace("(ft.)", "").replace("()", "").strip()
    if base_compact in DISPLAY_ROUNDING_BASE_OVERRIDES:
        return DISPLAY_ROUNDING_BASE_OVERRIDES[base_compact]
    return base


def _round_decimals_for_col(col: str) -> int:
    base = _rounding_metric_base(col)
    if base in {"age_used", "age"}:
        return 0
    if base in PITCHER_KPI_ZERO_DEC_BASES:
        return 0
    if base in PITCHER_KPI_ONE_DEC_BASES:
        return 1
    # Pitching display precision targets.
    if base in {"bb%", "gb%"}:
        return 1
    if "babip" in base:
        return 3
    if base in COUNTING_ONE_DEC_BASES:
        return 0
    if base in TWO_DEC_BASES:
        return 2
    if base in ONE_DEC_PERCENT_BASES:
        return 1
    if any(tok in base for tok in THREE_DEC_SLASH_TOKENS):
        return 3
    if any(tok in base for tok in KPI_ONE_DEC_TOKENS):
        return 1
    return 3


def _apply_custom_rounding(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return out
    by_decimals: dict[int, list[str]] = {}
    for col in numeric_cols:
        by_decimals.setdefault(_round_decimals_for_col(col), []).append(col)
    for decimals, cols in by_decimals.items():
        out[cols] = out[cols].apply(pd.to_numeric, errors="coerce").round(decimals)
    # Keep counting stats as nullable integers so plain dataframe rendering
    # does not show trailing ".0".
    for col in numeric_cols:
        if _round_decimals_for_col(col) == 0:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(0).astype("Int64")
    return out


def _float_format_map(df: pd.DataFrame) -> dict[str, str]:
    float_cols = df.select_dtypes(include=["float32", "float64", "Float64"]).columns
    return {col: f"{{:.{_round_decimals_for_col(col)}f}}" for col in float_cols}


def _looks_like_pitching_metric_set(metric_bases: list[str]) -> bool:
    base_set = {str(m).strip().lower() for m in metric_bases}
    signature = {"era", "whip", "ip", "tbf", "gs", "sv", "hr/9", "hr/bbe%", "gb%", "whiff%"}
    return sum(1 for m in signature if m in base_set) >= 3


def _looks_like_pitcher_kpi_metric_set(metric_bases: list[str]) -> bool:
    base_set = {str(m).strip().lower() for m in metric_bases}
    signature = {
        "stuff_raw",
        "stuff_raw_reg",
        "fastball_velo_reg",
        "max_velo_reg",
        "fastball_vaa_reg",
        "fa_pct_reg",
        "bb_rpm_reg",
        "swstr_reg",
        "ball_pct_reg",
        "z_contact_reg",
        "chase_reg",
        "csw_reg",
        "la_lte_0_reg",
    }
    return sum(1 for m in signature if m in base_set) >= 5


def _reverse_metric_bases(metric_bases: list[str]) -> set[str]:
    out: set[str] = set()
    is_pitching = _looks_like_pitching_metric_set(metric_bases)
    for metric in metric_bases:
        text = str(metric).strip().lower()
        if is_pitching and text == "bb%":
            out.add(metric)
            continue
        if is_pitching and text in {"k%", "whiff%", "swstr%", "so"}:
            continue
        if is_pitching and ("babip" in text):
            out.add(metric)
            continue
        if any(tok in text for tok in HIGHER_IS_WORSE_TOKENS):
            out.add(metric)
    return out


def _render_projection_styled_table(
    display_df: pd.DataFrame,
    *,
    stats_source: pd.DataFrame,
    metric_bases: list[str],
    enable_conditional_formatting: bool = True,
) -> None:
    if display_df.empty:
        st.dataframe(display_df, width="stretch", hide_index=True)
        return

    out = _apply_custom_rounding(display_df)

    max_elements = pd.get_option("styler.render.max_elements")
    total_cells = out.shape[0] * out.shape[1]
    if (
        not enable_conditional_formatting
        or total_cells > max_elements
        or total_cells > MAX_STYLED_CELLS
        or out.shape[0] > MAX_STYLED_ROWS
    ):
        st.dataframe(out, width="stretch", hide_index=True)
        return

    src = stats_source.copy()
    if "appeared_in_MLB" in src.columns:
        mlb_flags = (
            src["appeared_in_MLB"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        src = src[mlb_flags.eq("T")].copy()
    format_cols = [
        c
        for c in out.select_dtypes(include="number").columns
        if c in src.columns and c not in DEFAULT_NO_FORMAT_COLS
    ]
    if not format_cols:
        st.dataframe(out, width="stretch", hide_index=True)
        return

    group_cols = [c for c in ["target_season", "source_level", "level_id_source"] if c in src.columns]
    reverse_bases = _reverse_metric_bases(metric_bases)
    cmap = colors.LinearSegmentedColormap.from_list(
        "rwgn", ["#c75c5c", "#f7f7f7", "#5cb85c"]
    )
    cmap_rev = colors.LinearSegmentedColormap.from_list(
        "gnrw", ["#5cb85c", "#f7f7f7", "#c75c5c"]
    )
    alpha = 0.9

    if group_cols:
        q05 = src.groupby(group_cols, dropna=False)[format_cols].quantile(0.05)
        q95 = src.groupby(group_cols, dropna=False)[format_cols].quantile(0.95)
        med = src.groupby(group_cols, dropna=False)[format_cols].median()
    else:
        q05 = src[format_cols].quantile(0.05)
        q95 = src[format_cols].quantile(0.95)
        med = src[format_cols].median()

    def style_row(row: pd.Series) -> list[str]:
        if group_cols:
            group_vals = out.loc[row.name, group_cols]
            if isinstance(group_vals, pd.Series):
                group_key = tuple(group_vals.values.tolist())
            else:
                group_key = group_vals
            if group_key not in q05.index:
                return [""] * len(row)
            row_q05 = q05.loc[group_key]
            row_q95 = q95.loc[group_key]
            row_med = med.loc[group_key]
        else:
            row_q05 = q05
            row_q95 = q95
            row_med = med

        styles: list[str] = []
        for col in row.index:
            if col not in format_cols:
                styles.append("")
                continue
            vmin = row_q05[col]
            vmax = row_q95[col]
            vcenter = row_med[col]
            if pd.isna(vmin) or pd.isna(vmax) or pd.isna(vcenter) or vmin == vmax:
                styles.append("")
                continue
            if not (vmin < vcenter < vmax):
                vcenter = (vmin + vmax) / 2.0
            if not (vmin < vcenter < vmax):
                styles.append("")
                continue

            val = row[col]
            if pd.isna(val):
                styles.append("")
                continue
            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            base = _metric_base_from_proj_col(col)
            col_cmap = cmap_rev if base in reverse_bases else cmap
            rgb = colors.to_rgb(col_cmap(norm(float(np.clip(val, vmin, vmax)))))
            styles.append(
                "background-color: "
                f"rgba({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)},{alpha}); color: #000000"
            )
        return styles

    try:
        styler = out.style.apply(style_row, axis=1)
        format_map = _float_format_map(out)
        if format_map:
            styler = styler.format(format_map)
        st.dataframe(styler, width="stretch", hide_index=True)
    except Exception as exc:  # pragma: no cover - UI fallback
        st.warning(f"Styling fallback applied due to render issue: {exc}")
        st.dataframe(out, width="stretch", hide_index=True)


def _apply_binary_series_operation(
    left: pd.Series,
    right: pd.Series,
    *,
    op: str,
) -> pd.Series:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        denom = right.where(right.abs() > 1e-12, np.nan)
        return left / denom
    return pd.Series(np.nan, index=left.index, dtype="float64")


def _apply_custom_arithmetic_columns(
    display_df: pd.DataFrame,
    stats_source: pd.DataFrame,
    *,
    key_prefix: str,
    table_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_display = display_df.copy()
    out_stats = stats_source.copy()
    defs_key = f"{key_prefix}_defs"
    if not isinstance(st.session_state.get(defs_key), list):
        st.session_state[defs_key] = []
    calc_defs = st.session_state.get(defs_key, [])

    numeric_candidates: list[str] = []
    for col in out_stats.columns:
        vals = pd.to_numeric(out_stats[col], errors="coerce")
        if vals.notna().any():
            numeric_candidates.append(col)

    with st.expander(f"{table_label} calculated columns", expanded=False):
        st.caption("Create a new column from two numeric columns using +, -, *, or /.")
        if len(numeric_candidates) < 2:
            st.info("Need at least two numeric columns to create a calculated column.")
        else:
            op_options = ["+", "-", "*", "/"]
            add_cols = st.columns([2.2, 2.2, 1.0, 2.2, 1.2])
            with add_cols[0]:
                new_name = st.text_input(
                    "New column name",
                    value="",
                    key=f"{key_prefix}_new_name",
                ).strip()
            with add_cols[1]:
                left_col = st.selectbox(
                    "Column A",
                    numeric_candidates,
                    key=f"{key_prefix}_left",
                )
            with add_cols[2]:
                op = st.selectbox(
                    "Op",
                    op_options,
                    key=f"{key_prefix}_op",
                )
            with add_cols[3]:
                right_col = st.selectbox(
                    "Column B",
                    numeric_candidates,
                    index=1 if len(numeric_candidates) > 1 else 0,
                    key=f"{key_prefix}_right",
                )
            with add_cols[4]:
                add_clicked = st.button("Add", key=f"{key_prefix}_add")
            if add_clicked:
                if not new_name:
                    st.warning("Enter a column name.")
                elif new_name in out_display.columns:
                    st.warning(f"Column `{new_name}` already exists.")
                elif any(str(d.get("name", "")) == new_name for d in calc_defs):
                    st.warning(f"Calculated column `{new_name}` already exists.")
                else:
                    calc_defs.append(
                        {
                            "name": str(new_name),
                            "left": str(left_col),
                            "op": str(op),
                            "right": str(right_col),
                        }
                    )
                    st.session_state[defs_key] = calc_defs
                    st.rerun()

        if calc_defs:
            labels = [
                f"{d.get('name', 'calc')} = {d.get('left', '?')} {d.get('op', '?')} {d.get('right', '?')}"
                for d in calc_defs
            ]
            rm_cols = st.columns([4.2, 1.3, 1.5])
            with rm_cols[0]:
                rm_label = st.selectbox(
                    "Existing calculated columns",
                    labels,
                    key=f"{key_prefix}_rm_label",
                )
            rm_idx = labels.index(rm_label) if rm_label in labels else -1
            with rm_cols[1]:
                rm_clicked = st.button("Remove", key=f"{key_prefix}_rm_one")
            with rm_cols[2]:
                clear_clicked = st.button("Clear all", key=f"{key_prefix}_rm_all")
            if rm_clicked and rm_idx >= 0:
                calc_defs.pop(rm_idx)
                st.session_state[defs_key] = calc_defs
                st.rerun()
            if clear_clicked:
                st.session_state[defs_key] = []
                st.rerun()

    for d in calc_defs:
        name = str(d.get("name", "")).strip()
        left_col = str(d.get("left", "")).strip()
        right_col = str(d.get("right", "")).strip()
        op = str(d.get("op", "")).strip()
        if not name or left_col not in out_stats.columns or right_col not in out_stats.columns:
            continue
        left_vals = pd.to_numeric(out_stats[left_col], errors="coerce")
        right_vals = pd.to_numeric(out_stats[right_col], errors="coerce")
        result = _apply_binary_series_operation(left_vals, right_vals, op=op)
        out_stats[name] = result
        out_display[name] = result
    return out_display, out_stats


def show_table(
    df: pd.DataFrame,
    title: str,
    name_col: str,
    id_col: str,
    *,
    key_prefix: str = "",
    hide_display_cols: set[str] | None = None,
    prefer_kpi_defaults: bool = True,
    compact_ui: bool = False,
) -> None:
    st.subheader(title)
    if df.empty:
        st.info(f"Missing {title} file.")
        return
    df = _add_derived_slashline_metrics(df)
    df = _add_derived_pitcher_kpi_metrics(df)
    df = _add_derived_points_metrics(df)

    key_root = f"{title}_{key_prefix}".strip("_")
    controls = st.expander(f"{title} controls", expanded=not compact_ui)
    with controls:
        with st.form(f"{key_root}_controls_form"):
            season_opts = sorted(df["target_season"].dropna().unique().tolist()) if "target_season" in df.columns else []
            season = st.selectbox(f"{title} season", ["All"] + season_opts, index=0, key=f"{key_root}_season")
            if season != "All":
                df = df[df["target_season"] == season]
    
            if "level_id_source" in df.columns:
                level_id = pd.to_numeric(df["level_id_source"], errors="coerce").astype("Int64")
                df = df.assign(level_id_source=level_id)
                df["source_level"] = df["level_id_source"].map(LEVEL_LABELS).fillna("Other")
                level_opts = ["All"] + [lvl for lvl in LEVEL_LABELS.values() if lvl in set(df["source_level"].dropna().tolist())]
                level = st.selectbox(f"{title} source level", level_opts, index=0, key=f"{key_root}_level")
                if level != "All":
                    df = df[df["source_level"] == level]
            else:
                df = _apply_levels_played_filter(
                    df,
                    label=f"{title} source level",
                    key=f"{key_root}_level",
                )
    
            df = _apply_team_filter(
                df,
                label=f"{title} team",
                key=f"{key_root}_team",
            )
            df = _apply_opening_day_status_filter(
                df,
                label=f"{title} Opening Day Status",
                key=f"{key_root}_opening_day_status",
            )
    
            pos_opts = _position_options(df)
            if len(pos_opts) > 1:
                selected_positions = st.multiselect(
                    f"{title} position",
                    pos_opts,
                    default=["All"],
                    key=f"{key_root}_position_multi",
                    format_func=lambda v: (
                        "All" if v == "All" else POSITION_FILTER_LABELS.get(str(v), str(v))
                    ),
                )
                df = _filter_by_positions(df, selected_positions)
            elif "position" in df.columns:
                pos_values = sorted(df["position"].dropna().astype(str).unique().tolist())
                position = st.selectbox(
                    f"{title} position",
                    ["All"] + pos_values,
                    index=0,
                    key=f"{key_root}_position",
                )
                if position != "All":
                    df = df[df["position"].astype(str) == position]
    
            if name_col in df.columns:
                player_names = sorted(df[name_col].dropna().astype(str).unique().tolist())
                title_l = str(title).strip().lower()
                supports_custom_player_list = ("hitters" in title_l) or ("pitchers" in title_l)
                if supports_custom_player_list:
                    player_mode = st.selectbox(
                        f"{title} player selection",
                        ["All", "Single", "Custom List"],
                        index=0,
                        key=f"{key_root}_player_mode",
                    )
                    if player_mode == "Single":
                        player = st.selectbox(
                            f"{title} player",
                            ["All"] + player_names,
                            index=0,
                            key=f"{key_root}_player",
                        )
                        if player != "All":
                            df = df[df[name_col].astype(str) == str(player)]
                    elif player_mode == "Custom List":
                        selected_players = st.multiselect(
                            f"{title} player list",
                            player_names,
                            default=[],
                            key=f"{key_root}_player_list",
                        )
                        if selected_players:
                            selected_set = {str(p) for p in selected_players}
                            df = df[df[name_col].astype(str).isin(selected_set)]
                else:
                    player = st.selectbox(
                        f"{title} player",
                        ["All"] + player_names,
                        index=0,
                        key=f"{key_root}_player",
                    )
                    if player != "All":
                        df = df[df[name_col].astype(str) == str(player)]
    
            quantity_candidates = [
                ("PA", "PA_proj_p50"),
                ("BBE", "bbe_proj_p50"),
                ("TBF", "TBF_marcel_proj_p50"),
                ("TBF", "TBF_proj_p50"),
                ("G", "G_marcel_proj_p50"),
                ("G", "G_proj_p50"),
                ("IP", "IP_marcel_proj_p50"),
                ("IP", "IP_proj_p50"),
                ("GS", "GS_marcel_proj_p50"),
                ("GS", "GS_proj_p50"),
            ]
            available_qty = []
            seen_labels: set[str] = set()
            for label, col in quantity_candidates:
                if col not in df.columns or label in seen_labels:
                    continue
                available_qty.append((label, col))
                seen_labels.add(label)
            if available_qty:
                st.caption("Minimum projected quantity filters (P50)")
            for label, col in available_qty:
                max_val = float(pd.to_numeric(df[col], errors="coerce").max())
                if not pd.notna(max_val):
                    max_val = 0.0
                step = 1.0 if max_val <= 100 else 5.0
                min_qty = st.number_input(
                    f"{title} min {label}",
                    min_value=0.0,
                    value=0.0,
                    step=step,
                    key=f"{key_root}_min_{col}",
                )
                if min_qty > 0:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    df = df[vals >= float(min_qty)]
    
            if "OPS_proj_p50" in df.columns:
                source_mu, source_sigma, target_sigma = _ops_env_reference_stats(df)
                ref_season = _resolve_ops_reference_season(df)
                if (
                    np.isfinite(source_mu)
                    and np.isfinite(source_sigma)
                    and source_sigma > 1e-12
                    and np.isfinite(target_sigma)
                    and target_sigma > 1e-12
                ):
                    apply_ops_env = st.checkbox(
                        f"{title} apply league OPS environment (z-space rescale)",
                        value=False,
                        key=f"{key_root}_ops_env_apply",
                        help=(
                            "Applies an OPS-anchored z-space environment shift across required "
                            "underlying rate stats using prior-season reference mu/sigma, then "
                            "recomputes counting stats and slashline outputs."
                        ),
                    )
                    ops_env_cols = st.columns(2)
                    with ops_env_cols[0]:
                        target_ops_median = st.number_input(
                            f"{title} target league OPS (median)",
                            min_value=0.0,
                            max_value=2.5,
                            value=float(np.round(source_mu, 3)),
                            step=0.001,
                            format="%.3f",
                            key=f"{key_root}_ops_env_target_mu",
                        )
                    with ops_env_cols[1]:
                        target_ops_sigma = st.number_input(
                            f"{title} target league OPS sigma",
                            min_value=0.001,
                            max_value=1.0,
                            value=float(np.round(target_sigma, 3)),
                            step=0.001,
                            format="%.3f",
                            key=f"{key_root}_ops_env_target_sigma",
                        )
                    ref_season_txt = (
                        str(int(ref_season))
                        if ref_season is not None and np.isfinite(ref_season)
                        else "unknown"
                    )
                    st.caption(
                        "OPS environment reference: "
                        f"source season={ref_season_txt}, "
                        "source mu=PA-weighted MLB OPS, "
                        f"source mu value={source_mu:.3f}, "
                        f"source sigma (mlb_pa>=100)={source_sigma:.3f}, "
                        f"target sigma={float(target_ops_sigma):.3f}. "
                        f"Calibration pool uses MLB PA >= {int(MLB_PA_REFERENCE_MIN)} in source season when available."
                    )
                    if bool(apply_ops_env):
                        df = _apply_ops_league_environment(
                            df,
                            target_ops_median=float(target_ops_median),
                            target_ops_sigma=float(target_ops_sigma),
                        )
                        # Refresh z-columns after OPS override so displayed z is coherent.
                        df = _add_derived_slashline_metrics(df)
                else:
                    st.caption(
                        "OPS environment rescale unavailable: could not derive stable OPS reference distribution."
                    )
    
            if (
                "ERA_proj_p50" in df.columns
                and "ER_per_IP_mlb_eq_non_ar_delta_proj_p50" in df.columns
            ):
                source_mu, source_sigma, target_sigma = _era_env_reference_stats(df)
                ref_season = _resolve_ops_reference_season(df)
                if (
                    np.isfinite(source_mu)
                    and np.isfinite(source_sigma)
                    and source_sigma > 1e-12
                    and np.isfinite(target_sigma)
                    and target_sigma > 1e-12
                ):
                    apply_era_env = st.checkbox(
                        f"{title} apply league ERA environment (z-space rescale)",
                        value=False,
                        key=f"{key_root}_era_env_apply",
                        help=(
                            "Applies an ERA-anchored z-space environment shift through "
                            "ER_per_IP projections, then recomputes displayed ERA outputs."
                        ),
                    )
                    era_env_cols = st.columns(2)
                    with era_env_cols[0]:
                        target_era_median = st.number_input(
                            f"{title} target league ERA (median)",
                            min_value=0.0,
                            max_value=15.0,
                            value=float(np.round(source_mu, 3)),
                            step=0.001,
                            format="%.3f",
                            key=f"{key_root}_era_env_target_mu",
                        )
                    with era_env_cols[1]:
                        target_era_sigma = st.number_input(
                            f"{title} target league ERA sigma",
                            min_value=0.001,
                            max_value=5.0,
                            value=float(np.round(target_sigma, 3)),
                            step=0.001,
                            format="%.3f",
                            key=f"{key_root}_era_env_target_sigma",
                        )
                    ref_season_txt = (
                        str(int(ref_season))
                        if ref_season is not None and np.isfinite(ref_season)
                        else "unknown"
                    )
                    st.caption(
                        "ERA environment reference: "
                        f"source season={ref_season_txt}, "
                        "source mu=IP-weighted MLB ERA, "
                        f"source mu value={source_mu:.3f}, "
                        f"source sigma (projected MLB reference)={source_sigma:.3f}, "
                        f"target sigma={float(target_era_sigma):.3f}. "
                        f"Calibration pool uses projected IP >= {int(MLB_IP_REFERENCE_MIN)} when available."
                    )
                    if bool(apply_era_env):
                        df = _apply_era_league_environment(
                            df,
                            target_era_median=float(target_era_median),
                            target_era_sigma=float(target_era_sigma),
                        )
                else:
                    st.caption(
                        "ERA environment rescale unavailable: could not derive stable ERA reference distribution."
                    )
    
            if "WHIP_proj_p50" in df.columns:
                source_mu, source_sigma, target_sigma = _whip_env_reference_stats(df)
                ref_season = _resolve_ops_reference_season(df)
                if (
                    np.isfinite(source_mu)
                    and np.isfinite(source_sigma)
                    and source_sigma > 1e-12
                    and np.isfinite(target_sigma)
                    and target_sigma > 1e-12
                ):
                    apply_whip_env = st.checkbox(
                        f"{title} apply league WHIP environment (z-space rescale)",
                        value=False,
                        key=f"{key_root}_whip_env_apply",
                        help=(
                            "Applies a WHIP-anchored z-space environment shift through "
                            "WHIP projections."
                        ),
                    )
                    whip_env_cols = st.columns(2)
                    with whip_env_cols[0]:
                        target_whip_median = st.number_input(
                            f"{title} target league WHIP (median)",
                            min_value=0.0,
                            max_value=5.0,
                            value=float(np.round(source_mu, 3)),
                            step=0.001,
                            format="%.3f",
                            key=f"{key_root}_whip_env_target_mu",
                        )
                    with whip_env_cols[1]:
                        target_whip_sigma = st.number_input(
                            f"{title} target league WHIP sigma",
                            min_value=0.001,
                            max_value=2.0,
                            value=float(np.round(target_sigma, 3)),
                            step=0.001,
                            format="%.3f",
                            key=f"{key_root}_whip_env_target_sigma",
                        )
                    ref_season_txt = (
                        str(int(ref_season))
                        if ref_season is not None and np.isfinite(ref_season)
                        else "unknown"
                    )
                    st.caption(
                        "WHIP environment reference: "
                        f"source season={ref_season_txt}, "
                        "source mu=IP-weighted MLB WHIP, "
                        f"source mu value={source_mu:.3f}, "
                        f"source sigma (projected MLB reference)={source_sigma:.3f}, "
                        f"target sigma={float(target_whip_sigma):.3f}. "
                        f"Calibration pool uses projected IP >= {int(MLB_IP_REFERENCE_MIN)} when available."
                    )
                    if bool(apply_whip_env):
                        df = _apply_whip_league_environment(
                            df,
                            target_whip_median=float(target_whip_median),
                            target_whip_sigma=float(target_whip_sigma),
                        )
                else:
                    st.caption(
                        "WHIP environment rescale unavailable: could not derive stable WHIP reference distribution."
                    )
    
            metric_bases_all = sorted(
                c[: -len("_proj_p50")] for c in df.columns if c.endswith("_proj_p50")
            )
            metric_base_set = set(metric_bases_all)
            metric_bases = [
                m
                for m in metric_bases_all
                if not (m.endswith("_reg") and (m[: -len("_reg")] in metric_base_set))
            ]
            pct_choices = st.multiselect(
                f"{title} percentiles",
                ["P20", "P25", "P50", "P75", "P80", "P600", "Spread"],
                default=["P50", "P25", "P75"],
                key=f"{key_root}_pct_cols",
            )
            bp_default_order = [
                "Points",
                "SGP",
                "Points/PA",
                "PA",
                "AB",
                "H",
                "HR",
                "Runs",
                "RBI",
                "SB",
                "CS",
                "K%",
                "BB%",
                "AVG",
                "OBP",
                "SLG",
                "OPS",
                "babip_recalc_rate_mlb_eq_non_ar_delta",
                "ISO",
            ]
            pitching_default_order = [
                "Points",
                "SGP",
                "Points/IP",
                "G",
                "GS",
                "IP",
                "TBF",
                "ERA",
                "WHIP",
                "SO",
                "W",
                "SV",
                "BB",
                "H",
                "HR",
                "HBP",
                "K%",
                "BB%",
                "K-BB%",
                "BABIP",
                "Whiff%",
                "HR/BBE%",
                "GB%",
            ]
            pitcher_kpi_default_order = [
                "Points",
                "SGP",
                "Points/IP",
                "TBF",
                "IP",
                "GS",
                "stuff_raw",
                "stuff_raw_reg",
                "fastball_velo_reg",
                "max_velo_reg",
                "fastball_vaa_reg",
                "FA_pct_reg",
                "BB_rpm_reg",
                "SwStr_reg",
                "Zone_reg",
                "Ball_pct_reg",
                "Z_Contact_reg",
                "Chase_reg",
                "CSW_reg",
                "LA_gte_20_reg",
                "LA_lte_0_reg",
                "rel_z_reg",
                "rel_x_reg",
                "ext_reg",
            ]
            kpi_default_order = [
                "Points",
                "SGP",
                "Points/PA",
                "PA",
                "bbe",
                "damage_rate",
                "EV90th",
                "max_EV",
                "SEAGER",
                "pull_FB_pct",
                "LA_gte_20",
                "LA_lte_0",
                "selection_skill",
                "hittable_pitches_taken",
                "chase",
                "z_con",
                "secondary_whiff_pct",
                "whiffs_vs_95",
                "contact_vs_avg",
            ]
            kpi_signature = {
                "damage_rate",
                "EV90th",
                "max_EV",
                "SEAGER",
                "chase",
                "z_con",
                "damage_rate_reg",
                "EV90th_reg",
                "max_EV_reg",
                "SEAGER_reg",
                "chase_reg",
                "z_con_reg",
            }
            use_kpi_defaults = bool(prefer_kpi_defaults) and (
                sum(1 for m in kpi_signature if m in metric_bases) >= 4
            )
            is_pitching_dataset = _looks_like_pitching_metric_set(metric_bases)
            is_pitcher_kpi_dataset = _looks_like_pitcher_kpi_metric_set(metric_bases)
            if is_pitcher_kpi_dataset:
                requested_default_order = pitcher_kpi_default_order
            elif use_kpi_defaults:
                requested_default_order = kpi_default_order
            elif is_pitching_dataset:
                requested_default_order = pitching_default_order
            else:
                requested_default_order = bp_default_order
            if is_pitching_dataset:
                requested_default_order = _prefer_marcel_volume_bases(requested_default_order)
            default_metrics = [m for m in requested_default_order if m in metric_bases]
            if not default_metrics:
                default_metrics = metric_bases[:10]
            selected_metrics = st.multiselect(
                f"{title} metrics",
                metric_bases,
                default=default_metrics,
                key=f"{key_root}_metrics",
            )
            if (
                not bool(prefer_kpi_defaults)
                and set(selected_metrics) == {"PA"}
                and len(default_metrics) > 1
            ):
                selected_metrics = list(default_metrics)
    
            metric_order = [m for m in requested_default_order if m in selected_metrics]
            metric_order += [m for m in selected_metrics if m not in metric_order]
    
            sort_options = list(selected_metrics)
            for metric_name in ["Points/PA", "Points/IP"]:
                col_name = f"{metric_name}_proj_p50"
                if col_name in df.columns and metric_name not in sort_options:
                    sort_options.append(metric_name)
            if "ADP" in df.columns:
                sort_options.append("ADP")
            if sort_options and "P50" in pct_choices:
                if is_pitching_dataset:
                    if "K-BB%" in selected_metrics:
                        sort_default_idx = selected_metrics.index("K-BB%")
                    elif "IP_marcel" in selected_metrics:
                        sort_default_idx = selected_metrics.index("IP_marcel")
                    elif "IP" in selected_metrics:
                        sort_default_idx = selected_metrics.index("IP")
                    else:
                        sort_default_idx = 0
                else:
                    sort_default_idx = selected_metrics.index("OPS") if "OPS" in selected_metrics else 0
                sort_metric = st.selectbox(
                    f"{title} sort metric (P50)",
                    sort_options,
                    index=sort_default_idx,
                    key=f"{key_root}_sort_metric",
                )
                sort_dir_default_idx = 1 if sort_metric == "ADP" else 0
                sort_direction = st.selectbox(
                    f"{title} sort direction",
                    ["Descending", "Ascending"],
                    index=sort_dir_default_idx,
                    key=f"{key_root}_sort_direction",
                )
                sort_col = "ADP" if sort_metric == "ADP" else f"{sort_metric}_proj_p50"
                if sort_col in df.columns:
                    df = df.sort_values(sort_col, ascending=(sort_direction == "Ascending"))
            st.form_submit_button("Apply Filters", type="primary")
    
    if df.empty:
        st.info("No rows after filters. Clear one or more filters to view data.")
        return

    hidden_cols = set(hide_display_cols or set())
    show_cols = [
        c
        for c in [
            name_col,
            id_col,
            "age_used",
            "volatility_index",
            "ADP",
            "Pick",
            "team",
            "team_abbreviations",
            "opening_day_status_40man",
            "is_mlb_ip_source",
            "is_mlb_pa_source",
            "is_mlb_sb_source",
            "position",
            "appeared_in_MLB",
        ]
        if c in df.columns and c not in hidden_cols
    ]

    # Requested display pattern: first p50 list in order, then adjacent lower/upper pairs.
    if "P50" in pct_choices:
        for metric in metric_order:
            col = f"{metric}_proj_p50"
            if col in df.columns:
                show_cols.append(col)
    for permanent_metric in ["Points/PA", "Points/IP"]:
        col = f"{permanent_metric}_proj_p50"
        if col in df.columns and col not in show_cols:
            show_cols.append(col)
    if ("P25" in pct_choices) or ("P75" in pct_choices):
        for metric in metric_order:
            c25 = f"{metric}_proj_p25"
            c75 = f"{metric}_proj_p75"
            if "P25" in pct_choices and c25 in df.columns:
                show_cols.append(c25)
            if "P75" in pct_choices and c75 in df.columns:
                show_cols.append(c75)
    if ("P20" in pct_choices) or ("P80" in pct_choices):
        for metric in metric_order:
            c20 = f"{metric}_proj_p20"
            c80 = f"{metric}_proj_p80"
            if "P20" in pct_choices and c20 in df.columns:
                show_cols.append(c20)
            if "P80" in pct_choices and c80 in df.columns:
                show_cols.append(c80)
    if "P600" in pct_choices:
        for metric in metric_order:
            col = f"{metric}_proj_p600"
            if col in df.columns:
                show_cols.append(col)
    if "Spread" in pct_choices:
        for metric in metric_order:
            col = f"{metric}_proj_spread"
            if col in df.columns:
                show_cols.append(col)
    if not show_cols:
        st.info("No columns selected to display.")
        return
    display_df = df[show_cols].copy()
    stats_source = df[show_cols].copy()
    flipped_bases = set(FLIPPED_PERCENTILE_BASES)
    if _looks_like_pitching_metric_set(metric_order):
        flipped_bases.discard("K%")
        flipped_bases.update(PITCHING_FLIPPED_PERCENTILE_BASES)
    display_df = _flip_percentile_values_for_bases(
        display_df,
        bases=flipped_bases,
    )
    stats_source = _flip_percentile_values_for_bases(
        stats_source,
        bases=flipped_bases,
    )
    rename_map = {
        col: _display_label_for_projection_col(col)
        for col in show_cols
    }
    rename_map = {old: new for old, new in rename_map.items() if new != old}
    if rename_map:
        display_df = display_df.rename(columns=rename_map)
        stats_source = stats_source.rename(columns=rename_map)

    display_df, stats_source = _apply_custom_arithmetic_columns(
        display_df,
        stats_source,
        key_prefix=f"{key_root}_calc_cols",
        table_label=title,
    )

    display_df = _apply_column_filters(
        display_df,
        key_prefix=f"{key_root}_table_col_filters",
        expander_label=f"{title} column filters",
    )
    if display_df.empty:
        st.info("No rows after column filters. Clear one or more filters to view data.")
        return
    stats_source = stats_source.loc[display_df.index].copy()

    total_rows = len(display_df)
    rows_default_idx = 3 if compact_ui else 2
    page_size_choice = st.selectbox(
        f"{title} rows per page",
        ["All", 50, 100, 200, 500],
        index=rows_default_idx,
        key=f"{key_root}_rows_per_page",
    )
    if page_size_choice == "All":
        page_df = display_df
    else:
        page_size = int(page_size_choice)
        total_pages = max(1, int(np.ceil(total_rows / page_size)))
        page = st.number_input(
            f"{title} page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=f"{key_root}_page_num",
        )
        start = (int(page) - 1) * page_size
        end = start + page_size
        page_df = display_df.iloc[start:end].copy()

    _render_projection_styled_table(
        page_df,
        stats_source=stats_source,
        metric_bases=metric_order,
    )


def _show_all_players_adp_points_view(
    hitters_df: pd.DataFrame,
    pitchers_df: pd.DataFrame,
    *,
    key_prefix: str = "all_players",
    compact_ui: bool = False,
) -> None:
    st.subheader("All Players (ADP + Points)")
    if hitters_df.empty and pitchers_df.empty:
        st.info("No hitter or pitcher projections are available for this view.")
        return

    def _pick_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
        for col in candidates:
            if col in df.columns:
                return df[col]
        return pd.Series(np.nan, index=df.index, dtype="float64")

    def _build_player_view(df: pd.DataFrame, *, player_type: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        out = _add_derived_points_metrics(df.copy())
        view = pd.DataFrame(index=out.index)
        is_pitcher_pool = str(player_type).strip().lower() == "pitcher"
        raw_name = _pick_series(out, ["hitter_name", "name", "player_name"]).astype(str).str.strip()
        view["Player"] = raw_name.where(raw_name.ne(""), np.nan)
        view["Player Type"] = str(player_type)
        view["ID"] = pd.to_numeric(
            _pick_series(out, ["batter_mlbid", "pitcher_mlbid", "mlbid"]),
            errors="coerce",
        ).astype("Int64")
        if is_pitcher_pool:
            pitcher_order = ["SP", "RP"]
            pos_tokens = out.apply(_roster_pitcher_tokens_from_row, axis=1)
            pos_text = pos_tokens.map(
                lambda toks: "/".join(tok for tok in pitcher_order if tok in set(toks))
            )
        else:
            hitter_order = ["C", "1B", "2B", "3B", "SS", "OF", "UT"]
            pos_tokens = out.apply(_roster_hitter_tokens_from_row, axis=1)
            pos_text = pos_tokens.map(
                lambda toks: "/".join(tok for tok in hitter_order if tok in set(toks))
            )
        raw_position = _pick_series(out, ["position"]).astype(str).replace({"nan": ""})
        view["Position"] = pos_text.where(pos_text.astype(str).str.len() > 0, raw_position)
        team_vals = _pick_series(out, ["team", "team_abbreviations", "team_abbreviation"]).astype(str)
        view["Team"] = team_vals.replace({"nan": ""})
        view["ADP"] = pd.to_numeric(_pick_series(out, ["ADP"]), errors="coerce")
        if "Pick" in out.columns:
            view["Pick"] = out["Pick"].astype(str).replace({"nan": ""})
        else:
            view["Pick"] = view["ADP"].map(_adp_to_pick_notation)
        view["PA_proj_p50"] = pd.to_numeric(_pick_series(out, ["PA_proj_p50"]), errors="coerce")
        view["IP_proj_p50"] = pd.to_numeric(
            _pick_series(out, ["IP_proj_p50", "IP_marcel_proj_p50"]),
            errors="coerce",
        )
        for tag in ("p25", "p50", "p75"):
            col = f"Points_proj_{tag}"
            view[col] = pd.to_numeric(_pick_series(out, [col]), errors="coerce")
            sgp_col = f"SGP_proj_{tag}"
            view[sgp_col] = pd.to_numeric(_pick_series(out, [sgp_col]), errors="coerce")
            ppa_col = f"Points/PA_proj_{tag}"
            pip_col = f"Points/IP_proj_{tag}"
            view[ppa_col] = pd.to_numeric(_pick_series(out, [ppa_col]), errors="coerce")
            view[pip_col] = pd.to_numeric(_pick_series(out, [pip_col]), errors="coerce")
        return view[view["Player"].notna()].copy()

    hitters_view = _build_player_view(hitters_df, player_type="Hitter")
    pitchers_view = _build_player_view(pitchers_df, player_type="Pitcher")
    combined = pd.concat([hitters_view, pitchers_view], ignore_index=True)
    if combined.empty:
        st.info("No player rows available for ADP/Points view.")
        return

    player_type_opts = ["Hitter", "Pitcher"]
    selected_types = st.multiselect(
        "Player type",
        player_type_opts,
        default=player_type_opts,
        key=f"{key_prefix}_player_types",
    )
    if selected_types:
        combined = combined[combined["Player Type"].isin(selected_types)].copy()

    player_query = st.text_input(
        "Player search",
        value="",
        key=f"{key_prefix}_player_search",
    ).strip()
    if player_query:
        combined = combined[
            combined["Player"].astype(str).str.contains(player_query, case=False, na=False)
        ].copy()

    sort_cols = st.columns(3)
    with sort_cols[0]:
        points_pct = st.selectbox(
            "Points percentile",
            ["P25", "P50", "P75"],
            index=1,
            key=f"{key_prefix}_points_pct",
        )
    sort_metric_opts = [
        f"Points ({points_pct})",
        f"SGP ({points_pct})",
        f"Points/PA ({points_pct})",
        f"Points/IP ({points_pct})",
        "ADP",
    ]
    with sort_cols[1]:
        sort_metric = st.selectbox(
            "Sort by",
            sort_metric_opts,
            index=0,
            key=f"{key_prefix}_sort_metric",
        )
    with sort_cols[2]:
        sort_dir_default_idx = 1 if sort_metric == "ADP" else 0
        sort_direction = st.selectbox(
            "Sort direction",
            ["Descending", "Ascending"],
            index=sort_dir_default_idx,
            key=f"{key_prefix}_sort_direction",
        )

    pct_tag = str(points_pct).lower()
    sort_map = {
        f"Points ({points_pct})": f"Points_proj_{pct_tag}",
        f"SGP ({points_pct})": f"SGP_proj_{pct_tag}",
        f"Points/PA ({points_pct})": f"Points/PA_proj_{pct_tag}",
        f"Points/IP ({points_pct})": f"Points/IP_proj_{pct_tag}",
        "ADP": "ADP",
    }
    sort_col = sort_map.get(str(sort_metric), f"Points_proj_{pct_tag}")
    if sort_col in combined.columns:
        combined = combined.sort_values(
            sort_col,
            ascending=(sort_direction == "Ascending"),
            na_position="last",
        )

    if combined.empty:
        st.info("No rows after filters. Clear one or more filters to view data.")
        return

    display_cols = [
        "Player",
        "Player Type",
        "Position",
        "Team",
        "ID",
        "ADP",
        "Pick",
        "SGP_proj_p25",
        "SGP_proj_p50",
        "SGP_proj_p75",
        "Points_proj_p25",
        "Points_proj_p50",
        "Points_proj_p75",
        "Points/PA_proj_p50",
        "Points/IP_proj_p50",
        "PA_proj_p50",
        "IP_proj_p50",
    ]
    display_cols = [c for c in display_cols if c in combined.columns]
    display_df = combined[display_cols].copy()
    display_df = display_df.rename(
        columns={
            "SGP_proj_p25": "SGP P25",
            "SGP_proj_p50": "SGP P50",
            "SGP_proj_p75": "SGP P75",
            "Points_proj_p25": "Points P25",
            "Points_proj_p50": "Points P50",
            "Points_proj_p75": "Points P75",
            "Points/PA_proj_p50": "Points/PA P50",
            "Points/IP_proj_p50": "Points/IP P50",
            "PA_proj_p50": "PA P50",
            "IP_proj_p50": "IP P50",
        }
    )
    display_df, _ = _apply_custom_arithmetic_columns(
        display_df,
        display_df.copy(),
        key_prefix=f"{key_prefix}_calc_cols",
        table_label="All Players",
    )

    display_df = _apply_column_filters(
        display_df,
        key_prefix=f"{key_prefix}_table_col_filters",
        expander_label="All Players column filters",
    )
    if display_df.empty:
        st.info("No rows after column filters. Clear one or more filters to view data.")
        return

    total_rows = len(display_df)
    rows_default_idx = 3 if compact_ui else 2
    page_size_choice = st.selectbox(
        "All Players rows per page",
        ["All", 50, 100, 200, 500],
        index=rows_default_idx,
        key=f"{key_prefix}_rows_per_page",
    )
    if page_size_choice == "All":
        page_df = display_df
    else:
        page_size = int(page_size_choice)
        total_pages = max(1, int(np.ceil(total_rows / page_size)))
        page = st.number_input(
            "All Players page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=f"{key_prefix}_page_num",
        )
        start = (int(page) - 1) * page_size
        end = start + page_size
        page_df = display_df.iloc[start:end].copy()

    st.dataframe(_apply_custom_rounding(page_df), width="stretch", hide_index=True)


def _show_two_stage_before_after(
    fused_df: pd.DataFrame,
    base_df: pd.DataFrame,
    *,
    key_prefix: str = "two_stage_compare",
) -> None:
    st.subheader("Before vs After (Two-Stage)")
    if fused_df.empty:
        st.info("Two-stage fused dataset is empty.")
        return
    if base_df.empty:
        st.info("Missing baseline traditional dataset for comparison.")
        return

    after = fused_df.copy()
    before = base_df.copy()
    after = _add_derived_points_metrics(after)
    before = _add_derived_points_metrics(before)
    if "mlbid" not in after.columns and "batter_mlbid" in after.columns:
        after["mlbid"] = pd.to_numeric(after["batter_mlbid"], errors="coerce")
    if "mlbid" not in before.columns and "batter_mlbid" in before.columns:
        before["mlbid"] = pd.to_numeric(before["batter_mlbid"], errors="coerce")
    if "mlbid" not in after.columns or "mlbid" not in before.columns:
        st.info("Comparison requires `mlbid` in both datasets.")
        return
    if "target_season" not in after.columns or "target_season" not in before.columns:
        st.info("Comparison requires `target_season` in both datasets.")
        return

    keys = ["mlbid", "target_season"]
    after["mlbid"] = pd.to_numeric(after["mlbid"], errors="coerce")
    before["mlbid"] = pd.to_numeric(before["mlbid"], errors="coerce")
    after["target_season"] = pd.to_numeric(after["target_season"], errors="coerce")
    before["target_season"] = pd.to_numeric(before["target_season"], errors="coerce")
    after = after[after["mlbid"].notna() & after["target_season"].notna()].copy()
    before = before[before["mlbid"].notna() & before["target_season"].notna()].copy()
    after["mlbid"] = after["mlbid"].astype("int64")
    before["mlbid"] = before["mlbid"].astype("int64")
    after["target_season"] = after["target_season"].astype("int64")
    before["target_season"] = before["target_season"].astype("int64")

    pct_choice = st.selectbox(
        "Comparison percentile",
        ["P25", "P50", "P75"],
        index=1,
        key=f"{key_prefix}_pct_choice",
    )
    pct_tag = str(pct_choice).lower()
    shared_metric_bases = sorted(
        {
            c[: -len(f"_proj_{pct_tag}")]
            for c in after.columns
            if c.endswith(f"_proj_{pct_tag}") and c in before.columns
        }
    )
    if not shared_metric_bases:
        st.info(
            f"No shared `*_proj_{pct_tag}` metrics found between before/after datasets."
        )
        return

    direct_adjusted = sorted(
        c[len("two_stage_resid_applied_") :]
        for c in after.columns
        if c.startswith("two_stage_resid_applied_")
    )
    if direct_adjusted:
        st.caption(
            "Directly adjusted metrics: "
            + ", ".join(direct_adjusted)
        )
    if "two_stage_applied" in after.columns:
        applied = pd.to_numeric(after["two_stage_applied"], errors="coerce").fillna(0.0)
        st.caption(
            f"Rows with stage-2 applied: {int((applied > 0).sum()):,} / {len(after):,}"
        )

    key_root = f"before_after_{key_prefix}".strip("_")
    if "target_season" in after.columns:
        season_opts = sorted(after["target_season"].dropna().astype(int).unique().tolist())
        season = st.selectbox(
            "Comparison season",
            ["All"] + season_opts,
            index=0,
            key=f"{key_root}_season",
        )
        if season != "All":
            after = after[after["target_season"].astype(int) == int(season)]

    if "level_id_source" in after.columns:
        level_id = pd.to_numeric(after["level_id_source"], errors="coerce").astype("Int64")
        after = after.assign(level_id_source=level_id)
        after["source_level"] = after["level_id_source"].map(LEVEL_LABELS).fillna("Other")
        level_opts = ["All"] + [lvl for lvl in LEVEL_LABELS.values() if lvl in set(after["source_level"].dropna().tolist())]
        level = st.selectbox("Comparison source level", level_opts, index=0, key=f"{key_root}_level")
        if level != "All":
            after = after[after["source_level"] == level]
    else:
        after = _apply_levels_played_filter(
            after,
            label="Comparison source level",
            key=f"{key_root}_level",
        )
    after = _apply_team_filter(
        after,
        label="Comparison team",
        key=f"{key_root}_team",
    )

    player_col = "hitter_name" if "hitter_name" in after.columns else ("player_name" if "player_name" in after.columns else None)
    if player_col:
        player_opts = ["All"] + sorted(after[player_col].dropna().astype(str).unique().tolist())
        player = st.selectbox("Comparison player", player_opts, index=0, key=f"{key_root}_player")
        if player != "All":
            after = after[after[player_col].astype(str) == player]

    if "PA_proj_p50" in after.columns:
        min_pa = st.number_input(
            "Comparison min PA (after, P50)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            key=f"{key_root}_min_pa",
        )
        if float(min_pa) > 0:
            pa_vals = pd.to_numeric(after["PA_proj_p50"], errors="coerce")
            after = after[pa_vals >= float(min_pa)]

    if after.empty:
        st.info("No rows after filters.")
        return

    bp_default_order = [
        "Points",
        "PA",
        "AB",
        "H",
        "HR",
        "Runs",
        "RBI",
        "SB",
        "CS",
        "K%",
        "BB%",
        "AVG",
        "OBP",
        "SLG",
        "OPS",
        "babip_recalc_rate_mlb_eq_non_ar_delta",
        "ISO",
    ]
    pitching_default_order = [
        "Points",
        "G",
        "GS",
        "IP",
        "TBF",
        "ERA",
        "WHIP",
        "SO",
        "W",
        "SV",
        "BB",
        "H",
        "HR",
        "HBP",
        "K%",
        "BB%",
        "K-BB%",
        "BABIP",
        "Whiff%",
        "HR/BBE%",
        "GB%",
    ]
    requested_default_order = (
        pitching_default_order
        if _looks_like_pitching_metric_set(shared_metric_bases)
        else bp_default_order
    )
    if _looks_like_pitching_metric_set(shared_metric_bases):
        requested_default_order = _prefer_marcel_volume_bases(requested_default_order)
    default_metrics = [m for m in requested_default_order if m in shared_metric_bases]
    if not default_metrics:
        default_metrics = shared_metric_bases[:12]
    selected_metrics = st.multiselect(
        f"Comparison metrics ({pct_choice})",
        shared_metric_bases,
        default=default_metrics,
        key=f"{key_root}_metrics",
    )
    if not selected_metrics:
        st.info("Select at least one metric.")
        return

    after_keep = [*keys]
    before_keep = [*keys]
    meta_cols = [c for c in [player_col, "team", "team_abbreviations", "two_stage_applied"] if c and c in after.columns]
    after_keep.extend(meta_cols)
    for m in selected_metrics:
        c = f"{m}_proj_{pct_tag}"
        if c in after.columns:
            after_keep.append(c)
        if c in before.columns:
            before_keep.append(c)

    after_v = after[sorted(set(after_keep), key=after_keep.index)].copy()
    before_v = before[sorted(set(before_keep), key=before_keep.index)].copy()
    comp = after_v.merge(before_v, on=keys, how="left", suffixes=("_after", "_before"))
    if comp.empty:
        st.info("No overlapping rows between before/after after filters.")
        return

    for m in selected_metrics:
        a_col = f"{m}_proj_{pct_tag}_after"
        b_col = f"{m}_proj_{pct_tag}_before"
        d_col = f"delta_{m}_proj_{pct_tag}"
        if a_col in comp.columns and b_col in comp.columns:
            comp[d_col] = (
                pd.to_numeric(comp[a_col], errors="coerce")
                - pd.to_numeric(comp[b_col], errors="coerce")
            )

    sort_choices = [
        f"delta_{m}_proj_{pct_tag}"
        for m in selected_metrics
        if f"delta_{m}_proj_{pct_tag}" in comp.columns
    ]
    sort_choice = st.selectbox(
        "Comparison sort",
        ["None"] + sort_choices,
        index=0,
        key=f"{key_root}_sort",
    )
    sort_asc = st.checkbox("Ascending sort", value=False, key=f"{key_root}_sort_asc")
    if sort_choice != "None" and sort_choice in comp.columns:
        comp = comp.sort_values(sort_choice, ascending=bool(sort_asc))

    show_cols = [c for c in [player_col, "mlbid", "team", "team_abbreviations", "two_stage_applied"] if c and c in comp.columns]
    for m in selected_metrics:
        a_col = f"{m}_proj_{pct_tag}_after"
        b_col = f"{m}_proj_{pct_tag}_before"
        d_col = f"delta_{m}_proj_{pct_tag}"
        if a_col in comp.columns:
            show_cols.append(a_col)
        if b_col in comp.columns:
            show_cols.append(b_col)
        if d_col in comp.columns:
            show_cols.append(d_col)

    out = comp[show_cols].copy()
    flipped_bases = set(FLIPPED_PERCENTILE_BASES)
    if _looks_like_pitching_metric_set(selected_metrics):
        flipped_bases.discard("K%")
        flipped_bases.update(PITCHING_FLIPPED_PERCENTILE_BASES)
    out = _flip_percentile_values_for_bases(
        out,
        bases=flipped_bases,
    )
    styled_source = _flip_percentile_values_for_bases(
        comp[show_cols].copy(),
        bases=flipped_bases,
    )
    out = _apply_column_filters(
        out,
        key_prefix=f"{key_root}_table_col_filters",
        expander_label="Comparison column filters",
    )
    if out.empty:
        st.info("No rows after column filters. Clear one or more filters to view data.")
        return
    styled_source = styled_source.loc[out.index].copy()
    _render_projection_styled_table(
        out,
        stats_source=styled_source,
        metric_bases=selected_metrics,
        enable_conditional_formatting=False,
    )


def _show_component_slash_historical_fit(
    df: pd.DataFrame,
    *,
    key_prefix: str = "component_slash",
) -> None:
    if df.empty:
        st.info(
            "Missing component slash predictions file. "
            "Run `build_component_slash_hitter_predictions.py` first."
        )
        return

    view = df.copy()
    key_root = f"component_slash_{key_prefix}".strip("_")

    if "season" in view.columns:
        season_opts = sorted(view["season"].dropna().astype(int).unique().tolist())
        season = st.selectbox(
            "Component Slash season",
            ["All"] + season_opts,
            index=0,
            key=f"{key_root}_season",
        )
        if season != "All":
            view = view[view["season"].astype(int) == int(season)]

    view = _apply_team_filter(
        view,
        label="Component Slash team",
        key=f"{key_root}_team",
    )

    player_col = "hitter_name" if "hitter_name" in view.columns else "player_display_text"
    if player_col in view.columns:
        player_opts = ["All"] + sorted(view[player_col].dropna().astype(str).unique().tolist())
        player = st.selectbox(
            "Component Slash player",
            player_opts,
            index=0,
            key=f"{key_root}_player",
        )
        if player != "All":
            view = view[view[player_col].astype(str) == player]

    if "plate_appearances_agg" in view.columns:
        min_pa = st.number_input(
            "Component Slash min PA",
            min_value=0.0,
            value=0.0,
            step=10.0,
            key=f"{key_root}_min_pa",
        )
        if float(min_pa) > 0:
            pa_vals = pd.to_numeric(view["plate_appearances_agg"], errors="coerce")
            view = view[pa_vals >= float(min_pa)]

    if view.empty:
        st.info("No rows after filters. Clear one or more filters to view data.")
        return

    available_metrics = [
        m
        for m, alias in COMPONENT_METRICS.items()
        if f"actual_{alias}" in view.columns and f"pred_{alias}" in view.columns
    ]
    selected_metrics = st.multiselect(
        "Component Slash metrics",
        available_metrics,
        default=available_metrics,
        key=f"{key_root}_metrics",
    )
    if not selected_metrics:
        st.info("Select at least one metric.")
        return

    sort_choices = ["PA", *selected_metrics]
    sort_metric = st.selectbox(
        "Component Slash sort",
        sort_choices,
        index=0,
        key=f"{key_root}_sort",
    )
    ascending = st.checkbox(
        "Ascending sort",
        value=False,
        key=f"{key_root}_sort_asc",
    )

    if sort_metric == "PA" and "plate_appearances_agg" in view.columns:
        view = view.sort_values("plate_appearances_agg", ascending=ascending)
    elif sort_metric in COMPONENT_METRICS:
        alias = COMPONENT_METRICS[sort_metric]
        pred_col = f"pred_{alias}"
        if pred_col in view.columns:
            view = view.sort_values(pred_col, ascending=ascending)

    base_cols = [
        c
        for c in [
            "season",
            "hitter_name",
            "player_display_text",
            "mlbid",
            "team_abbreviation",
            "plate_appearances_agg",
            "at_bats_agg",
        ]
        if c in view.columns
    ]
    metric_cols: list[str] = []
    rename_map: dict[str, str] = {}
    for metric in selected_metrics:
        alias = COMPONENT_METRICS[metric]
        for stem in ["actual", "pred", "delta", "abs_err"]:
            col = f"{stem}_{alias}"
            if col in view.columns:
                metric_cols.append(col)
                if stem == "actual":
                    rename_map[col] = f"actual_{metric}"
                elif stem == "pred":
                    rename_map[col] = f"pred_{metric}"
                elif stem == "delta":
                    rename_map[col] = f"delta_{metric}"
                else:
                    rename_map[col] = f"abs_err_{metric}"

    display = view[base_cols + metric_cols].copy().rename(columns=rename_map)
    display = _apply_column_filters(
        display,
        key_prefix=f"{key_root}_table_col_filters",
        expander_label="Component Slash column filters",
    )
    if display.empty:
        st.info("No rows after column filters. Clear one or more filters to view data.")
        return
    _render_projection_styled_table(
        display,
        stats_source=display,
        metric_bases=selected_metrics,
        enable_conditional_formatting=False,
    )


def _show_component_slash_from_kpi(
    df: pd.DataFrame,
    *,
    key_prefix: str = "component_slash_kpi",
) -> None:
    if df.empty:
        st.info(
            "Missing KPI-based component slash file. "
            "Run `build_component_slash_from_kpi_projections.py` first."
        )
        return

    view = df.copy()
    key_root = f"component_slash_{key_prefix}".strip("_")

    if "target_season" in view.columns:
        season_opts = sorted(view["target_season"].dropna().astype(int).unique().tolist())
        season = st.selectbox(
            "KPI Component Slash target season",
            ["All"] + season_opts,
            index=0,
            key=f"{key_root}_season",
        )
        if season != "All":
            view = view[view["target_season"].astype(int) == int(season)]

    view = _apply_team_filter(
        view,
        label="KPI Component Slash team",
        key=f"{key_root}_team",
    )

    if "hitter_name" in view.columns:
        player_opts = ["All"] + sorted(view["hitter_name"].dropna().astype(str).unique().tolist())
        player = st.selectbox(
            "KPI Component Slash player",
            player_opts,
            index=0,
            key=f"{key_root}_player",
        )
        if player != "All":
            view = view[view["hitter_name"].astype(str) == player]

    if "PA_proj_p50" in view.columns:
        min_pa = st.number_input(
            "KPI Component Slash min projected PA (p50)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            key=f"{key_root}_min_pa",
        )
        if float(min_pa) > 0:
            pa_vals = pd.to_numeric(view["PA_proj_p50"], errors="coerce")
            view = view[pa_vals >= float(min_pa)]

    if view.empty:
        st.info("No rows after filters. Clear one or more filters to view data.")
        return

    pct_options = [
        tag
        for tag in ["p20", "p25", "p50", "p75", "p80"]
        if any(
            f"{base}_proj_{tag}" in view.columns for base in COMPONENT_KPI_METRICS.values()
        )
    ]
    if not pct_options:
        st.info("No supported percentile columns found in KPI component slash file.")
        return
    selected_pcts = st.multiselect(
        "KPI Component Slash percentiles",
        pct_options,
        default=["p50"] if "p50" in pct_options else [pct_options[0]],
        key=f"{key_root}_pcts",
    )
    if not selected_pcts:
        st.info("Select at least one percentile.")
        return

    available_metrics = [
        metric
        for metric, base in COMPONENT_KPI_METRICS.items()
        if any(f"{base}_proj_{tag}" in view.columns for tag in selected_pcts)
    ]
    selected_metrics = st.multiselect(
        "KPI Component Slash metrics",
        available_metrics,
        default=available_metrics,
        key=f"{key_root}_metrics",
    )
    if not selected_metrics:
        st.info("Select at least one metric.")
        return

    include_kpi_compare = st.checkbox(
        "Show KPI baseline and model-minus-KPI deltas",
        value=True,
        key=f"{key_root}_show_kpi_compare",
    )

    sort_metric = st.selectbox(
        "KPI Component Slash sort metric",
        ["PA", *selected_metrics],
        index=0,
        key=f"{key_root}_sort_metric",
    )
    sort_pct = st.selectbox(
        "KPI Component Slash sort percentile",
        selected_pcts,
        index=selected_pcts.index("p50") if "p50" in selected_pcts else 0,
        key=f"{key_root}_sort_pct",
    )
    sort_asc = st.checkbox(
        "Ascending sort",
        value=False,
        key=f"{key_root}_sort_asc",
    )

    if sort_metric == "PA" and "PA_proj_p50" in view.columns:
        view = view.sort_values("PA_proj_p50", ascending=sort_asc)
    elif sort_metric in COMPONENT_KPI_METRICS:
        sort_col = f"{COMPONENT_KPI_METRICS[sort_metric]}_proj_{sort_pct}"
        if sort_col in view.columns:
            view = view.sort_values(sort_col, ascending=sort_asc)

    base_cols = [
        c
        for c in [
            "target_season",
            "hitter_name",
            "batter_mlbid",
            "team",
            "position",
            "PA_proj_p50",
            "source_season",
            "age_used",
            "volatility_index",
        ]
        if c in view.columns
    ]

    metric_cols: list[str] = []
    rename_map: dict[str, str] = {}
    for metric in selected_metrics:
        model_base = COMPONENT_KPI_METRICS[metric]
        kpi_base = KPI_COMPARE_BASES[metric]
        for tag in selected_pcts:
            model_col = f"{model_base}_proj_{tag}"
            if model_col in view.columns:
                metric_cols.append(model_col)
                rename_map[model_col] = f"model_{metric}_proj_{tag}"
            if include_kpi_compare:
                kpi_col = f"kpi_{kpi_base}_proj_{tag}"
                delta_col = f"delta_{model_base}_minus_kpi_proj_{tag}"
                if kpi_col in view.columns:
                    metric_cols.append(kpi_col)
                    rename_map[kpi_col] = f"kpi_{metric}_proj_{tag}"
                if delta_col in view.columns:
                    metric_cols.append(delta_col)
                    rename_map[delta_col] = f"delta_model_minus_kpi_{metric}_proj_{tag}"

    display = view[base_cols + metric_cols].copy().rename(columns=rename_map)
    display = _apply_column_filters(
        display,
        key_prefix=f"{key_root}_table_col_filters",
        expander_label="KPI Component Slash column filters",
    )
    if display.empty:
        st.info("No rows after column filters. Clear one or more filters to view data.")
        return
    _render_projection_styled_table(
        display,
        stats_source=display,
        metric_bases=selected_metrics,
        enable_conditional_formatting=False,
    )


def show_roster_manager(
    hitters_two_stage: pd.DataFrame,
    pitchers_two_stage: pd.DataFrame,
    *,
    key_prefix: str = "roster_manager",
) -> None:
    st.subheader("Roster Manager")
    if hitters_two_stage.empty or pitchers_two_stage.empty:
        st.info("Roster Manager requires both two-stage hitter and pitcher datasets.")
        return

    base_hitter_pool = _build_roster_player_pool(hitters_two_stage, for_pitchers=False)
    base_pitcher_pool = _build_roster_player_pool(pitchers_two_stage, for_pitchers=True)
    if base_hitter_pool.empty or base_pitcher_pool.empty:
        st.info("Roster Manager player pools are empty after preprocessing.")
        return

    state_key = f"{key_prefix}_state"
    live_state = _roster_prepare_live_state(st.session_state.get(state_key))
    st.session_state[state_key] = live_state
    state = st.session_state[state_key]

    widget_sync_pending_key = f"{key_prefix}_external_widget_sync_pending"

    def _prime_widget_state_from_roster_state() -> None:
        starters = state.get("starters", {}) if isinstance(state.get("starters"), dict) else {}
        starter_pct = (
            state.get("starter_percentiles", {})
            if isinstance(state.get("starter_percentiles"), dict)
            else {}
        )
        starter_manual = (
            state.get("starter_manual_stats", {})
            if isinstance(state.get("starter_manual_stats"), dict)
            else {}
        )
        default_pct_local = _roster_norm_percentile(state.get("default_percentile", "p50"))
        for slot in ROSTER_HITTER_STARTER_SLOTS:
            st.session_state[f"{key_prefix}_starter_h_{slot}"] = _roster_parse_int(starters.get(slot))
            st.session_state[f"{key_prefix}_starter_h_pct_{slot}"] = _roster_norm_percentile(
                starter_pct.get(slot),
                default=default_pct_local,
            )
            st.session_state[f"{key_prefix}_starter_h_manual_toggle_{slot}"] = (
                _roster_normalize_manual_stats(starter_manual.get(slot), hitter=True) is not None
            )
        for slot in ROSTER_PITCHER_STARTER_SLOTS:
            st.session_state[f"{key_prefix}_starter_p_{slot}"] = _roster_parse_int(starters.get(slot))
            st.session_state[f"{key_prefix}_starter_p_pct_{slot}"] = _roster_norm_percentile(
                starter_pct.get(slot),
                default=default_pct_local,
            )
            st.session_state[f"{key_prefix}_starter_p_manual_toggle_{slot}"] = (
                _roster_normalize_manual_stats(starter_manual.get(slot), hitter=False) is not None
            )

        for idx, row in enumerate(state.get("hitter_reserves", [])):
            row_dict = row if isinstance(row, dict) else {"player_id": row}
            st.session_state[f"{key_prefix}_hres_pick_{idx}"] = _roster_parse_int(
                row_dict.get("player_id")
            )
            st.session_state[f"{key_prefix}_hres_pct_{idx}"] = _roster_norm_percentile(
                row_dict.get("percentile"),
                default=default_pct_local,
            )
            st.session_state[f"{key_prefix}_hres_manual_toggle_{idx}"] = (
                _roster_normalize_manual_stats(row_dict.get("manual_stats"), hitter=True) is not None
            )

        for idx, row in enumerate(state.get("pitcher_reserves", [])):
            row_dict = row if isinstance(row, dict) else {"player_id": row}
            st.session_state[f"{key_prefix}_pres_pick_{idx}"] = _roster_parse_int(
                row_dict.get("player_id")
            )
            st.session_state[f"{key_prefix}_pres_pct_{idx}"] = _roster_norm_percentile(
                row_dict.get("percentile"),
                default=default_pct_local,
            )
            st.session_state[f"{key_prefix}_pres_manual_toggle_{idx}"] = (
                _roster_normalize_manual_stats(row_dict.get("manual_stats"), hitter=False) is not None
            )

    if bool(st.session_state.get(widget_sync_pending_key, False)):
        _roster_reset_ui_widget_state(key_prefix)
        _prime_widget_state_from_roster_state()
        st.session_state[widget_sync_pending_key] = False

    def _slot_natural_pos(slot: str) -> str:
        """Return the default position a roster slot should count toward."""
        natural_map = {
            "C1": "C", "C2": "C",
            "1B": "1B", "2B": "2B", "SS": "SS", "3B": "3B",
            "OF1": "OF", "OF2": "OF", "OF3": "OF", "OF4": "OF", "OF5": "OF",
            "MI": "2B", "CI": "1B", "UT": "UT",
            "SP1": "SP", "SP2": "SP", "SP3": "SP", "SP4": "SP",
            "SP5": "SP", "SP6": "SP", "SP7/RP3": "SP",
            "RP1": "RP", "RP2": "RP",
        }
        return natural_map.get(slot, "UT")

    loaded_file_name_key = f"{key_prefix}_external_loaded_file_name"
    export_file_name_key = f"{key_prefix}_external_export_file_name"
    export_name_synced_from_key = f"{key_prefix}_external_export_name_synced_from"
    load_notice_key = f"{key_prefix}_external_load_notice"
    loaded_file_name = str(st.session_state.get(loaded_file_name_key, "") or "").strip()
    last_synced_loaded_name = str(
        st.session_state.get(export_name_synced_from_key, "") or ""
    ).strip()
    if msg := str(st.session_state.pop(load_notice_key, "") or "").strip():
        st.success(msg)
    if export_file_name_key not in st.session_state:
        st.session_state[export_file_name_key] = loaded_file_name or "roster_manager_roster.json"
        st.session_state[export_name_synced_from_key] = loaded_file_name
    elif loaded_file_name and (loaded_file_name != last_synced_loaded_name):
        st.session_state[export_file_name_key] = loaded_file_name
        st.session_state[export_name_synced_from_key] = loaded_file_name
    elif (not loaded_file_name) and (not str(st.session_state.get(export_file_name_key, "") or "").strip()):
        st.session_state[export_file_name_key] = "roster_manager_roster.json"
        st.session_state[export_name_synced_from_key] = ""

    top_cols = st.columns([1, 1, 2])
    with top_cols[0]:
        default_pct_choice = st.selectbox(
            "Default percentile",
            list(ROSTER_PERCENTILE_OPTIONS),
            index=list(ROSTER_PERCENTILE_OPTIONS).index(
                _roster_norm_percentile(state.get("default_percentile", "p50"))
            ),
            key=f"{key_prefix}_pct_choice",
            help="Used as default for new slots/reserves. Each player row can be set independently.",
        )
    state["default_percentile"] = _roster_norm_percentile(default_pct_choice)
    state["percentile"] = state["default_percentile"]
    with top_cols[1]:
        uploaded_roster_file = st.file_uploader(
            "Load roster JSON",
            type=["json"],
            key=f"{key_prefix}_upload_roster_json",
        )
    with top_cols[2]:
        save_file_name_input = st.text_input(
            "Save file name",
            key=export_file_name_key,
            placeholder="my_roster.json",
        )
        if loaded_file_name:
            st.caption(f"Loaded file: `{loaded_file_name}`")
        else:
            st.caption("Loaded file: none")

    save_file_name = str(save_file_name_input or "").strip() or "roster_manager_roster.json"
    if not save_file_name.lower().endswith(".json"):
        save_file_name = f"{save_file_name}.json"

    action_cols = st.columns([1, 1, 1])
    with action_cols[0]:
        if st.button("Load uploaded roster", key=f"{key_prefix}_load_uploaded_btn"):
            if uploaded_roster_file is None:
                st.warning("Upload a JSON roster file first.")
            else:
                compat_team_name: str | None = None
                try:
                    raw_import = json.loads(uploaded_roster_file.getvalue().decode("utf-8"))
                except Exception:
                    st.error("Uploaded file is not valid JSON.")
                    raw_import = None
                if raw_import is not None:
                    raw_roster_state = raw_import
                    if isinstance(raw_import, dict) and isinstance(raw_import.get("teams"), dict):
                        team_map = {
                            str(name): payload
                            for name, payload in raw_import.get("teams", {}).items()
                            if str(name).strip()
                        }
                        if not team_map:
                            st.error("Uploaded file has no roster entries.")
                            raw_roster_state = None
                        else:
                            compat_team_name = sorted(team_map.keys())[0]
                            raw_roster_state = team_map[compat_team_name]
                    if raw_roster_state is not None:
                        loaded_state, dropped = _deserialize_roster_state(
                            raw_roster_state,
                            base_hitter_pool,
                            base_pitcher_pool,
                        )
                        st.session_state[state_key] = loaded_state
                        state = st.session_state[state_key]
                        loaded_file_name = str(uploaded_roster_file.name or "").strip()
                        if loaded_file_name:
                            st.session_state[loaded_file_name_key] = loaded_file_name
                        st.session_state[widget_sync_pending_key] = True
                        notices = ["Roster loaded from uploaded JSON."]
                        if compat_team_name is not None:
                            notices.append(
                                "Uploaded file used legacy multi-team format; "
                                f"loaded first team `{compat_team_name}`."
                            )
                        if dropped > 0:
                            notices.append(
                                f"Dropped {int(dropped)} invalid/duplicate player selections."
                            )
                        st.session_state[load_notice_key] = " ".join(notices)
                        st.rerun()
    with action_cols[1]:
        pass
    with action_cols[2]:
        if st.button("Clear roster", key=f"{key_prefix}_clear_btn"):
            st.session_state[state_key] = _roster_default_state()
            state = st.session_state[state_key]
            _roster_reset_ui_widget_state(key_prefix)
            st.success("Roster cleared.")

    with st.expander("External JSON save/load notes", expanded=False):
        st.caption(
            "This workflow no longer uses in-session/local app saves. "
            "Use `Save current roster JSON` to download and `Load roster JSON` to restore."
        )

    def _render_hitter_manual_inputs(
        *,
        row_key: str,
        defaults: dict[str, float] | None,
    ) -> dict[str, float]:
        base_defaults = _roster_normalize_manual_stats(defaults, hitter=True) or {
            k: 0.0 for k in ROSTER_HITTER_MANUAL_COUNT_STATS
        }
        cols = st.columns(6)
        out: dict[str, float] = {}
        for idx, stat in enumerate(ROSTER_HITTER_MANUAL_COUNT_STATS):
            with cols[idx]:
                out[stat] = float(
                    st.number_input(
                        stat,
                        min_value=0.0,
                        value=float(base_defaults.get(stat, 0.0)),
                        step=1.0,
                        key=f"{key_prefix}_{row_key}_manual_{stat}",
                    )
                )
        derived = _roster_hitter_stats_from_manual(out) or {}
        st.caption(
            f"Manual counts (auto AVG): AVG={float(derived.get('AVG', float('nan'))):.3f}"
        )
        return out

    def _render_pitcher_manual_inputs(
        *,
        row_key: str,
        defaults: dict[str, float] | None,
    ) -> dict[str, float]:
        base_defaults = _roster_normalize_manual_stats(defaults, hitter=False) or {
            k: 0.0 for k in ROSTER_PITCHER_MANUAL_COUNT_STATS
        }
        cols = st.columns(7)
        out: dict[str, float] = {}
        for idx, stat in enumerate(ROSTER_PITCHER_MANUAL_COUNT_STATS):
            with cols[idx]:
                out[stat] = float(
                    st.number_input(
                        stat,
                        min_value=0.0,
                        value=float(base_defaults.get(stat, 0.0)),
                        step=(0.1 if stat == "IP" else 1.0),
                        key=f"{key_prefix}_{row_key}_manual_{stat}",
                    )
                )
        derived = _roster_pitcher_stats_from_manual(out) or {}
        whip_val = float(derived.get("WHIP", float("nan")))
        era_val = float(derived.get("ERA", float("nan")))
        whip_txt = f"{whip_val:.3f}" if np.isfinite(whip_val) else "n/a"
        era_txt = f"{era_val:.3f}" if np.isfinite(era_val) else "n/a"
        st.caption(f"Manual counts (auto WHIP/ERA): WHIP={whip_txt}, ERA={era_txt}")
        return out

    st.markdown("### Manual Placeholders")
    st.caption(
        "Create a hitter or pitcher that is missing from the dataset. "
        "These placeholders are added to the selector lists below and saved into roster JSON."
    )
    placeholder_action_cols = st.columns([1, 1, 4])
    with placeholder_action_cols[0]:
        if st.button("Add hitter placeholder", key=f"{key_prefix}_add_custom_hitter"):
            _roster_add_custom_player(state, hitter=True)
            st.rerun()
    with placeholder_action_cols[1]:
        if st.button("Add pitcher placeholder", key=f"{key_prefix}_add_custom_pitcher"):
            _roster_add_custom_player(state, hitter=False)
            st.rerun()

    remove_custom_hitter_id: int | None = None
    remove_custom_pitcher_id: int | None = None

    for idx, player in enumerate(state.get("custom_hitters", [])):
        player_id = int(player.get("player_id"))
        header_cols = st.columns([3, 1, 2, 1, 1])
        with header_cols[0]:
            name = st.text_input(
                f"Hitter Placeholder {idx + 1}",
                value=str(player.get("name") or ""),
                key=f"{key_prefix}_custom_h_name_{player_id}",
            )
        with header_cols[1]:
            team = st.text_input(
                "Team",
                value=str(player.get("team") or "FA"),
                key=f"{key_prefix}_custom_h_team_{player_id}",
            )
        with header_cols[2]:
            positions = st.multiselect(
                "Positions",
                list(ROSTER_PLACEHOLDER_HITTER_POSITIONS),
                default=[
                    tok
                    for tok in player.get("positions", [])
                    if tok in set(ROSTER_PLACEHOLDER_HITTER_POSITIONS)
                ],
                key=f"{key_prefix}_custom_h_pos_{player_id}",
                help="Leave all positions empty to make the player UT-only.",
            )
        with header_cols[3]:
            plate_appearances = st.number_input(
                "PA",
                min_value=0.0,
                value=float(_roster_parse_nonneg_float(player.get("plate_appearances")) or 0.0),
                step=1.0,
                key=f"{key_prefix}_custom_h_pa_{player_id}",
            )
        with header_cols[4]:
            if st.button("Remove", key=f"{key_prefix}_custom_h_remove_{player_id}"):
                remove_custom_hitter_id = player_id
        manual_counts = _render_hitter_manual_inputs(
            row_key=f"custom_h_{player_id}",
            defaults=_roster_normalize_placeholder_manual_stats(player.get("manual_stats"), hitter=True),
        )
        state["custom_hitters"][idx] = _roster_normalize_custom_player_entry(
            {
                "player_id": player_id,
                "name": name,
                "team": team,
                "positions": positions,
                "plate_appearances": plate_appearances,
                "manual_stats": manual_counts,
            },
            hitter=True,
        ) or player

    for idx, player in enumerate(state.get("custom_pitchers", [])):
        player_id = int(player.get("player_id"))
        header_cols = st.columns([3, 1, 2, 2])
        with header_cols[0]:
            name = st.text_input(
                f"Pitcher Placeholder {idx + 1}",
                value=str(player.get("name") or ""),
                key=f"{key_prefix}_custom_p_name_{player_id}",
            )
        with header_cols[1]:
            team = st.text_input(
                "Team",
                value=str(player.get("team") or "FA"),
                key=f"{key_prefix}_custom_p_team_{player_id}",
            )
        with header_cols[2]:
            roles = st.multiselect(
                "Roles",
                list(ROSTER_PLACEHOLDER_PITCHER_ROLES),
                default=[
                    tok
                    for tok in player.get("roles", [])
                    if tok in set(ROSTER_PLACEHOLDER_PITCHER_ROLES)
                ],
                key=f"{key_prefix}_custom_p_roles_{player_id}",
            )
        with header_cols[3]:
            if st.button("Remove", key=f"{key_prefix}_custom_p_remove_{player_id}"):
                remove_custom_pitcher_id = player_id
        manual_counts = _render_pitcher_manual_inputs(
            row_key=f"custom_p_{player_id}",
            defaults=_roster_normalize_placeholder_manual_stats(player.get("manual_stats"), hitter=False),
        )
        state["custom_pitchers"][idx] = _roster_normalize_custom_player_entry(
            {
                "player_id": player_id,
                "name": name,
                "team": team,
                "roles": roles,
                "manual_stats": manual_counts,
            },
            hitter=False,
        ) or player

    if remove_custom_hitter_id is not None:
        state["custom_hitters"] = [
            player
            for player in state.get("custom_hitters", [])
            if _roster_parse_int(player.get("player_id")) != int(remove_custom_hitter_id)
        ]
        _roster_clear_player_from_state(state, int(remove_custom_hitter_id))
        _roster_normalize_custom_players_in_state(state)
        st.rerun()
    if remove_custom_pitcher_id is not None:
        state["custom_pitchers"] = [
            player
            for player in state.get("custom_pitchers", [])
            if _roster_parse_int(player.get("player_id")) != int(remove_custom_pitcher_id)
        ]
        _roster_clear_player_from_state(state, int(remove_custom_pitcher_id))
        _roster_normalize_custom_players_in_state(state)
        st.rerun()

    _roster_normalize_custom_players_in_state(state)
    hitter_pool = _combine_roster_player_pools(
        base_hitter_pool.sort_values("label").reset_index(drop=True),
        state.get("custom_hitters", []),
        for_pitchers=False,
    )
    pitcher_pool = _combine_roster_player_pools(
        base_pitcher_pool.sort_values("label").reset_index(drop=True),
        state.get("custom_pitchers", []),
        for_pitchers=True,
    )
    hitter_pool_by_id = hitter_pool.set_index("mlbid", drop=False)
    pitcher_pool_by_id = pitcher_pool.set_index("mlbid", drop=False)
    hitter_tokens_by_id = hitter_pool_by_id["elig_tokens"].to_dict()
    pitcher_tokens_by_id = pitcher_pool_by_id["elig_tokens"].to_dict()
    hitter_labels = hitter_pool_by_id["label"].to_dict()
    pitcher_labels = pitcher_pool_by_id["label"].to_dict()
    hitter_ids_sorted = hitter_pool["mlbid"].astype(int).tolist()
    pitcher_ids_sorted = pitcher_pool["mlbid"].astype(int).tolist()
    hitter_placeholder_ids = {
        int(player["player_id"]) for player in state.get("custom_hitters", []) if "player_id" in player
    }
    pitcher_placeholder_ids = {
        int(player["player_id"]) for player in state.get("custom_pitchers", []) if "player_id" in player
    }

    dropped_from_state = _roster_validate_live_state(
        state,
        hitter_pool_by_id=hitter_pool_by_id,
        pitcher_pool_by_id=pitcher_pool_by_id,
        hitter_tokens_by_id=hitter_tokens_by_id,
        pitcher_tokens_by_id=pitcher_tokens_by_id,
    )
    if dropped_from_state > 0:
        _roster_reset_ui_widget_state(key_prefix)
        _prime_widget_state_from_roster_state()
        st.caption(f"Roster Manager normalized {int(dropped_from_state)} invalid/duplicate stored selections.")

    def _starter_candidate_ids(slot: str, *, hitter: bool, current_id: int | None) -> list[int]:
        blocked = _roster_selected_player_ids(state)
        if current_id is not None:
            blocked.discard(current_id)
        out_ids: list[int] = []
        if hitter:
            for pid in hitter_ids_sorted:
                if pid in blocked:
                    continue
                toks = tuple(hitter_tokens_by_id.get(pid, tuple()))
                if _roster_hitter_slot_is_eligible(slot, toks):
                    out_ids.append(int(pid))
        else:
            for pid in pitcher_ids_sorted:
                if pid in blocked:
                    continue
                toks = tuple(pitcher_tokens_by_id.get(pid, tuple()))
                if _roster_pitcher_slot_is_eligible(slot, toks):
                    out_ids.append(int(pid))
        if current_id is not None and current_id not in out_ids:
            out_ids.append(int(current_id))
        labels = hitter_labels if hitter else pitcher_labels
        return sorted(set(out_ids), key=lambda pid: labels.get(pid, str(pid)))

    def _reserve_candidate_ids(*, hitter: bool, current_id: int | None) -> list[int]:
        blocked = _roster_selected_player_ids(state)
        if current_id is not None:
            blocked.discard(current_id)
        base_ids = hitter_ids_sorted if hitter else pitcher_ids_sorted
        out_ids = [int(pid) for pid in base_ids if int(pid) not in blocked]
        if current_id is not None and current_id not in out_ids:
            out_ids.append(int(current_id))
        labels = hitter_labels if hitter else pitcher_labels
        return sorted(set(out_ids), key=lambda pid: labels.get(pid, str(pid)))

    def _hitter_stats_for_player(
        pid: int,
        *,
        pct: str,
        custom_pa: float | None = None,
    ) -> dict[str, float]:
        row = hitter_pool_by_id.loc[int(pid)]
        out_stats: dict[str, float] = {}
        pct = _roster_norm_percentile(pct, default=state.get("default_percentile", "p50"))
        for stat, base in ROSTER_HITTER_TO_PROJ_BASE.items():
            col = f"{base}_proj_{pct}"
            out_stats[stat] = _roster_numeric_from_row(row, col)
        pa_base = _roster_numeric_from_row(row, f"PA_proj_{pct}")
        target_pa = _roster_parse_nonneg_float(custom_pa)
        if (
            target_pa is not None
            and pd.notna(pa_base)
            and np.isfinite(pa_base)
            and float(pa_base) > 0.0
        ):
            scale = float(target_pa) / float(pa_base)
            for stat in ["HR", "R", "RBI", "SB", "AB", "H"]:
                raw = out_stats.get(stat, float("nan"))
                if pd.notna(raw) and np.isfinite(float(raw)):
                    out_stats[stat] = float(raw) * scale
            ab = out_stats.get("AB", float("nan"))
            h = out_stats.get("H", float("nan"))
            if (
                pd.notna(ab)
                and np.isfinite(float(ab))
                and float(ab) > 0.0
                and pd.notna(h)
                and np.isfinite(float(h))
            ):
                out_stats["AVG"] = float(h) / float(ab)
            else:
                out_stats["AVG"] = float("nan")
        return out_stats

    def _pitcher_stats_for_player(
        pid: int,
        *,
        pct: str,
        custom_ip: float | None = None,
    ) -> dict[str, float]:
        row = pitcher_pool_by_id.loc[int(pid)]
        out_stats: dict[str, float] = {}
        pct = _roster_norm_percentile(pct, default=state.get("default_percentile", "p50"))
        for stat, base in ROSTER_PITCHER_TO_PROJ_BASE.items():
            col = f"{base}_proj_{pct}"
            out_stats[stat] = _roster_numeric_from_row(row, col)
        era = out_stats.get("ERA", float("nan"))
        ip = out_stats.get("IP", float("nan"))
        if pd.notna(era) and pd.notna(ip) and np.isfinite(era) and np.isfinite(ip) and float(ip) > 0.0:
            out_stats["ER"] = float(era) * float(ip) / 9.0
        else:
            out_stats["ER"] = float("nan")
        target_ip = _roster_parse_nonneg_float(custom_ip)
        if (
            target_ip is not None
            and pd.notna(ip)
            and np.isfinite(float(ip))
            and float(ip) > 0.0
        ):
            scale = float(target_ip) / float(ip)
            out_stats["IP"] = float(target_ip)
            for stat in ["W", "K", "SV", "H", "BB", "ER"]:
                raw = out_stats.get(stat, float("nan"))
                if pd.notna(raw) and np.isfinite(float(raw)):
                    out_stats[stat] = float(raw) * scale
            h_val = out_stats.get("H", float("nan"))
            bb_val = out_stats.get("BB", float("nan"))
            er_val = out_stats.get("ER", float("nan"))
            ip_val = out_stats.get("IP", float("nan"))
            if (
                pd.notna(ip_val)
                and np.isfinite(float(ip_val))
                and float(ip_val) > 0.0
                and pd.notna(h_val)
                and np.isfinite(float(h_val))
                and pd.notna(bb_val)
                and np.isfinite(float(bb_val))
            ):
                out_stats["WHIP"] = (float(h_val) + float(bb_val)) / float(ip_val)
            else:
                out_stats["WHIP"] = float("nan")
            if (
                pd.notna(ip_val)
                and np.isfinite(float(ip_val))
                and float(ip_val) > 0.0
                and pd.notna(er_val)
                and np.isfinite(float(er_val))
            ):
                out_stats["ERA"] = 9.0 * float(er_val) / float(ip_val)
            else:
                out_stats["ERA"] = float("nan")
        return out_stats

    st.markdown("### Hitters (Starters)")
    st.caption("Pick players below — no page refresh until Apply Changes is clicked. Position picker defaults to slot position. Custom PA and manual stats are set after applying.")
    with st.form(f"{key_prefix}_hitter_starters_form"):
        for slot in ROSTER_HITTER_STARTER_SLOTS:
            current_id = _roster_parse_int(state["starters"].get(slot))
            current_pct = _roster_norm_percentile(
                state.get("starter_percentiles", {}).get(slot),
                default=state.get("default_percentile", "p50"),
            )
            opt_ids = _starter_candidate_ids(slot, hitter=True, current_id=current_id)
            options: list[int | None] = [None, *opt_ids]
            if current_id is not None and current_id not in options:
                options.append(current_id)
            default_idx = options.index(current_id) if current_id in options else 0
            _pos_opts: list[str] = []
            if current_id is not None and current_id in hitter_pool_by_id.index:
                _pi_toks = set(hitter_tokens_by_id.get(current_id, tuple()))
                _pos_opts = [p for p in ["C", "1B", "2B", "SS", "3B", "OF"] if p in _pi_toks]
                _pos_opts.append("UT")
            row_cols = st.columns([3, 1, 2])
            with row_cols[0]:
                st.selectbox(
                    slot,
                    options,
                    index=default_idx,
                    key=f"{key_prefix}_starter_h_{slot}",
                    format_func=lambda v: _roster_display_pick_label(v, hitter_labels, empty_label="(empty)"),
                )
            with row_cols[1]:
                st.selectbox(
                    f"{slot} pct",
                    list(ROSTER_PERCENTILE_OPTIONS),
                    index=list(ROSTER_PERCENTILE_OPTIONS).index(current_pct),
                    key=f"{key_prefix}_starter_h_pct_{slot}",
                    label_visibility="collapsed",
                )
            with row_cols[2]:
                if _pos_opts:
                    _ekey = f"{key_prefix}_helig_slot_{slot}"
                    natural = _slot_natural_pos(slot)
                    default_pos = natural if natural in _pos_opts else _pos_opts[0]
                    stored_pos = st.session_state.get(_ekey)
                    valid_pos = stored_pos if stored_pos in _pos_opts else default_pos
                    st.selectbox(
                        "Counts toward",
                        _pos_opts,
                        index=_pos_opts.index(valid_pos),
                        key=_ekey,
                        label_visibility="collapsed",
                    )
        _h_starter_apply = st.form_submit_button("Apply Changes", type="primary")
    if _h_starter_apply:
        for slot in ROSTER_HITTER_STARTER_SLOTS:
            state["starters"][slot] = _roster_parse_int(
                st.session_state.get(f"{key_prefix}_starter_h_{slot}")
            )
            state["starter_percentiles"][slot] = _roster_norm_percentile(
                st.session_state.get(f"{key_prefix}_starter_h_pct_{slot}"),
                default=state.get("default_percentile", "p50"),
            )
        st.rerun()

    hitter_rows: list[dict[str, object]] = []
    for slot in ROSTER_HITTER_STARTER_SLOTS:
        current_id = _roster_parse_int(state["starters"].get(slot))
        current_custom_pa = _roster_parse_nonneg_float(
            state.get("starter_custom_volume", {}).get(slot)
        )
        current_manual_stats = _roster_normalize_manual_stats(
            state.get("starter_manual_stats", {}).get(slot),
            hitter=True,
        )
        current_pct = _roster_norm_percentile(
            state.get("starter_percentiles", {}).get(slot),
            default=state.get("default_percentile", "p50"),
        )
        if current_id is not None and current_id in hitter_pool_by_id.index:
            base_pa = _roster_numeric_from_row(
                hitter_pool_by_id.loc[int(current_id)],
                f"PA_proj_{current_pct}",
            )
            pa_widget_key = (
                f"{key_prefix}_starter_h_pa_{slot}_"
                f"{current_id}_{current_pct}"
            )
            default_pa_value = current_custom_pa
            if default_pa_value is None:
                default_pa_value = (
                    float(base_pa)
                    if (pd.notna(base_pa) and np.isfinite(float(base_pa)) and float(base_pa) >= 0.0)
                    else 0.0
                )
            vol_cols = st.columns([2, 2, 1])
            with vol_cols[0]:
                st.caption(f"**{slot}** — {hitter_labels.get(current_id, str(current_id))}")
            with vol_cols[1]:
                custom_pa_input = st.number_input(
                    f"{slot} PA",
                    min_value=0.0,
                    value=float(default_pa_value),
                    step=1.0,
                    key=pa_widget_key,
                    label_visibility="collapsed",
                )
            with vol_cols[2]:
                use_manual = st.checkbox(
                    f"{slot} manual",
                    value=(current_manual_stats is not None),
                    key=f"{key_prefix}_starter_h_manual_toggle_{slot}",
                    label_visibility="collapsed",
                )
            state["starter_percentiles"][slot] = current_pct
            parsed_custom_pa = _roster_parse_nonneg_float(custom_pa_input)
            if (
                parsed_custom_pa is not None
                and pd.notna(base_pa)
                and np.isfinite(float(base_pa))
                and abs(float(parsed_custom_pa) - float(base_pa)) <= 1e-6
            ):
                parsed_custom_pa = None
            state["starter_custom_volume"][slot] = parsed_custom_pa
            if use_manual:
                projected_seed = _hitter_stats_for_player(
                    current_id,
                    pct=current_pct,
                    custom_pa=state["starter_custom_volume"].get(slot),
                )
                seed_manual = current_manual_stats or {
                    stat: float(projected_seed.get(stat, 0.0))
                    if np.isfinite(float(projected_seed.get(stat, float("nan"))))
                    else 0.0
                    for stat in ROSTER_HITTER_MANUAL_COUNT_STATS
                }
                with st.container():
                    entered_manual = _render_hitter_manual_inputs(
                        row_key=f"starter_h_{slot}",
                        defaults=seed_manual,
                    )
                state["starter_manual_stats"][slot] = _roster_normalize_manual_stats(
                    entered_manual,
                    hitter=True,
                )
            else:
                state["starter_manual_stats"][slot] = None
        else:
            state["starter_custom_volume"][slot] = None
            state["starter_manual_stats"][slot] = None

        row_out: dict[str, object] = {
            "Slot": slot,
            "Pct": (
                "MANUAL"
                if state.get("starter_manual_stats", {}).get(slot) is not None
                or (current_id in hitter_placeholder_ids)
                else str(state.get("starter_percentiles", {}).get(slot, "p50")).upper()
            ),
            "Player": _roster_display_pick_label(current_id, hitter_labels, empty_label=""),
        }
        for stat in ROSTER_HITTER_STATS:
            row_out[stat] = float("nan")
        if current_id is not None and current_id in hitter_pool_by_id.index:
            manual_stats = _roster_hitter_stats_from_manual(
                state.get("starter_manual_stats", {}).get(slot)
            )
            stats = manual_stats or _hitter_stats_for_player(
                current_id,
                pct=current_pct,
                custom_pa=state.get("starter_custom_volume", {}).get(slot),
            )
            for stat in ROSTER_HITTER_STATS:
                row_out[stat] = stats.get(stat, float("nan"))
        hitter_rows.append(row_out)

    hitter_df = pd.DataFrame(hitter_rows)
    if not hitter_df.empty:
        hitter_df = _roster_apply_int_display(hitter_df, ["HR", "R", "RBI", "SB", "AB", "H"])
        hitter_df["AVG"] = pd.to_numeric(hitter_df["AVG"], errors="coerce").round(3)
    st.dataframe(hitter_df, width="stretch", hide_index=True)

    hitter_hr = float(pd.to_numeric(hitter_df.get("HR"), errors="coerce").sum(skipna=True))
    hitter_r = float(pd.to_numeric(hitter_df.get("R"), errors="coerce").sum(skipna=True))
    hitter_rbi = float(pd.to_numeric(hitter_df.get("RBI"), errors="coerce").sum(skipna=True))
    hitter_sb = float(pd.to_numeric(hitter_df.get("SB"), errors="coerce").sum(skipna=True))
    hitter_ab = float(pd.to_numeric(hitter_df.get("AB"), errors="coerce").sum(skipna=True))
    hitter_h = float(pd.to_numeric(hitter_df.get("H"), errors="coerce").sum(skipna=True))
    hitter_avg = (hitter_h / hitter_ab) if hitter_ab > 0 else float("nan")
    hitter_benchmark_h = float(
        ROSTER_HITTER_BENCHMARK_TOTAL.get("H", ROSTER_HITTER_BENCHMARK_TOTAL["AVG"] * ROSTER_HITTER_BENCHMARK_TOTAL["AB"])
    )
    hitter_total_rows = pd.DataFrame(
        [
            {
                "Row": "Total",
                "HR": hitter_hr,
                "R": hitter_r,
                "RBI": hitter_rbi,
                "SB": hitter_sb,
                "AVG": hitter_avg,
                "AB": hitter_ab,
                "H": hitter_h,
            },
            {
                "Row": "80th",
                "HR": ROSTER_HITTER_BENCHMARK_TOTAL["HR"],
                "R": ROSTER_HITTER_BENCHMARK_TOTAL["R"],
                "RBI": ROSTER_HITTER_BENCHMARK_TOTAL["RBI"],
                "SB": ROSTER_HITTER_BENCHMARK_TOTAL["SB"],
                "AVG": ROSTER_HITTER_BENCHMARK_TOTAL["AVG"],
                "AB": ROSTER_HITTER_BENCHMARK_TOTAL["AB"],
                "H": hitter_benchmark_h,
            },
            {
                "Row": "Needed",
                "HR": ROSTER_HITTER_BENCHMARK_TOTAL["HR"] - hitter_hr,
                "R": ROSTER_HITTER_BENCHMARK_TOTAL["R"] - hitter_r,
                "RBI": ROSTER_HITTER_BENCHMARK_TOTAL["RBI"] - hitter_rbi,
                "SB": ROSTER_HITTER_BENCHMARK_TOTAL["SB"] - hitter_sb,
                "AVG": ROSTER_HITTER_BENCHMARK_TOTAL["AVG"] - hitter_avg if pd.notna(hitter_avg) else float("nan"),
                "AB": ROSTER_HITTER_BENCHMARK_TOTAL["AB"] - hitter_ab,
                "H": hitter_benchmark_h - hitter_h,
            },
        ]
    )
    hitter_total_rows = _roster_apply_int_display(hitter_total_rows, ["HR", "R", "RBI", "SB", "AB", "H"])
    hitter_total_rows["AVG"] = pd.to_numeric(hitter_total_rows["AVG"], errors="coerce").round(3)
    hitter_fail_cols: set[str] = set()
    if pd.notna(hitter_avg) and float(hitter_avg) < float(ROSTER_HITTER_BENCHMARK_TOTAL["AVG"]):
        hitter_fail_cols.add("AVG")
    hitter_total_view = _roster_style_needed_cells(
        hitter_total_rows,
        fail_cols=hitter_fail_cols,
    )
    st.dataframe(hitter_total_view, width="stretch", hide_index=True)
    hitter_points_total = (
        -1.0 * hitter_ab
        + 4.0 * hitter_h
        + 2.0 * hitter_r
        + 6.0 * hitter_hr
        + 2.0 * hitter_rbi
        + 5.0 * hitter_sb
    )
    hitter_points_benchmark = (
        -1.0 * float(ROSTER_HITTER_BENCHMARK_TOTAL["AB"])
        + 4.0 * float(hitter_benchmark_h)
        + 2.0 * float(ROSTER_HITTER_BENCHMARK_TOTAL["R"])
        + 6.0 * float(ROSTER_HITTER_BENCHMARK_TOTAL["HR"])
        + 2.0 * float(ROSTER_HITTER_BENCHMARK_TOTAL["RBI"])
        + 5.0 * float(ROSTER_HITTER_BENCHMARK_TOTAL["SB"])
    )
    hitter_points_needed = hitter_points_benchmark - hitter_points_total
    hitter_points_rows = pd.DataFrame(
        [
            {"Row": "Points Total", "Points": hitter_points_total},
            {"Row": "Points 80th", "Points": hitter_points_benchmark},
            {"Row": "Points Needed", "Points": hitter_points_needed},
        ]
    )
    hitter_points_rows = _roster_apply_int_display(hitter_points_rows, ["Points"])
    st.dataframe(hitter_points_rows, width="stretch", hide_index=True)

    hitter_slot_denom = float(len(ROSTER_HITTER_STARTER_SLOTS))
    hitter_avg_rows = pd.DataFrame(
        [
            {
                "Row": "80th pctile from each spot",
                **ROSTER_HITTER_BENCHMARK_PER_SPOT,
            },
            {
                "Row": "Current average from each spot",
                "HR": hitter_hr / hitter_slot_denom if hitter_slot_denom > 0 else float("nan"),
                "R": hitter_r / hitter_slot_denom if hitter_slot_denom > 0 else float("nan"),
                "RBI": hitter_rbi / hitter_slot_denom if hitter_slot_denom > 0 else float("nan"),
                "SB": hitter_sb / hitter_slot_denom if hitter_slot_denom > 0 else float("nan"),
                "AVG": hitter_avg,
            },
        ]
    )
    hitter_avg_rows = _roster_apply_int_display(hitter_avg_rows, ["HR", "R", "RBI", "SB"])
    hitter_avg_rows["AVG"] = pd.to_numeric(hitter_avg_rows["AVG"], errors="coerce").round(3)
    st.dataframe(hitter_avg_rows, width="stretch", hide_index=True)

    st.markdown("### Hitters (Reserves)")
    st.caption("Pick players below - no page refresh until Apply Changes is clicked. Custom PA and manual stats update live after applying.")
    with st.form(f"{key_prefix}_hitter_reserves_form"):
        for idx in range(ROSTER_RESERVE_SLOTS):
            row = state["hitter_reserves"][idx]
            current_id = _roster_parse_int(row.get("player_id"))
            current_pct = _roster_norm_percentile(
                row.get("percentile"),
                default=state.get("default_percentile", "p50"),
            )
            opt_ids = _reserve_candidate_ids(hitter=True, current_id=current_id)
            h_res_options: list[int | None] = [None, *opt_ids]
            h_res_default_idx = h_res_options.index(current_id) if current_id in h_res_options else 0
            h_res_cols = st.columns([5, 1])
            with h_res_cols[0]:
                st.selectbox(
                    f"Hitter Reserve {idx + 1}",
                    h_res_options,
                    index=h_res_default_idx,
                    key=f"{key_prefix}_hres_pick_{idx}",
                    format_func=lambda v: _roster_display_pick_label(v, hitter_labels, empty_label="(empty)"),
                )
            with h_res_cols[1]:
                st.selectbox(
                    f"H{idx + 1} pct",
                    list(ROSTER_PERCENTILE_OPTIONS),
                    index=list(ROSTER_PERCENTILE_OPTIONS).index(current_pct),
                    key=f"{key_prefix}_hres_pct_{idx}",
                    label_visibility="collapsed",
                )
        _h_res_apply = st.form_submit_button("Apply Changes", type="primary")

    if _h_res_apply:
        for idx in range(ROSTER_RESERVE_SLOTS):
            picked_id = _roster_parse_int(st.session_state.get(f"{key_prefix}_hres_pick_{idx}"))
            picked_pct = _roster_norm_percentile(
                st.session_state.get(f"{key_prefix}_hres_pct_{idx}"),
                default=state.get("default_percentile", "p50"),
            )
            old_id = _roster_parse_int(state["hitter_reserves"][idx].get("player_id"))
            if picked_id != old_id:
                state["hitter_reserves"][idx]["custom_volume"] = None
                state["hitter_reserves"][idx]["manual_stats"] = None
            state["hitter_reserves"][idx]["player_id"] = picked_id
            state["hitter_reserves"][idx]["percentile"] = picked_pct
        st.rerun()

    # Live controls for committed (applied) hitter reserve players.
    for idx in range(ROSTER_RESERVE_SLOTS):
        row = state["hitter_reserves"][idx]
        picked_id = _roster_parse_int(row.get("player_id"))
        if picked_id is None or picked_id not in hitter_pool_by_id.index:
            continue
        current_pct = _roster_norm_percentile(
            row.get("percentile"),
            default=state.get("default_percentile", "p50"),
        )
        current_custom_pa = _roster_parse_nonneg_float(row.get("custom_volume"))
        current_manual_stats = _roster_normalize_manual_stats(row.get("manual_stats"), hitter=True)
        base_pa = _roster_numeric_from_row(
            hitter_pool_by_id.loc[int(picked_id)],
            f"PA_proj_{current_pct}",
        )
        player_label = _roster_display_pick_label(picked_id, hitter_labels, empty_label="")
        _hres_toks = set(hitter_tokens_by_id.get(picked_id, tuple()))
        _hres_elig = [p for p in ["C", "1B", "2B", "SS", "3B", "OF"] if p in _hres_toks]
        _hres_elig.append("UT")
        _hres_ekey = f"{key_prefix}_helig_hres_{idx}"
        _hres_ecur = st.session_state.get(_hres_ekey)
        if _hres_ecur not in _hres_elig:
            _hres_ecur = _hres_elig[0]
        live_cols = st.columns([3, 1, 1, 4])
        with live_cols[0]:
            pa_widget_key = f"{key_prefix}_hres_pa_{idx}_{picked_id}_{current_pct}"
            default_pa_value = current_custom_pa
            if default_pa_value is None:
                default_pa_value = (
                    float(base_pa)
                    if (pd.notna(base_pa) and np.isfinite(float(base_pa)) and float(base_pa) >= 0.0)
                    else 0.0
                )
            custom_pa_input = st.number_input(
                f"H{idx+1} {player_label} PA",
                min_value=0.0,
                value=float(default_pa_value),
                step=1.0,
                key=pa_widget_key,
            )
            parsed_custom_pa = _roster_parse_nonneg_float(custom_pa_input)
            if (
                parsed_custom_pa is not None
                and pd.notna(base_pa)
                and np.isfinite(float(base_pa))
                and abs(float(parsed_custom_pa) - float(base_pa)) <= 1e-6
            ):
                parsed_custom_pa = None
            state["hitter_reserves"][idx]["custom_volume"] = parsed_custom_pa
        with live_cols[1]:
            st.selectbox(
                f"H{idx+1} counts toward",
                _hres_elig,
                index=_hres_elig.index(_hres_ecur),
                key=_hres_ekey,
                label_visibility="collapsed",
            )
        with live_cols[2]:
            use_manual = st.checkbox(
                f"H{idx+1} manual",
                value=(current_manual_stats is not None),
                key=f"{key_prefix}_hres_manual_toggle_{idx}",
                label_visibility="collapsed",
            )
        if use_manual:
            projected_seed = _hitter_stats_for_player(
                picked_id,
                pct=current_pct,
                custom_pa=state["hitter_reserves"][idx]["custom_volume"],
            )
            seed_manual = current_manual_stats or {
                stat: float(projected_seed.get(stat, 0.0))
                if np.isfinite(float(projected_seed.get(stat, float("nan"))))
                else 0.0
                for stat in ROSTER_HITTER_MANUAL_COUNT_STATS
            }
            with live_cols[3]:
                entered_manual = _render_hitter_manual_inputs(
                    row_key=f"hres_{idx}",
                    defaults=seed_manual,
                )
            state["hitter_reserves"][idx]["manual_stats"] = _roster_normalize_manual_stats(
                entered_manual,
                hitter=True,
            )
        else:
            state["hitter_reserves"][idx]["manual_stats"] = None

    hitter_reserve_rows: list[dict[str, object]] = []
    for idx, row in enumerate(state.get("hitter_reserves", [])):
        pid = _roster_parse_int(row.get("player_id"))
        if pid is None:
            continue
        row_pct = _roster_norm_percentile(
            row.get("percentile"),
            default=state.get("default_percentile", "p50"),
        )
        row_custom_pa = _roster_parse_nonneg_float(row.get("custom_volume"))
        row_manual_stats = _roster_hitter_stats_from_manual(row.get("manual_stats"))
        row_out: dict[str, object] = {
            "Reserve Slot": f"H{idx + 1}",
            "Pct": (
                "MANUAL"
                if (row_manual_stats is not None) or (pid in hitter_placeholder_ids)
                else str(row_pct).upper()
            ),
            "Player": _roster_display_pick_label(pid, hitter_labels, empty_label=""),
        }
        for stat in ROSTER_HITTER_STATS:
            row_out[stat] = float("nan")
        if pid is not None and pid in hitter_pool_by_id.index:
            stats = row_manual_stats or _hitter_stats_for_player(
                pid,
                pct=row_pct,
                custom_pa=row_custom_pa,
            )
            for stat in ROSTER_HITTER_STATS:
                row_out[stat] = stats.get(stat, float("nan"))
        hitter_reserve_rows.append(row_out)
    hitter_reserve_df = pd.DataFrame(hitter_reserve_rows)
    if not hitter_reserve_df.empty:
        hitter_reserve_df = _roster_apply_int_display(hitter_reserve_df, ["HR", "R", "RBI", "SB", "AB", "H"])
        hitter_reserve_df["AVG"] = pd.to_numeric(hitter_reserve_df["AVG"], errors="coerce").round(3)
        st.dataframe(hitter_reserve_df, width="stretch", hide_index=True)
    else:
        st.caption("No hitter reserves applied yet.")

    st.markdown("### Pitchers (Starters)")
    st.caption("Pick players below — no page refresh until Apply Changes is clicked. Role picker defaults to slot role. Custom IP and manual stats are set after applying.")
    with st.form(f"{key_prefix}_pitcher_starters_form"):
        for slot in ROSTER_PITCHER_STARTER_SLOTS:
            current_id = _roster_parse_int(state["starters"].get(slot))
            current_pct = _roster_norm_percentile(
                state.get("starter_percentiles", {}).get(slot),
                default=state.get("default_percentile", "p50"),
            )
            opt_ids = _starter_candidate_ids(slot, hitter=False, current_id=current_id)
            options: list[int | None] = [None, *opt_ids]
            if current_id is not None and current_id not in options:
                options.append(current_id)
            default_idx = options.index(current_id) if current_id in options else 0
            _role_opts: list[str] = []
            if current_id is not None and current_id in pitcher_pool_by_id.index:
                _pp_toks = set(pitcher_tokens_by_id.get(current_id, tuple()))
                _role_opts = [p for p in ["SP", "RP"] if p in _pp_toks]
                if not _role_opts:
                    _role_opts = ["SP", "RP"]
            row_cols = st.columns([3, 1, 2])
            with row_cols[0]:
                st.selectbox(
                    slot,
                    options,
                    index=default_idx,
                    key=f"{key_prefix}_starter_p_{slot}",
                    format_func=lambda v: _roster_display_pick_label(v, pitcher_labels, empty_label="(empty)"),
                )
            with row_cols[1]:
                st.selectbox(
                    f"{slot} pct",
                    list(ROSTER_PERCENTILE_OPTIONS),
                    index=list(ROSTER_PERCENTILE_OPTIONS).index(current_pct),
                    key=f"{key_prefix}_starter_p_pct_{slot}",
                    label_visibility="collapsed",
                )
            with row_cols[2]:
                if len(_role_opts) > 1:
                    _rekey = f"{key_prefix}_pelig_slot_{slot}"
                    natural_role = _slot_natural_pos(slot)
                    default_role = natural_role if natural_role in _role_opts else _role_opts[0]
                    stored_role = st.session_state.get(_rekey)
                    valid_role = stored_role if stored_role in _role_opts else default_role
                    st.selectbox(
                        "Counts toward",
                        _role_opts,
                        index=_role_opts.index(valid_role),
                        key=_rekey,
                        label_visibility="collapsed",
                    )
                elif _role_opts:
                    st.caption(_role_opts[0])
                else:
                    st.empty()
        _p_starter_apply = st.form_submit_button("Apply Changes", type="primary")
    if _p_starter_apply:
        for slot in ROSTER_PITCHER_STARTER_SLOTS:
            state["starters"][slot] = _roster_parse_int(
                st.session_state.get(f"{key_prefix}_starter_p_{slot}")
            )
            state["starter_percentiles"][slot] = _roster_norm_percentile(
                st.session_state.get(f"{key_prefix}_starter_p_pct_{slot}"),
                default=state.get("default_percentile", "p50"),
            )
        st.rerun()

    pitcher_rows: list[dict[str, object]] = []
    for slot in ROSTER_PITCHER_STARTER_SLOTS:
        current_id = _roster_parse_int(state["starters"].get(slot))
        current_custom_ip = _roster_parse_nonneg_float(
            state.get("starter_custom_volume", {}).get(slot)
        )
        current_manual_stats = _roster_normalize_manual_stats(
            state.get("starter_manual_stats", {}).get(slot),
            hitter=False,
        )
        current_pct = _roster_norm_percentile(
            state.get("starter_percentiles", {}).get(slot),
            default=state.get("default_percentile", "p50"),
        )
        if current_id is not None and current_id in pitcher_pool_by_id.index:
            base_ip = _roster_numeric_from_row(
                pitcher_pool_by_id.loc[int(current_id)],
                f"IP_proj_{current_pct}",
            )
            ip_widget_key = (
                f"{key_prefix}_starter_p_ip_{slot}_"
                f"{current_id}_{current_pct}"
            )
            default_ip_value = current_custom_ip
            if default_ip_value is None:
                default_ip_value = (
                    float(base_ip)
                    if (pd.notna(base_ip) and np.isfinite(float(base_ip)) and float(base_ip) >= 0.0)
                    else 0.0
                )
            vol_cols = st.columns([2, 2, 1])
            with vol_cols[0]:
                st.caption(f"**{slot}** — {pitcher_labels.get(current_id, str(current_id))}")
            with vol_cols[1]:
                custom_ip_input = st.number_input(
                    f"{slot} IP",
                    min_value=0.0,
                    value=float(default_ip_value),
                    step=0.1,
                    key=ip_widget_key,
                    label_visibility="collapsed",
                )
            with vol_cols[2]:
                use_manual = st.checkbox(
                    f"{slot} manual",
                    value=(current_manual_stats is not None),
                    key=f"{key_prefix}_starter_p_manual_toggle_{slot}",
                    label_visibility="collapsed",
                )
            state["starter_percentiles"][slot] = current_pct
            parsed_custom_ip = _roster_parse_nonneg_float(custom_ip_input)
            if (
                parsed_custom_ip is not None
                and pd.notna(base_ip)
                and np.isfinite(float(base_ip))
                and abs(float(parsed_custom_ip) - float(base_ip)) <= 1e-6
            ):
                parsed_custom_ip = None
            state["starter_custom_volume"][slot] = parsed_custom_ip
            if use_manual:
                projected_seed = _pitcher_stats_for_player(
                    current_id,
                    pct=current_pct,
                    custom_ip=state["starter_custom_volume"].get(slot),
                )
                seed_manual = current_manual_stats or {
                    stat: float(projected_seed.get(stat, 0.0))
                    if np.isfinite(float(projected_seed.get(stat, float("nan"))))
                    else 0.0
                    for stat in ROSTER_PITCHER_MANUAL_COUNT_STATS
                }
                with st.container():
                    entered_manual = _render_pitcher_manual_inputs(
                        row_key=f"starter_p_{slot}",
                        defaults=seed_manual,
                    )
                state["starter_manual_stats"][slot] = _roster_normalize_manual_stats(
                    entered_manual,
                    hitter=False,
                )
            else:
                state["starter_manual_stats"][slot] = None
        else:
            state["starter_custom_volume"][slot] = None
            state["starter_manual_stats"][slot] = None

        row_out: dict[str, object] = {
            "Slot": slot,
            "Pct": (
                "MANUAL"
                if state.get("starter_manual_stats", {}).get(slot) is not None
                or (current_id in pitcher_placeholder_ids)
                else str(state.get("starter_percentiles", {}).get(slot, "p50")).upper()
            ),
            "Player": _roster_display_pick_label(current_id, pitcher_labels, empty_label=""),
        }
        for stat in ROSTER_PITCHER_STATS:
            row_out[stat] = float("nan")
        if current_id is not None and current_id in pitcher_pool_by_id.index:
            manual_stats = _roster_pitcher_stats_from_manual(
                state.get("starter_manual_stats", {}).get(slot)
            )
            stats = manual_stats or _pitcher_stats_for_player(
                current_id,
                pct=current_pct,
                custom_ip=state.get("starter_custom_volume", {}).get(slot),
            )
            for stat in ROSTER_PITCHER_STATS:
                row_out[stat] = stats.get(stat, float("nan"))
        pitcher_rows.append(row_out)
    pitcher_df = pd.DataFrame(pitcher_rows)
    if not pitcher_df.empty:
        pitcher_df = _roster_apply_int_display(pitcher_df, ["W", "K", "SV", "H", "BB", "ER"])
        pitcher_df["IP"] = pd.to_numeric(pitcher_df["IP"], errors="coerce").round(1)
        for col in ["WHIP", "ERA"]:
            pitcher_df[col] = pd.to_numeric(pitcher_df[col], errors="coerce").round(3)
    st.dataframe(pitcher_df, width="stretch", hide_index=True)

    pit_w = float(pd.to_numeric(pitcher_df.get("W"), errors="coerce").sum(skipna=True))
    pit_k = float(pd.to_numeric(pitcher_df.get("K"), errors="coerce").sum(skipna=True))
    pit_sv = float(pd.to_numeric(pitcher_df.get("SV"), errors="coerce").sum(skipna=True))
    pit_ip = float(pd.to_numeric(pitcher_df.get("IP"), errors="coerce").sum(skipna=True))
    pit_h = float(pd.to_numeric(pitcher_df.get("H"), errors="coerce").sum(skipna=True))
    pit_bb = float(pd.to_numeric(pitcher_df.get("BB"), errors="coerce").sum(skipna=True))
    pit_er = float(pd.to_numeric(pitcher_df.get("ER"), errors="coerce").sum(skipna=True))
    pit_whip = ((pit_h + pit_bb) / pit_ip) if pit_ip > 0 else float("nan")
    pit_era = ((9.0 * pit_er) / pit_ip) if pit_ip > 0 else float("nan")
    pitcher_total_rows = pd.DataFrame(
        [
            {
                "Row": "Total",
                "W": pit_w,
                "K": pit_k,
                "SV": pit_sv,
                "WHIP": pit_whip,
                "ERA": pit_era,
                "IP": pit_ip,
                "H": pit_h,
                "BB": pit_bb,
                "ER": pit_er,
            },
            {
                "Row": "80th pctile",
                **ROSTER_PITCHER_BENCHMARK_TOTAL,
            },
            {
                "Row": "Needed",
                "W": ROSTER_PITCHER_BENCHMARK_TOTAL["W"] - pit_w,
                "K": ROSTER_PITCHER_BENCHMARK_TOTAL["K"] - pit_k,
                "SV": ROSTER_PITCHER_BENCHMARK_TOTAL["SV"] - pit_sv,
                "WHIP": ROSTER_PITCHER_BENCHMARK_TOTAL["WHIP"] - pit_whip if pd.notna(pit_whip) else float("nan"),
                "ERA": ROSTER_PITCHER_BENCHMARK_TOTAL["ERA"] - pit_era if pd.notna(pit_era) else float("nan"),
                "IP": ROSTER_PITCHER_BENCHMARK_TOTAL["IP"] - pit_ip,
                "H": ROSTER_PITCHER_BENCHMARK_TOTAL["H"] - pit_h,
                "BB": ROSTER_PITCHER_BENCHMARK_TOTAL["BB"] - pit_bb,
                "ER": ROSTER_PITCHER_BENCHMARK_TOTAL["ER"] - pit_er,
            },
        ]
    )
    pitcher_total_rows = _roster_apply_int_display(pitcher_total_rows, ["W", "K", "SV", "H", "BB", "ER"])
    pitcher_total_rows["IP"] = pd.to_numeric(pitcher_total_rows["IP"], errors="coerce").round(1)
    for col in ["WHIP", "ERA"]:
        pitcher_total_rows[col] = pd.to_numeric(pitcher_total_rows[col], errors="coerce").round(3)
    pitcher_fail_cols: set[str] = set()
    if pd.notna(pit_era) and float(pit_era) > float(ROSTER_PITCHER_BENCHMARK_TOTAL["ERA"]):
        pitcher_fail_cols.add("ERA")
    if pd.notna(pit_whip) and float(pit_whip) > float(ROSTER_PITCHER_BENCHMARK_TOTAL["WHIP"]):
        pitcher_fail_cols.add("WHIP")
    pitcher_total_view = _roster_style_needed_cells(
        pitcher_total_rows,
        fail_cols=pitcher_fail_cols,
    )
    st.dataframe(pitcher_total_view, width="stretch", hide_index=True)
    pitcher_points_total = (
        3.0 * pit_ip
        - 1.0 * pit_h
        - 2.0 * pit_er
        - 1.0 * pit_bb
        + 1.0 * pit_k
        + 6.0 * pit_w
        + 8.0 * pit_sv
    )
    pitcher_points_benchmark = (
        3.0 * float(ROSTER_PITCHER_BENCHMARK_TOTAL["IP"])
        - 1.0 * float(ROSTER_PITCHER_BENCHMARK_TOTAL["H"])
        - 2.0 * float(ROSTER_PITCHER_BENCHMARK_TOTAL["ER"])
        - 1.0 * float(ROSTER_PITCHER_BENCHMARK_TOTAL["BB"])
        + 1.0 * float(ROSTER_PITCHER_BENCHMARK_TOTAL["K"])
        + 6.0 * float(ROSTER_PITCHER_BENCHMARK_TOTAL["W"])
        + 8.0 * float(ROSTER_PITCHER_BENCHMARK_TOTAL["SV"])
    )
    pitcher_points_needed = pitcher_points_benchmark - pitcher_points_total
    pitcher_points_rows = pd.DataFrame(
        [
            {"Row": "Points Total", "Points": pitcher_points_total},
            {"Row": "Points 80th", "Points": pitcher_points_benchmark},
            {"Row": "Points Needed", "Points": pitcher_points_needed},
        ]
    )
    pitcher_points_rows = _roster_apply_int_display(pitcher_points_rows, ["Points"])
    st.dataframe(pitcher_points_rows, width="stretch", hide_index=True)

    pitcher_slot_denom = float(len(ROSTER_PITCHER_STARTER_SLOTS))
    pitcher_avg_rows = pd.DataFrame(
        [
            {
                "Row": "80th pctile from each spot",
                **ROSTER_PITCHER_BENCHMARK_PER_SPOT,
            },
            {
                "Row": "Current average from each spot",
                "W": pit_w / pitcher_slot_denom if pitcher_slot_denom > 0 else float("nan"),
                "K": pit_k / pitcher_slot_denom if pitcher_slot_denom > 0 else float("nan"),
                "SV": pit_sv / pitcher_slot_denom if pitcher_slot_denom > 0 else float("nan"),
                "WHIP": pit_whip,
                "ERA": pit_era,
            },
        ]
    )
    pitcher_avg_rows = _roster_apply_int_display(pitcher_avg_rows, ["W", "K", "SV"])
    for col in ["WHIP", "ERA"]:
        pitcher_avg_rows[col] = pd.to_numeric(pitcher_avg_rows[col], errors="coerce").round(3)
    st.dataframe(pitcher_avg_rows, width="stretch", hide_index=True)

    st.markdown("### Pitchers (Reserves)")
    st.caption("Pick players below - no page refresh until Apply Changes is clicked. Custom IP and manual stats update live after applying.")
    with st.form(f"{key_prefix}_pitcher_reserves_form"):
        for idx in range(ROSTER_RESERVE_SLOTS):
            row = state["pitcher_reserves"][idx]
            current_id = _roster_parse_int(row.get("player_id"))
            current_pct = _roster_norm_percentile(
                row.get("percentile"),
                default=state.get("default_percentile", "p50"),
            )
            opt_ids = _reserve_candidate_ids(hitter=False, current_id=current_id)
            p_res_options: list[int | None] = [None, *opt_ids]
            p_res_default_idx = p_res_options.index(current_id) if current_id in p_res_options else 0
            p_res_cols = st.columns([5, 1])
            with p_res_cols[0]:
                st.selectbox(
                    f"Pitcher Reserve {idx + 1}",
                    p_res_options,
                    index=p_res_default_idx,
                    key=f"{key_prefix}_pres_pick_{idx}",
                    format_func=lambda v: _roster_display_pick_label(v, pitcher_labels, empty_label="(empty)"),
                )
            with p_res_cols[1]:
                st.selectbox(
                    f"P{idx + 1} pct",
                    list(ROSTER_PERCENTILE_OPTIONS),
                    index=list(ROSTER_PERCENTILE_OPTIONS).index(current_pct),
                    key=f"{key_prefix}_pres_pct_{idx}",
                    label_visibility="collapsed",
                )
        _p_res_apply = st.form_submit_button("Apply Changes", type="primary")

    if _p_res_apply:
        for idx in range(ROSTER_RESERVE_SLOTS):
            picked_id = _roster_parse_int(st.session_state.get(f"{key_prefix}_pres_pick_{idx}"))
            picked_pct = _roster_norm_percentile(
                st.session_state.get(f"{key_prefix}_pres_pct_{idx}"),
                default=state.get("default_percentile", "p50"),
            )
            old_id = _roster_parse_int(state["pitcher_reserves"][idx].get("player_id"))
            if picked_id != old_id:
                state["pitcher_reserves"][idx]["custom_volume"] = None
                state["pitcher_reserves"][idx]["manual_stats"] = None
            state["pitcher_reserves"][idx]["player_id"] = picked_id
            state["pitcher_reserves"][idx]["percentile"] = picked_pct
        st.rerun()

    # Live controls for committed (applied) pitcher reserve players.
    for idx in range(ROSTER_RESERVE_SLOTS):
        row = state["pitcher_reserves"][idx]
        picked_id = _roster_parse_int(row.get("player_id"))
        if picked_id is None or picked_id not in pitcher_pool_by_id.index:
            continue
        current_pct = _roster_norm_percentile(
            row.get("percentile"),
            default=state.get("default_percentile", "p50"),
        )
        current_custom_ip = _roster_parse_nonneg_float(row.get("custom_volume"))
        current_manual_stats = _roster_normalize_manual_stats(row.get("manual_stats"), hitter=False)
        base_ip = _roster_numeric_from_row(
            pitcher_pool_by_id.loc[int(picked_id)],
            f"IP_proj_{current_pct}",
        )
        player_label = _roster_display_pick_label(picked_id, pitcher_labels, empty_label="")
        _pres_toks = set(pitcher_tokens_by_id.get(picked_id, tuple()))
        _pres_elig = [p for p in ["SP", "RP"] if p in _pres_toks]
        if not _pres_elig:
            _pres_elig = ["SP", "RP"]
        live_cols = st.columns([3, 1, 1, 4])
        with live_cols[0]:
            ip_widget_key = f"{key_prefix}_pres_ip_{idx}_{picked_id}_{current_pct}"
            default_ip_value = current_custom_ip
            if default_ip_value is None:
                default_ip_value = (
                    float(base_ip)
                    if (pd.notna(base_ip) and np.isfinite(float(base_ip)) and float(base_ip) >= 0.0)
                    else 0.0
                )
            custom_ip_input = st.number_input(
                f"P{idx+1} {player_label} IP",
                min_value=0.0,
                value=float(default_ip_value),
                step=0.1,
                key=ip_widget_key,
            )
            parsed_custom_ip = _roster_parse_nonneg_float(custom_ip_input)
            if (
                parsed_custom_ip is not None
                and pd.notna(base_ip)
                and np.isfinite(float(base_ip))
                and abs(float(parsed_custom_ip) - float(base_ip)) <= 1e-6
            ):
                parsed_custom_ip = None
            state["pitcher_reserves"][idx]["custom_volume"] = parsed_custom_ip
        if len(_pres_elig) > 1:
            _pres_ekey = f"{key_prefix}_pelig_pres_{idx}"
            _pres_ecur = st.session_state.get(_pres_ekey)
            if _pres_ecur not in _pres_elig:
                _pres_ecur = _pres_elig[0]
            with live_cols[1]:
                st.selectbox(
                    f"P{idx+1} role",
                    _pres_elig,
                    index=_pres_elig.index(_pres_ecur),
                    key=_pres_ekey,
                    label_visibility="collapsed",
                )
        with live_cols[2]:
            use_manual = st.checkbox(
                f"P{idx+1} manual",
                value=(current_manual_stats is not None),
                key=f"{key_prefix}_pres_manual_toggle_{idx}",
                label_visibility="collapsed",
            )
        if use_manual:
            projected_seed = _pitcher_stats_for_player(
                picked_id,
                pct=current_pct,
                custom_ip=state["pitcher_reserves"][idx]["custom_volume"],
            )
            seed_manual = current_manual_stats or {
                stat: float(projected_seed.get(stat, 0.0))
                if np.isfinite(float(projected_seed.get(stat, float("nan"))))
                else 0.0
                for stat in ROSTER_PITCHER_MANUAL_COUNT_STATS
            }
            with live_cols[3]:
                entered_manual = _render_pitcher_manual_inputs(
                    row_key=f"pres_{idx}",
                    defaults=seed_manual,
                )
            state["pitcher_reserves"][idx]["manual_stats"] = _roster_normalize_manual_stats(
                entered_manual,
                hitter=False,
            )
        else:
            state["pitcher_reserves"][idx]["manual_stats"] = None

    pitcher_reserve_rows: list[dict[str, object]] = []
    for idx, row in enumerate(state.get("pitcher_reserves", [])):
        pid = _roster_parse_int(row.get("player_id"))
        if pid is None:
            continue
        row_pct = _roster_norm_percentile(
            row.get("percentile"),
            default=state.get("default_percentile", "p50"),
        )
        row_custom_ip = _roster_parse_nonneg_float(row.get("custom_volume"))
        row_manual_stats = _roster_pitcher_stats_from_manual(row.get("manual_stats"))
        row_out: dict[str, object] = {
            "Reserve Slot": f"P{idx + 1}",
            "Pct": (
                "MANUAL"
                if (row_manual_stats is not None) or (pid in pitcher_placeholder_ids)
                else str(row_pct).upper()
            ),
            "Player": _roster_display_pick_label(pid, pitcher_labels, empty_label=""),
        }
        for stat in ROSTER_PITCHER_STATS:
            row_out[stat] = float("nan")
        if pid is not None and pid in pitcher_pool_by_id.index:
            stats = row_manual_stats or _pitcher_stats_for_player(
                pid,
                pct=row_pct,
                custom_ip=row_custom_ip,
            )
            for stat in ROSTER_PITCHER_STATS:
                row_out[stat] = stats.get(stat, float("nan"))
        pitcher_reserve_rows.append(row_out)
    pitcher_reserve_df = pd.DataFrame(pitcher_reserve_rows)
    if not pitcher_reserve_df.empty:
        pitcher_reserve_df = _roster_apply_int_display(pitcher_reserve_df, ["W", "K", "SV", "H", "BB", "ER"])
        pitcher_reserve_df["IP"] = pd.to_numeric(pitcher_reserve_df["IP"], errors="coerce").round(1)
        for col in ["WHIP", "ERA"]:
            pitcher_reserve_df[col] = pd.to_numeric(pitcher_reserve_df[col], errors="coerce").round(3)
        st.dataframe(pitcher_reserve_df, width="stretch", hide_index=True)
    else:
        st.caption("No pitcher reserves yet.")

    _hitter_elig_positions = ["C", "1B", "2B", "SS", "3B", "OF", "UT"]
    st.markdown("### Eligibility Counter")
    st.caption("Position picker appears next to each committed starter and reserve above. Counts reflect current selections.")
    elig_counts: dict[str, int] = {pos: 0 for pos in _hitter_elig_positions}
    hitter_elig_total = 0
    for slot in ROSTER_HITTER_STARTER_SLOTS:
        pid = _roster_parse_int(state["starters"].get(slot))
        if pid is not None and pid in hitter_pool_by_id.index:
            _h_toks = set(hitter_tokens_by_id.get(pid, tuple()))
            _h_elig = [p for p in ["C", "1B", "2B", "SS", "3B", "OF"] if p in _h_toks]
            _h_elig.append("UT")
            _ekey = f"{key_prefix}_helig_slot_{slot}"
            natural = _slot_natural_pos(slot)
            default_pos = natural if natural in _h_elig else _h_elig[0]
            chosen = st.session_state.get(_ekey, default_pos)
            if chosen not in _h_elig:
                chosen = default_pos
            elig_counts[chosen] += 1
            hitter_elig_total += 1
    for idx, row in enumerate(state.get("hitter_reserves", [])):
        pid = _roster_parse_int(row.get("player_id"))
        if pid is not None and pid in hitter_pool_by_id.index:
            _h_toks = set(hitter_tokens_by_id.get(pid, tuple()))
            _h_elig = [p for p in ["C", "1B", "2B", "SS", "3B", "OF"] if p in _h_toks]
            _h_elig.append("UT")
            _ekey = f"{key_prefix}_helig_hres_{idx}"
            chosen = st.session_state.get(_ekey, _h_elig[0])
            if chosen not in _h_elig:
                chosen = _h_elig[0]
            elig_counts[chosen] += 1
            hitter_elig_total += 1
    elig_df = pd.DataFrame(
        [{"Position": pos, "Players": int(cnt)} for pos, cnt in [*elig_counts.items(), ("Total", hitter_elig_total)]]
    )
    st.dataframe(elig_df, width="stretch", hide_index=True)

    st.markdown("### Pitcher Eligibility Counter")
    st.caption("Role picker appears next to each committed pitcher starter and reserve above. Counts reflect current selections.")
    pitcher_elig_counts: dict[str, int] = {"SP": 0, "RP": 0}
    pitcher_elig_total = 0
    for slot in ROSTER_PITCHER_STARTER_SLOTS:
        pid = _roster_parse_int(state["starters"].get(slot))
        if pid is not None and pid in pitcher_pool_by_id.index:
            _p_toks = set(pitcher_tokens_by_id.get(pid, tuple()))
            _p_elig = [p for p in ["SP", "RP"] if p in _p_toks]
            if not _p_elig:
                _p_elig = ["SP", "RP"]
            _rekey = f"{key_prefix}_pelig_slot_{slot}"
            natural_role = _slot_natural_pos(slot)
            default_role = natural_role if natural_role in _p_elig else _p_elig[0]
            chosen = st.session_state.get(_rekey, default_role)
            if chosen not in pitcher_elig_counts:
                chosen = default_role
            pitcher_elig_counts[chosen] += 1
            pitcher_elig_total += 1
    for idx, row in enumerate(state.get("pitcher_reserves", [])):
        pid = _roster_parse_int(row.get("player_id"))
        if pid is not None and pid in pitcher_pool_by_id.index:
            _p_toks = set(pitcher_tokens_by_id.get(pid, tuple()))
            _p_elig = [p for p in ["SP", "RP"] if p in _p_toks]
            if not _p_elig:
                _p_elig = ["SP", "RP"]
            _rekey = f"{key_prefix}_pelig_pres_{idx}"
            chosen = st.session_state.get(_rekey, _p_elig[0])
            if chosen not in pitcher_elig_counts:
                chosen = _p_elig[0]
            pitcher_elig_counts[chosen] += 1
            pitcher_elig_total += 1
    pitcher_elig_df = pd.DataFrame(
        [{"Position": pos, "Pitchers": int(cnt)} for pos, cnt in [*pitcher_elig_counts.items(), ("Total", pitcher_elig_total)]]
    )
    st.dataframe(pitcher_elig_df, width="stretch", hide_index=True)

    total_selected = len(_roster_selected_player_ids(state))
    st.caption(
        f"Total selected players: {int(total_selected)} "
        f"(starters={int(sum(1 for v in state.get('starters', {}).values() if _roster_parse_int(v) is not None))}, "
        f"reserves={int(total_selected - sum(1 for v in state.get('starters', {}).values() if _roster_parse_int(v) is not None))})"
    )
    if total_selected < 50:
        st.warning("Current roster has fewer than 50 players. This is informational only.")

    st.markdown("### Save Roster")
    st.download_button(
        "Save current roster JSON",
        data=json.dumps(
            _serialize_roster_state(state),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ),
        file_name=save_file_name,
        mime="application/json",
        key=f"{key_prefix}_download_roster_json",
    )

    st.session_state[state_key] = state


def show_component_slash_predictions(
    historical_df: pd.DataFrame,
    kpi_df: pd.DataFrame,
    *,
    key_prefix: str = "component_slash",
) -> None:
    st.subheader("Component Slash Predictions")
    sub_tabs = st.tabs(["Historical Fit", "From KPI Projections"])
    with sub_tabs[0]:
        _show_component_slash_historical_fit(
            historical_df,
            key_prefix=f"{key_prefix}_historical",
        )
    with sub_tabs[1]:
        _show_component_slash_from_kpi(
            kpi_df,
            key_prefix=f"{key_prefix}_kpi",
        )


def _run_projection_sandbox() -> None:
    st.title("Projection Sandbox")
    st.caption("Unified app: use the sidebar to switch between Projection Sandbox and Damage Interface.")
    compact_ui = st.toggle(
        "Compact UI",
        value=False,
        help="Collapse controls and tighten spacing so more table rows fit on screen.",
    )
    _apply_compact_ui_chrome(compact_ui)

    hitter_source_base_options = {
        "Traditional Two-Stage (KPI Adjusted, p50)": Path(
            "projection_outputs/sandbox/traditional_two_stage_projections_2026.parquet"
        ),
        "KPI Projections (Individual Stat Skills)": Path(
            "projection_outputs/sandbox/kpi_projections_2026.parquet"
        ),
    }
    pitcher_source_base_options = {
        "Two Stage KPI Projections, Pitchers": Path(
            "projection_outputs/sandbox/two_stage_kpi_pitcher_projections_2026.parquet"
        ),
        "Two Stage KPI Projections, Pitchers (3/1/1 Recency)": Path(
            "projection_outputs/sandbox/two_stage_kpi_pitcher_projections_2026_recency_3_1_1.parquet"
        ),
        "KPI Projections (Individual Stat Skills)": Path(
            "projection_outputs/sandbox/pitcher_kpi_projections_2026.parquet"
        ),
        "KPI Projections (Individual Stat Skills, Pitchers 3/1/1 Recency)": Path(
            "projection_outputs/sandbox/pitcher_kpi_projections_2026_recency_3_1_1.parquet"
        ),
    }
    fixed_view_label = "Per 600 PA / Per 180 IP Projections"
    hitter_source_options: dict[str, dict[str, object]] = {
        **{
            label: {"path": path, "fixed_volume": "standard"}
            for label, path in hitter_source_base_options.items()
        },
        fixed_view_label: {
            "path": Path("projection_outputs/sandbox/traditional_two_stage_projections_2026.parquet"),
            "fixed_volume": "per600pa",
        },
    }
    pitcher_source_options: dict[str, dict[str, object]] = {
        **{
            label: {"path": path, "fixed_volume": "standard"}
            for label, path in pitcher_source_base_options.items()
        },
        fixed_view_label: {
            "path": Path("projection_outputs/sandbox/two_stage_kpi_pitcher_projections_2026.parquet"),
            "fixed_volume": "per180ip",
        },
    }
    hitter_source_state_key = "projection_sandbox_hitter_source"
    pitcher_source_state_key = "projection_sandbox_pitcher_source"
    hitter_source_labels = list(hitter_source_options.keys())
    pitcher_source_labels = list(pitcher_source_options.keys())

    if st.session_state.get(hitter_source_state_key) not in hitter_source_options:
        st.session_state[hitter_source_state_key] = hitter_source_labels[0]
    if st.session_state.get(pitcher_source_state_key) not in pitcher_source_options:
        st.session_state[pitcher_source_state_key] = pitcher_source_labels[0]

    selected_hitter_source = str(st.session_state[hitter_source_state_key])
    selected_pitcher_source = str(st.session_state[pitcher_source_state_key])
    selected_hitter_cfg = hitter_source_options[selected_hitter_source]
    selected_pitcher_cfg = pitcher_source_options[selected_pitcher_source]
    hitters_path = Path(str(selected_hitter_cfg.get("path")))
    pitchers_path = Path(str(selected_pitcher_cfg.get("path")))
    hitter_fixed_volume_mode = str(selected_hitter_cfg.get("fixed_volume") or "standard")
    pitcher_fixed_volume_mode = str(selected_pitcher_cfg.get("fixed_volume") or "standard")
    backtest = pd.DataFrame()

    def _source_slug(*parts: str) -> str:
        raw = "_".join(str(p) for p in parts if p)
        out = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_").lower()
        return out or "projection_sandbox"

    source_combo_key = _source_slug(selected_hitter_source, selected_pitcher_source)

    hitters = load_projection(hitters_path)
    hitters = _attach_source_season_mlb_pa(hitters)
    hitters = _attach_hitter_position_counts(hitters)
    if not hitters.empty:
        if "hitter_name" not in hitters.columns and "player_name" in hitters.columns:
            hitters = hitters.assign(hitter_name=hitters["player_name"])
        if "batter_mlbid" not in hitters.columns and "mlbid" in hitters.columns:
            hitters = hitters.assign(batter_mlbid=hitters["mlbid"])
    hitters = _attach_40man_context(hitters, for_pitchers=False)
    hitters = _attach_adp_context(hitters, for_pitchers=False)

    pitchers = load_projection(pitchers_path)
    pitchers = _attach_source_season_mlb_pa(pitchers)
    if not pitchers.empty:
        if "name" not in pitchers.columns and "player_name" in pitchers.columns:
            pitchers = pitchers.assign(name=pitchers["player_name"])
        if "pitcher_mlbid" not in pitchers.columns and "mlbid" in pitchers.columns:
            pitchers = pitchers.assign(pitcher_mlbid=pitchers["mlbid"])
    pitchers = _attach_40man_context(pitchers, for_pitchers=True)
    pitchers = _attach_adp_context(pitchers, for_pitchers=True)

    show_before_after = (
        hitters_path
        == Path("projection_outputs/sandbox/traditional_two_stage_projections_2026.parquet")
    ) and (hitter_fixed_volume_mode == "standard")
    base_traditional = (
        load_projection(Path("projection_outputs/sandbox/traditional_projections_2026.parquet"))
        if show_before_after
        else pd.DataFrame()
    )
    base_traditional = _attach_source_season_mlb_pa(base_traditional)
    base_traditional = _attach_hitter_position_counts(base_traditional)

    component_slash_hist = load_projection(COMPONENT_SLASH_PREDICTIONS_PATH)
    component_slash_kpi = load_projection(COMPONENT_SLASH_KPI_PROJECTIONS_PATH)
    component_slash_kpi = _attach_hitter_position_counts(component_slash_kpi)

    st.caption(
        "Auto-loaded projection sources: "
        f"Hitters=`{hitters_path}` ({hitter_fixed_volume_mode}) | "
        f"Pitchers=`{pitchers_path}` ({pitcher_fixed_volume_mode})"
    )

    enable_hitter_role_edit_mode = st.checkbox(
        "Enable edit mode (hitter Opening Day role overrides)",
        value=False,
        key=f"{source_combo_key}_enable_hitter_role_edit_mode",
        help=(
            "Lets you override Opening Day role for individual hitters in this session. "
            "Overrides feed into role-based PA blending and hitter role-priority allocators."
        ),
    )
    if bool(enable_hitter_role_edit_mode):
        with st.expander("Edit mode: hitter Opening Day roles", expanded=False):
            hitters, hitter_edit_summary = _apply_hitter_role_overrides_edit_mode(
                hitters,
                key_prefix=source_combo_key,
            )
        st.caption(
            "Hitter edit mode overrides: "
            f"active={hitter_edit_summary.get('overrides_active', 0)}, "
            f"rows impacted={hitter_edit_summary.get('rows_overridden', 0)}"
        )

    enable_pitcher_role_edit_mode = st.checkbox(
        "Enable edit mode (pitcher Opening Day role overrides)",
        value=False,
        key=f"{source_combo_key}_enable_pitcher_role_edit_mode",
        help=(
            "Lets you override Opening Day role for individual pitchers in this session. "
            "Overrides feed into role-based IP blending and saves redistribution."
        ),
    )
    if bool(enable_pitcher_role_edit_mode):
        with st.expander("Edit mode: pitcher Opening Day roles", expanded=False):
            pitchers, edit_summary = _apply_pitcher_role_overrides_edit_mode(
                pitchers,
                key_prefix=source_combo_key,
            )
        st.caption(
            "Edit mode overrides: "
            f"active={edit_summary.get('overrides_active', 0)}, "
            f"rows impacted={edit_summary.get('rows_overridden', 0)}"
        )

    role_apply_cols = st.columns(5)
    with role_apply_cols[0]:
        apply_role_pt = st.checkbox(
            "Apply 40-man role PA/IP blend",
            value=True,
            key=f"{source_combo_key}_apply_40man_role_pt",
            help=(
                "For each Opening Day Status role, computes mean projected PA/IP and "
                "replaces each player p50 volume with the average of player p50 and role mean."
            ),
        )
    with role_apply_cols[1]:
        apply_sv_redistribution = st.checkbox(
            "Redistribute saves by bullpen role",
            value=True,
            key=f"{source_combo_key}_apply_40man_sv_redist",
            help=(
                "Sets team saves from 2023-2025 mean/stdev inputs "
                f"(mean={TEAM_SV_MEAN:.1f}, sd={TEAM_SV_STD:.1f}) and redistributes "
                "to Closer, Setup Man, and Middle Reliever pitchers using existing SV rank order."
            ),
        )
    with role_apply_cols[2]:
        apply_team_ip_source = st.checkbox(
            "Apply MLB IP source allocator",
            value=True,
            key=f"{source_combo_key}_apply_mlb_ip_source_allocator",
            help=(
                "Allocates MLB pitching volume by Opening Day role hierarchy with team caps "
                "(IP clipped to 1400-1470, GS=162; G unconstrained), and flags source rows."
            ),
        )
    with role_apply_cols[3]:
        apply_hitter_pa_source = st.checkbox(
            "Apply MLB PA source allocator",
            value=True,
            key=f"{source_combo_key}_apply_mlb_pa_source_allocator",
            help=(
                "Allocates MLB hitter PA by Opening Day role hierarchy with team caps "
                f"(PA clipped to {int(TEAM_PA_MIN)}-{int(TEAM_PA_MAX)}), and flags source rows."
            ),
        )
    with role_apply_cols[4]:
        apply_hitter_sb_source = st.checkbox(
            "Apply MLB SB/SBA source allocator",
            value=True,
            key=f"{source_combo_key}_apply_mlb_sb_sba_source_allocator",
            help=(
                "Reallocates hitter SB/SBA projections by Opening Day Status priority so "
                f"league totals match 2025 MLB targets (SB={int(MLB_SB_TARGET_2025)}, "
                f"SBA={int(MLB_SBA_TARGET_2025)})."
            ),
        )

    if bool(apply_role_pt):
        hitters, hit_summary = _apply_role_playing_time_adjustments(hitters, for_pitchers=False)
        pitchers, pit_summary = _apply_role_playing_time_adjustments(pitchers, for_pitchers=True)
        st.caption(
            "40-man role blend applied: "
            f"hitters adjusted={hit_summary.get('rows_adjusted', 0)} "
            f"across roles={hit_summary.get('roles_used', 0)} | "
            f"pitchers adjusted={pit_summary.get('rows_adjusted', 0)} "
            f"across roles={pit_summary.get('roles_used', 0)}"
        )
    if bool(apply_sv_redistribution):
        pitchers, sv_summary = _redistribute_team_saves_by_role(pitchers)
        st.caption(
            "40-man saves redistribution applied: "
            f"teams adjusted={sv_summary.get('teams_adjusted', 0)}, "
            f"pitchers receiving SV={sv_summary.get('players_receiving_sv', 0)}"
        )
    pitchers, il_cut_summary = _apply_projected_il_volume_cut(
        pitchers,
        cut_share=0.33,
    )
    if int(il_cut_summary.get("rows_adjusted", 0)) > 0:
        st.caption(
            "Projected IL volume cut applied: "
            f"rows adjusted={il_cut_summary.get('rows_adjusted', 0)}, "
            f"cut={100.0 * float(il_cut_summary.get('cut_share', 0.33)):.0f}%"
        )
    if bool(apply_team_ip_source):
        pitchers, ip_src_summary = _apply_team_mlb_ip_source_allocation(pitchers)
        ip_mu = ip_src_summary.get("team_ip_mean")
        ip_sd = ip_src_summary.get("team_ip_sd")
        g_mu = ip_src_summary.get("team_g_mean")
        gs_mu = ip_src_summary.get("team_gs_mean")
        ip_mu_txt = "n/a" if (ip_mu is None or not np.isfinite(float(ip_mu))) else f"{float(ip_mu):.1f}"
        ip_sd_txt = "n/a" if (ip_sd is None or not np.isfinite(float(ip_sd))) else f"{float(ip_sd):.1f}"
        g_mu_txt = "n/a" if (g_mu is None or not np.isfinite(float(g_mu))) else f"{float(g_mu):.1f}"
        gs_mu_txt = "n/a" if (gs_mu is None or not np.isfinite(float(gs_mu))) else f"{float(gs_mu):.1f}"
        st.caption(
            "MLB IP source allocator applied: "
            f"teams adjusted={ip_src_summary.get('teams_adjusted', 0)}, "
            f"rows selected={ip_src_summary.get('rows_selected', 0)}, "
            f"team IP mean={ip_mu_txt}, sd={ip_sd_txt}, "
            f"team G mean={g_mu_txt}, team GS mean={gs_mu_txt}"
        )
    if bool(apply_hitter_pa_source):
        hitters, pa_src_summary = _apply_team_mlb_pa_source_allocation(hitters)
        pa_mu = pa_src_summary.get("team_pa_mean")
        pa_sd = pa_src_summary.get("team_pa_sd")
        pa_min = pa_src_summary.get("team_pa_min")
        pa_max = pa_src_summary.get("team_pa_max")
        pa_mu_txt = "n/a" if (pa_mu is None or not np.isfinite(float(pa_mu))) else f"{float(pa_mu):.1f}"
        pa_sd_txt = "n/a" if (pa_sd is None or not np.isfinite(float(pa_sd))) else f"{float(pa_sd):.1f}"
        pa_min_txt = "n/a" if (pa_min is None or not np.isfinite(float(pa_min))) else f"{float(pa_min):.1f}"
        pa_max_txt = "n/a" if (pa_max is None or not np.isfinite(float(pa_max))) else f"{float(pa_max):.1f}"
        st.caption(
            "MLB PA source allocator applied: "
            f"teams adjusted={pa_src_summary.get('teams_adjusted', 0)}, "
            f"rows selected={pa_src_summary.get('rows_selected', 0)}, "
            f"team PA mean={pa_mu_txt}, sd={pa_sd_txt}, "
            f"min={pa_min_txt}, max={pa_max_txt}"
        )
    if bool(apply_hitter_sb_source):
        hitters, sb_src_summary = _apply_hitter_mlb_sb_sba_source_allocation(
            hitters,
            target_sb_total=float(MLB_SB_TARGET_2025),
            target_sba_total=float(MLB_SBA_TARGET_2025),
        )
        sb_tot = sb_src_summary.get("sb_total_p50")
        sba_tot = sb_src_summary.get("sba_total_p50")
        sb_txt = "n/a" if (sb_tot is None or not np.isfinite(float(sb_tot))) else f"{float(sb_tot):.0f}"
        sba_txt = "n/a" if (sba_tot is None or not np.isfinite(float(sba_tot))) else f"{float(sba_tot):.0f}"
        st.caption(
            "MLB SB/SBA source allocator applied: "
            f"rows selected={sb_src_summary.get('rows_selected', 0)}, "
            f"SB p50 total={sb_txt}, SBA p50 total={sba_txt}"
        )

    if hitter_fixed_volume_mode.startswith("per600pa"):
        hitters = _apply_fixed_volume_projection_view(
            hitters,
            target_pa=600.0,
        )
        st.caption("Hitter source mode: fixed 600 PA across projection percentiles.")
    if pitcher_fixed_volume_mode.startswith("per180ip"):
        pitchers = _apply_fixed_volume_projection_view(
            pitchers,
            target_ip=180.0,
        )
        st.caption("Pitcher source mode: fixed 180 IP across projection percentiles.")

    hide_cols_hitters: set[str] = set()
    if hitters_path == Path("projection_outputs/sandbox/traditional_two_stage_projections_2026.parquet"):
        hide_cols_hitters = {"age_source", "source_season"}
    hide_cols_pitchers: set[str] = set()
    prefer_kpi_hitters = hitters_path == Path("projection_outputs/sandbox/kpi_projections_2026.parquet")
    prefer_kpi_pitchers = pitchers_path in {
        Path("projection_outputs/sandbox/pitcher_kpi_projections_2026.parquet"),
        Path("projection_outputs/sandbox/pitcher_kpi_projections_2026_recency_3_1_1.parquet"),
    }

    tab_labels = ["Hitters", "Pitchers", "All Players", "Backtest"]
    if show_before_after:
        tab_labels.append("Before vs After")
    tab_labels.append("Component Slash")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.selectbox(
            "Hitters projection source",
            hitter_source_labels,
            key=hitter_source_state_key,
            help="Choose base source or the fixed-volume projections view.",
        )
        show_table(
            hitters,
            "Hitters",
            "hitter_name",
            "batter_mlbid",
            key_prefix=f"{source_combo_key}_hitters",
            hide_display_cols=hide_cols_hitters,
            prefer_kpi_defaults=prefer_kpi_hitters,
            compact_ui=compact_ui,
        )
    with tabs[1]:
        st.selectbox(
            "Pitchers projection source",
            pitcher_source_labels,
            key=pitcher_source_state_key,
            help="Choose base source or the fixed-volume projections view.",
        )
        show_table(
            pitchers,
            "Pitchers",
            "name",
            "pitcher_mlbid",
            key_prefix=f"{source_combo_key}_pitchers",
            hide_display_cols=hide_cols_pitchers,
            prefer_kpi_defaults=prefer_kpi_pitchers,
            compact_ui=compact_ui,
        )
    with tabs[2]:
        _show_all_players_adp_points_view(
            hitters,
            pitchers,
            key_prefix=f"{source_combo_key}_all_players",
            compact_ui=compact_ui,
        )
    with tabs[3]:
        st.subheader("Backtest Summary")
        if backtest.empty:
            st.info("Missing projection_backtest_summary.parquet")
        else:
            backtest_df = backtest.copy()
            backtest_df = _apply_column_filters(
                backtest_df,
                key_prefix=f"{source_combo_key}_backtest_col_filters",
                expander_label="Backtest column filters",
            )
            if backtest_df.empty:
                st.info("No rows after column filters. Clear one or more filters to view data.")
            else:
                backtest_df = _apply_custom_rounding(backtest_df)
                st.dataframe(backtest_df, width="stretch", hide_index=True)
    comp_tab_idx = 4
    if show_before_after:
        with tabs[4]:
            _show_two_stage_before_after(
                hitters,
                base_traditional,
                key_prefix=f"{source_combo_key}_before_after",
            )
        comp_tab_idx = 5
    with tabs[comp_tab_idx]:
        show_component_slash_predictions(
            component_slash_hist,
            component_slash_kpi,
            key_prefix=f"{source_combo_key}_component_slash",
        )



def _run_roster_manager_interface() -> None:
    st.title("Roster Manager")
    st.caption("Standalone roster workflow (sidebar Application menu).")

    roster_hitters = load_projection(
        Path("projection_outputs/sandbox/traditional_two_stage_projections_2026.parquet")
    )
    roster_pitchers = load_projection(
        Path("projection_outputs/sandbox/two_stage_kpi_pitcher_projections_2026.parquet")
    )
    roster_hitters = _attach_source_season_mlb_pa(roster_hitters)
    roster_hitters = _attach_hitter_position_counts(roster_hitters)
    roster_pitchers = _attach_source_season_mlb_pa(roster_pitchers)

    if not roster_hitters.empty:
        if "hitter_name" not in roster_hitters.columns and "player_name" in roster_hitters.columns:
            roster_hitters = roster_hitters.assign(hitter_name=roster_hitters["player_name"])
        if "batter_mlbid" not in roster_hitters.columns and "mlbid" in roster_hitters.columns:
            roster_hitters = roster_hitters.assign(batter_mlbid=roster_hitters["mlbid"])
    if not roster_pitchers.empty:
        if "name" not in roster_pitchers.columns and "player_name" in roster_pitchers.columns:
            roster_pitchers = roster_pitchers.assign(name=roster_pitchers["player_name"])
        if "pitcher_mlbid" not in roster_pitchers.columns and "mlbid" in roster_pitchers.columns:
            roster_pitchers = roster_pitchers.assign(pitcher_mlbid=roster_pitchers["mlbid"])

    roster_hitters = _attach_40man_context(roster_hitters, for_pitchers=False)
    roster_pitchers = _attach_40man_context(roster_pitchers, for_pitchers=True)

    # Keep roster values aligned with Projection Sandbox default transforms.
    roster_hitters, _ = _apply_role_playing_time_adjustments(
        roster_hitters,
        for_pitchers=False,
    )
    roster_pitchers, _ = _apply_role_playing_time_adjustments(
        roster_pitchers,
        for_pitchers=True,
    )
    roster_pitchers, _ = _redistribute_team_saves_by_role(roster_pitchers)
    roster_pitchers, _ = _apply_projected_il_volume_cut(
        roster_pitchers,
        cut_share=0.33,
    )
    roster_pitchers, _ = _apply_team_mlb_ip_source_allocation(roster_pitchers)
    roster_hitters, _ = _apply_team_mlb_pa_source_allocation(roster_hitters)
    roster_hitters, _ = _apply_hitter_mlb_sb_sba_source_allocation(
        roster_hitters,
        target_sb_total=float(MLB_SB_TARGET_2025),
        target_sba_total=float(MLB_SBA_TARGET_2025),
    )
    st.caption(
        "Applied default Projection Sandbox adjustments: role blend, saves redistribution, "
        "projected IL volume cut (33%), MLB IP source allocation, MLB PA source allocation, "
        "and MLB SB/SBA source allocation."
    )

    show_roster_manager(
        roster_hitters,
        roster_pitchers,
        key_prefix="sidebar_roster_manager",
    )


def _run_damage_interface() -> None:
    damage_app_path = Path(__file__).with_name("damage_streamlit.py")
    if not damage_app_path.exists():
        st.error(f"Missing damage app script: {damage_app_path}")
        return

    prev_embed = os.environ.get("PY_PROJECTIONS_EMBED_DAMAGE")
    os.environ["PY_PROJECTIONS_EMBED_DAMAGE"] = "1"
    try:
        runpy.run_path(str(damage_app_path), run_name="__pyprojections_damage__")
    finally:
        if prev_embed is None:
            os.environ.pop("PY_PROJECTIONS_EMBED_DAMAGE", None)
        else:
            os.environ["PY_PROJECTIONS_EMBED_DAMAGE"] = prev_embed


def main() -> None:
    st.set_page_config(page_title="Projection + Damage", layout="wide")
    app_mode = st.sidebar.radio(
        "Application",
        options=["Projection Sandbox", "Roster Manager", "Damage Interface"],
        index=0,
        key="py_projections_app_mode",
    )
    if app_mode == "Roster Manager":
        _run_roster_manager_interface()
        return
    if app_mode == "Damage Interface":
        _run_damage_interface()
        return
    _run_projection_sandbox()


if __name__ == "__main__":
    main()

