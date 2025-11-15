#!/usr/bin/env python
# coding: utf-8

# # FDS Challenge
# 
# This notebook will guide you through the first steps of the competition. Our goal here is to show you how to:
# 
# 1.  Load the `train.jsonl` and `test.jsonl` files from the competition data.
# 2.  Create a very simple set of features from the data.
# 3.  Train a basic model.
# 4.  Generate a `submission.csv` file in the correct format.
# 5.  Submit your results.
# 
# Let's get started!
# =========================
#       GLOBAL IMPORTS
# =========================

import os
import json
import numpy as np
import pandas as pd

from IPython.display import display
from tqdm import tqdm

from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

# Scikit-learn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# XGBoost
import xgboost as xgb

# # 1. Loading and Inspecting the Data

# In[ ]:



# # 2. Features Engineering

# In[2]:


import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------
# Base stat keys used throughout the feature extraction
# ---------------------------------------------
BASE_STAT_KEYS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]


# ---------------------------------------------
# Static team composition and stats
# ---------------------------------------------
def unique_types(team: List[Dict[str, Any]]) -> int:
    collected = []
    for p in team or []:
        ts = p.get("types") or []
        if isinstance(ts, str):
            ts = [ts]
        collected.extend([t for t in ts if t])
    return len(set(collected))


def sum_stats_of_team(team: List[Dict[str, Any]]) -> float:
    total = 0.0
    for p in team or []:
        for k in BASE_STAT_KEYS:
            v = p.get(k)
            if isinstance(v, (int, float)):
                total += float(v)
    return total


def avg_stats_of_team(team: List[Dict[str, Any]]) -> float:
    if not team:
        return 0.0
    per = []
    for p in team:
        vals = [p.get(k) for k in BASE_STAT_KEYS if isinstance(p.get(k), (int, float))]
        if vals:
            per.append(sum(vals) / len(vals))
    return float(sum(per) / len(per)) if per else 0.0


def sum_and_avg_of_single(poke: dict) -> Tuple[float, float]:
    vals = [poke.get(k) for k in BASE_STAT_KEYS if isinstance(poke.get(k), (int, float))]
    if not vals:
        return 0.0, 0.0
    total = float(sum(vals))
    return total, total / len(vals)


def team_stat_variance(team: List[Dict[str, Any]]) -> float:
    if not team:
        return 0.0
    per = []
    for p in team:
        vals = [p.get(k) for k in BASE_STAT_KEYS if isinstance(p.get(k), (int, float))]
        if vals:
            per.append(sum(vals) / len(vals))
    if len(per) < 2:
        return 0.0
    return float(pd.Series(per).var())


def _team_speed_stats(team):
    """Return mean and max base speed over a team."""
    sp = [p.get("base_spe", 0.0) for p in team or [] if isinstance(p.get("base_spe", None), (int, float))]
    if not sp:
        return 0.0, 0.0
    return float(np.mean(sp)), float(np.max(sp))


# ---------------------------------------------
# General numeric helpers
# ---------------------------------------------
def _safe_mean(arr):
    return float(np.mean(arr)) if arr else 0.0


def _safe_ratio(a, b, cap=10.0):
    r = a / (b + 1e-6)
    if r < 0:
        r = 0.0
    if r > cap:
        r = cap
    if not np.isfinite(r):
        r = 0.0
    return float(r)


# ---------------------------------------------
# Timeline-based HP feature extraction
# ---------------------------------------------
def get_timeline(r: Dict[str, Any], max_turns: int = 30):
    tl = r.get("battle_timeline", []) or []
    return tl[:max_turns] if isinstance(tl, list) else []


def _extract_hp_series(tl):
    p1 = []
    p2 = []
    for t in tl:
        if not isinstance(t, dict):
            continue
        s1 = t.get("p1_pokemon_state") or {}
        s2 = t.get("p2_pokemon_state") or {}
        v1 = s1.get("hp_pct")
        v2 = s2.get("hp_pct")
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            p1.append(float(v1))
            p2.append(float(v2))
    return p1, p2


def _mean_last_std_min(arr):
    if not arr:
        return 0.0, 0.0, 0.0, 0.0
    x = np.array(arr, dtype=float)
    return float(x.mean()), float(x[-1]), float(x.std(ddof=0)), float(x.min())


def _window(arr, n):
    return arr[:n] if arr else []


def _frac_positive(arr):
    return float((np.array(arr) > 0).mean()) if arr else 0.0


def _slope(arr):
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr))
    m, _ = np.polyfit(x, np.array(arr), 1)
    return float(m)


def _auc_pct(arr):
    return float(np.sum(arr) / (100.0 * len(arr))) if arr else 0.0


def _status_count(tl, who):
    cnt = 0
    k = f"{who}_pokemon_state"
    for t in tl:
        if not isinstance(t, dict):
            continue
        st = (t.get(k) or {}).get("status", None)
        if st not in (None, "", "none", "NONE"):
            cnt += 1
    return float(cnt)


def _ko_count(arr):
    return float(sum(1 for v in arr if v == 0))


# ---------------------------------------------
# Move-related statistics from timeline
# ---------------------------------------------
def _move_stats_for_side(tl, who, window=None):
    key = f"{who}_move_details"
    seq = tl if window is None else tl[:window]
    pw, ac, pr = [], [], []
    for t in seq:
        if not isinstance(t, dict):
            continue
        md = t.get(key) or {}
        bp = md.get("base_power")
        acc = md.get("accuracy")
        pri = md.get("priority")
        if isinstance(bp, (int, float)):
            pw.append(float(bp))
        if isinstance(acc, (int, float)):
            ac.append(float(acc))
        if isinstance(pri, (int, float)):
            pr.append(float(pri))
    suf = "" if window is None else f"_{window}"
    return {
        f"mv_{who}_power_mean{suf}": _safe_mean(pw),
        f"mv_{who}_acc_mean{suf}": _safe_mean(ac),
        f"mv_{who}_priority_mean{suf}": _safe_mean(pr),
    }


# ---------------------------------------------
# Type effectiveness helpers (uppercase canonical)
# ---------------------------------------------
_TYPE_CHART = {
    "NORMAL": {"ROCK": 0.5, "GHOST": 0.0, "STEEL": 0.5},
    "FIRE": {
        "FIRE": 0.5,
        "WATER": 0.5,
        "GRASS": 2.0,
        "ICE": 2.0,
        "BUG": 2.0,
        "ROCK": 0.5,
        "DRAGON": 0.5,
        "STEEL": 2.0,
    },
    "WATER": {"FIRE": 2.0, "WATER": 0.5, "GRASS": 0.5, "GROUND": 2.0, "ROCK": 2.0, "DRAGON": 0.5},
    "ELECTRIC": {
        "WATER": 2.0,
        "ELECTRIC": 0.5,
        "GRASS": 0.5,
        "GROUND": 0.0,
        "FLYING": 2.0,
        "DRAGON": 0.5,
    },
    "GRASS": {
        "FIRE": 0.5,
        "WATER": 2.0,
        "GRASS": 0.5,
        "POISON": 0.5,
        "GROUND": 2.0,
        "FLYING": 0.5,
        "BUG": 0.5,
        "ROCK": 2.0,
        "DRAGON": 0.5,
        "STEEL": 0.5,
    },
    "ICE": {
        "FIRE": 0.5,
        "WATER": 0.5,
        "GRASS": 2.0,
        "GROUND": 2.0,
        "FLYING": 2.0,
        "DRAGON": 2.0,
        "STEEL": 0.5,
    },
    "FIGHTING": {
        "NORMAL": 2.0,
        "ICE": 2.0,
        "POISON": 0.5,
        "FLYING": 0.5,
        "PSYCHIC": 0.5,
        "BUG": 0.5,
        "ROCK": 2.0,
        "GHOST": 0.0,
        "DARK": 2.0,
        "STEEL": 2.0,
        "FAIRY": 0.5,
    },
    "POISON": {
        "GRASS": 2.0,
        "POISON": 0.5,
        "GROUND": 0.5,
        "ROCK": 0.5,
        "GHOST": 0.5,
        "STEEL": 0.0,
        "FAIRY": 2.0,
    },
    "GROUND": {
        "FIRE": 2.0,
        "ELECTRIC": 2.0,
        "GRASS": 0.5,
        "POISON": 2.0,
        "FLYING": 0.0,
        "BUG": 0.5,
        "ROCK": 2.0,
        "STEEL": 2.0,
    },
    "FLYING": {
        "ELECTRIC": 0.5,
        "GRASS": 2.0,
        "FIGHTING": 2.0,
        "BUG": 2.0,
        "ROCK": 0.5,
        "STEEL": 0.5,
    },
    "PSYCHIC": {
        "FIGHTING": 2.0,
        "POISON": 2.0,
        "PSYCHIC": 0.5,
        "DARK": 0.0,
        "STEEL": 0.5,
    },
    "BUG": {
        "FIRE": 0.5,
        "GRASS": 2.0,
        "FIGHTING": 0.5,
        "POISON": 0.5,
        "FLYING": 0.5,
        "PSYCHIC": 2.0,
        "GHOST": 0.5,
        "DARK": 2.0,
        "STEEL": 0.5,
        "FAIRY": 0.5,
    },
    "ROCK": {
        "FIRE": 2.0,
        "ICE": 2.0,
        "FIGHTING": 0.5,
        "GROUND": 0.5,
        "FLYING": 2.0,
        "BUG": 2.0,
        "STEEL": 0.5,
    },
    "GHOST": {"NORMAL": 0.0, "PSYCHIC": 2.0, "GHOST": 2.0, "DARK": 0.5},
    "DRAGON": {"DRAGON": 2.0, "STEEL": 0.5, "FAIRY": 0.0},
    "DARK": {"FIGHTING": 0.5, "PSYCHIC": 2.0, "GHOST": 2.0, "DARK": 0.5, "FAIRY": 0.5},
    "STEEL": {
        "FIRE": 0.5,
        "WATER": 0.5,
        "ELECTRIC": 0.5,
        "ICE": 2.0,
        "ROCK": 2.0,
        "FAIRY": 2.0,
        "STEEL": 0.5,
    },
    "FAIRY": {
        "FIRE": 0.5,
        "FIGHTING": 2.0,
        "POISON": 0.5,
        "DRAGON": 2.0,
        "DARK": 2.0,
        "STEEL": 0.5,
    },
}


def _type_multiplier(move_type: str, target_types: List[str] | set) -> float:
    """Effectiveness multiplier for move_type against mono/dual target types."""
    if not move_type:
        return 1.0
    mt = move_type.strip().upper()
    mult = 1.0
    for tt in target_types or []:
        tt_up = str(tt).strip().upper()
        mult *= _TYPE_CHART.get(mt, {}).get(tt_up, 1.0)
    return float(mult) if np.isfinite(mult) else 1.0


def _avg_type_eff_p1_vs_p2lead(tl: list[dict], p2_lead_types: List[str] | set, window: int | None = None) -> float:
    """Mean effectiveness of P1 used moves vs P2 lead types over full/early window."""
    seq = tl if window is None else tl[:window]
    vals = []
    for t in seq:
        if not isinstance(t, dict):
            continue
        md = t.get("p1_move_details") or {}
        mv_t = md.get("type")
        if isinstance(mv_t, str) and p2_lead_types:
            vals.append(_type_multiplier(mv_t, p2_lead_types))
    return float(np.mean(vals)) if vals else 1.0  # neutral if unknown


# ---------------------------------------------
# STAB features (Same-Type Attack Bonus)
# ---------------------------------------------
def _name_to_types_map_p1(record: Dict[str, Any]) -> Dict[str, set]:
    mp = {}
    for p in record.get("p1_team_details", []) or []:
        nm = (p.get("name") or "").strip().lower()
        ts = p.get("types") or []
        if isinstance(ts, str):
            ts = [ts]
        ts_norm = {str(t).strip().upper() for t in ts if t and str(t).strip().upper() != "NOTYPE"}
        if nm:
            mp[nm] = ts_norm
    return mp


def _active_name_and_move_type(turn: Dict[str, Any], who: str) -> tuple[str, str]:
    state = turn.get(f"{who}_pokemon_state") or {}
    md = turn.get(f"{who}_move_details") or {}
    nm = (state.get("name") or "").strip().lower()
    mv_t = (md.get("type") or "").strip().upper()
    return nm, mv_t


def _stab_features(record: Dict[str, Any], max_turns: int = 30) -> Dict[str, float]:
    tl = get_timeline(record, max_turns=max_turns)

    # type maps of P1 (name -> set(types))
    p1_types_map = _name_to_types_map_p1(record)

    def _accumulate(seq):
        p1_total = p1_stab = 0
        p2_total = p2_stab = 0  # kept for symmetry / future use

        for t in seq:
            # P1
            nm1, mv1_type = _active_name_and_move_type(t, "p1")
            if mv1_type:
                p1_total += 1
                types1 = p1_types_map.get(nm1, set())
                is_stab = (mv1_type in types1) if types1 else False
                if is_stab:
                    p1_stab += 1

        p1_ratio = (p1_stab / p1_total) if p1_total > 0 else 0.0
        p2_ratio = 0.0

        return {
            "stab_stab_ratio_diff": float(p1_ratio - p2_ratio),
            "stab_stab_ratio_ratio": _safe_ratio(p1_ratio, p2_ratio if p2_ratio > 0 else 1e-6, cap=10.0),
        }

    full = _accumulate(tl)
    w5 = _accumulate(tl[:5])

    return {
        "stab_stab_ratio_diff_full": float(full["stab_stab_ratio_diff"]),
        "stab_stab_ratio_ratio_full": float(full["stab_stab_ratio_ratio"]),
        "stab_stab_ratio_diff_w5": float(w5["stab_stab_ratio_diff"]),
        "stab_stab_ratio_ratio_w5": float(w5["stab_stab_ratio_ratio"]),
    }


# ---------------------------------------------
# Early momentum (first 3 turns)
# ---------------------------------------------
def _first_ko_flag(hp_series: list[float]) -> int:
    for v in hp_series:
        if isinstance(v, (int, float)) and float(v) == 0.0:
            return 1
    return 0


def _first_status_advantage(tl: list[dict], first_n: int = 3) -> float:
    p1 = p2 = 0
    for t in tl[:first_n]:
        s1 = (t.get("p1_pokemon_state") or {}).get("status", None)
        s2 = (t.get("p2_pokemon_state") or {}).get("status", None)
        if s1 not in (None, "", "none", "NONE"):
            p1 += 1
        if s2 not in (None, "", "none", "NONE"):
            p2 += 1
    return float(p1 - p2)


def _early_momentum_features(record: Dict[str, Any], first_n: int = 3) -> Dict[str, float]:
    tl = get_timeline(record, max_turns=30)
    p1, p2 = _extract_hp_series(tl)
    p1w, p2w = _window(p1, first_n), _window(p2, first_n)

    diffw = [a - b for a, b in zip(p1w, p2w)] if p1w and p2w and len(p1w) == len(p2w) else []
    mean_diff_first = float(np.mean(diffw)) if diffw else 0.0

    p1_first_ko = _first_ko_flag(p2w)
    p2_first_ko = _first_ko_flag(p1w)
    first_ko_score = float(p1_first_ko - p2_first_ko)

    status_adv = _first_status_advantage(tl, first_n=first_n)

    return {
        f"early_hp_diff_mean_{first_n}": mean_diff_first,
        f"early_first_ko_score_{first_n}": first_ko_score,
        f"early_status_advantage_{first_n}": status_adv,
    }


# ---------------------------------------------
# Priority counts and advantage (full / 5)
# ---------------------------------------------
def _priority_counts(record: Dict[str, Any], max_turns: int = 30, window: int | None = None) -> Dict[str, float]:
    tl = get_timeline(record, max_turns=max_turns)
    turns = tl if window is None else tl[:window]

    p1_count = 0.0
    p2_count = 0.0
    for t in turns:
        if not isinstance(t, dict):
            continue
        md1 = t.get("p1_move_details") or {}
        md2 = t.get("p2_move_details") or {}
        pri1 = md1.get("priority")
        pri2 = md2.get("priority")
        if isinstance(pri1, (int, float)) and float(pri1) > 0:
            p1_count += 1.0
        if isinstance(pri2, (int, float)) and float(pri2) > 0:
            p2_count += 1.0

    suf = "" if window is None else f"_{window}"
    return {
        f"mv_p1_priority_count{suf}": p1_count,
        f"mv_p2_priority_count{suf}": p2_count,
        f"mv_priority_count_diff{suf}": p1_count - p2_count,
    }


def _priority_feature_block(record: Dict[str, Any]) -> Dict[str, float]:
    f = {}
    f.update(_priority_counts(record, max_turns=30, window=None))
    f.update(_priority_counts(record, max_turns=30, window=5))
    return f


# ====================================
# LEAD MATCHUP / DAMAGE-INDEX HELPERS
# ====================================
def _simple_damage_index(base_power: float, stab: bool, eff: float, atk_proxy: float, def_proxy: float) -> float:
    if not isinstance(base_power, (int, float)) or base_power <= 0:
        return 0.0
    s = 1.5 if stab else 1.0
    ratio = (float(atk_proxy) + 1e-3) / (float(def_proxy) + 1e-3)
    val = float(base_power) * s * float(eff) * ratio
    return float(val) if np.isfinite(val) else 0.0


def _p1_vs_p2lead_matchup_index(record: dict, tl: list[dict]) -> dict:
    p1_team = record.get("p1_team_details", []) or []
    p1_mean_atk = float(np.mean([p.get("base_atk", 0) for p in p1_team])) if p1_team else 0.0
    p1_mean_spa = float(np.mean([p.get("base_spa", 0) for p in p1_team])) if p1_team else 0.0

    lead = record.get("p2_lead_details") or {}
    p2_types = lead.get("types") or []
    if isinstance(p2_types, str):
        p2_types = [p2_types]
    p2_types = [t for t in p2_types if t]
    p2_def = float(lead.get("base_def", 0.0) or 0.0)
    p2_spd = float(lead.get("base_spd", 0.0) or 0.0)

    p1map = {}
    for p in p1_team:
        nm = (p.get("name") or "").strip().lower()
        ts = p.get("types") or []
        if isinstance(ts, str):
            ts = [ts]
        p1map[nm] = {str(x).strip().upper() for x in ts if x}

    def _acc(window=None):
        seq = tl if window is None else tl[:window]
        vals = []
        for t in seq:
            if not isinstance(t, dict):
                continue
            md = t.get("p1_move_details") or {}
            bp = md.get("base_power")
            cat = md.get("category")
            mtype = md.get("type")
            if not isinstance(bp, (int, float)) or bp <= 0:
                continue
            nm = (t.get("p1_pokemon_state") or {}).get("name", "")
            nm = (nm or "").strip().lower()
            is_stab = str(mtype or "").strip().upper() in p1map.get(nm, set())
            eff = _type_multiplier(mtype, p2_types)
            if (cat or "").upper() == "PHYSICAL":
                idx = _simple_damage_index(bp, is_stab, eff, p1_mean_atk, p2_def)
            elif (cat or "").upper() == "SPECIAL":
                idx = _simple_damage_index(bp, is_stab, eff, p1_mean_spa, p2_spd)
            else:
                idx = 0.0
            vals.append(idx)
        return float(np.mean(vals)) if vals else 0.0

    return {
        "lead_matchup_p1_index_full": _acc(None),
        "lead_matchup_p1_index_5": _acc(5),
    }


# ==========================
# SWITCH / HAZARD / MOMENTUM
# ==========================
def _switch_count(tl: list[dict], who: str) -> float:
    last = None
    cnt = 0
    key = f"{who}_pokemon_state"
    for t in tl:
        if not isinstance(t, dict):
            continue
        nm = (t.get(key) or {}).get("name")
        if nm is None:
            continue
        if last is not None and nm != last:
            cnt += 1
        last = nm
    return float(cnt)


HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}


def _hazard_flags(tl: list[dict]) -> dict:
    p1 = p2 = 0.0
    for t in tl:
        if not isinstance(t, dict):
            continue
        m1 = (t.get("p1_move_details") or {}).get("name")
        m2 = (t.get("p2_move_details") or {}).get("name")
        if m1 and str(m1).strip().lower() in HAZARD_MOVES:
            p1 = 1.0
        if m2 and str(m2).strip().lower() in HAZARD_MOVES:
            p2 = 1.0
    return {"hazard_p1_flag": p1, "hazard_p2_flag": p2, "hazard_flag_diff": p1 - p2}


def _momentum_shift(tl: list[dict], t1: int = 3, t2: int = 10) -> dict:
    def _hp_diff_mean(win):
        p1, p2 = _extract_hp_series(win)
        if not p1 or not p2 or len(p1) != len(p2):
            return 0.0
        d = [a - b for a, b in zip(p1, p2)]
        return float(np.mean(d)) if d else 0.0

    d1 = _hp_diff_mean(tl[:t1])
    d2 = _hp_diff_mean(tl[:t2])
    return {"momentum_shift_3_10": float(d1 - d2), "momentum_shift_abs_3_10": float(abs(d1 - d2))}


HEAL_MOVES = {
    "recover",
    "roost",
    "softboiled",
    "rest",
    "wish",
    "synthesis",
    "morningsun",
    "moonlight",
    "drainpunch",
    "leechseed",
}


def _recovery_pressure(tl: list[dict]) -> dict:
    p1 = p2 = 0.0
    for t in tl:
        if not isinstance(t, dict):
            continue
        m1 = (t.get("p1_move_details") or {}).get("name")
        m2 = (t.get("p2_move_details") or {}).get("name")
        if m1 and str(m1).strip().lower() in HEAL_MOVES:
            p1 += 1.0
        if m2 and str(m2).strip().lower() in HEAL_MOVES:
            p2 += 1.0
    return {"recover_p1_count": p1, "recover_p2_count": p2, "recover_count_diff": p1 - p2}


# ---------------------------------------------
# NEW FEATURES (simple HP / stats logic)
# ---------------------------------------------
def new_features(r):
    tl = get_timeline(r, max_turns=30)
    p1, p2 = _extract_hp_series(tl)
    t1 = r.get("p1_team_details", []) or []
    lead = r.get("p2_lead_details", {}) or {}

    f = {}

    if len(p1) >= 3 and len(p2) >= 3:
        f["early_hp_winner"] = 1.0 if np.mean(p1[:3]) > np.mean(p2[:3]) else 0.0
        f["early_hp_difference"] = np.mean(p1[:3]) - np.mean(p2[:3])

    if p1 and p2:
        f["final_hp_winner"] = 1.0 if p1[-1] > p2[-1] else 0.0
        f["final_hp_difference"] = p1[-1] - p2[-1]

    p1_total_stats = sum(p.get(k, 0) for p in t1 for k in BASE_STAT_KEYS)
    p2_total_stats = sum(lead.get(k, 0) for k in BASE_STAT_KEYS)
    f["stronger_team"] = 1.0 if p1_total_stats > p2_total_stats else 0.0
    f["team_strength_gap"] = p1_total_stats - p2_total_stats

    p1_speeds = [p.get("base_spe", 0) for p in t1]
    p2_speed = lead.get("base_spe", 0)
    f["faster_team"] = 1.0 if max(p1_speeds, default=0) > p2_speed else 0.0
    f["speed_advantage"] = max(p1_speeds, default=0) - p2_speed
    f["num_faster_pokemon"] = sum(1 for s in p1_speeds if s > p2_speed)

    f["p1_danger_count"] = sum(1 for hp in p1 if 0 < hp < 25)
    f["p2_danger_count"] = sum(1 for hp in p2 if 0 < hp < 25)
    f["survived_more_danger"] = 1.0 if f["p1_danger_count"] < f["p2_danger_count"] else 0.0
    return f


# ---------------------------------------------
# Mirko & Deb — defensive profile helper
# ---------------------------------------------
def get_defensive_profile(types):
    """
    Combined defensive multipliers for a defender with 'types' against every attack type.
    """
    types = types or []
    if isinstance(types, str):
        types = [types]
    types_up = [str(t).strip().upper() for t in types if t]

    combined = {}
    for atk_type in _TYPE_CHART.keys():
        mult = 1.0
        for tdef in types_up:
            mult *= _TYPE_CHART.get(atk_type, {}).get(tdef, 1.0)
        combined[atk_type] = float(mult)
    return combined


def new_features_mirko(battle):
    """
    Mirko's extra aggregate features for P1 team + P2 lead + timeline heuristics.
    """
    features = {}

    # Player 1 Team aggregate
    p1_team = battle.get("p1_team_details", []) or []
    if p1_team:
        ratios = []
        v_hp = []
        v_spe = []
        v_atk = []
        v_def = []
        all_types = []
        weaknesses = []
        resistances = []
        immunities = []
        for p in p1_team:
            if not isinstance(p, dict):
                continue
            off = p.get("base_atk", 0) + p.get("base_spa", 0)
            deff = p.get("base_def", 0) + p.get("base_spd", 0)
            ratios.append(off / deff if deff > 0 else 0.0)

            v_hp.append(p.get("base_hp", 0))
            v_spe.append(p.get("base_spe", 0))
            v_atk.append(p.get("base_atk", 0))
            v_def.append(p.get("base_def", 0))

            ts = p.get("types") or []
            if isinstance(ts, str):
                ts = [ts]
            all_types.extend([t for t in ts if str(t).lower() != "notype"])

            prof = get_defensive_profile(ts)
            w = sum(1 for m in prof.values() if m > 1)
            r = sum(1 for m in prof.values() if 0 < m < 1)
            i = sum(1 for m in prof.values() if m == 0)
            weaknesses.append(w)
            resistances.append(r)
            immunities.append(i)

        features["avg_type_role_ratio"] = float(np.mean(ratios)) if ratios else 0.0
        features["p1_var_hp"] = float(np.std(v_hp)) if v_hp else 0.0
        features["p1_var_spe"] = float(np.std(v_spe)) if v_spe else 0.0
        features["p1_var_atk"] = float(np.std(v_atk)) if v_atk else 0.0
        features["p1_var_def"] = float(np.std(v_def)) if v_def else 0.0

        unique_types = len(set(all_types))
        features["diversity_ratio"] = unique_types / 6.0

        features["avg_weaknesses"] = float(np.mean(weaknesses)) if weaknesses else 0.0
        features["avg_resistances"] = float(np.mean(resistances)) if resistances else 0.0
        features["avg_immunities"] = float(np.mean(immunities)) if immunities else 0.0

    # P2 lead raw stats
    p2_lead = battle.get("p2_lead_details") or {}
    if isinstance(p2_lead, dict) and p2_lead:
        features["p2_lead_hp"] = p2_lead.get("base_hp", 0)
        features["p2_lead_spe"] = p2_lead.get("base_spe", 0)
        features["p2_lead_atk"] = p2_lead.get("base_atk", 0)
        features["p2_lead_def"] = p2_lead.get("base_def", 0)

    # Voluntary leave counters (None move_details ~ skipped)
    tl = battle.get("battle_timeline", []) or []
    idx_none_p2 = [i + 1 for i, e in enumerate(tl) if e.get("p2_move_details") is None]
    idx_none_p1 = [i + 1 for i, e in enumerate(tl) if e.get("p1_move_details") is None]

    def _bucket_count(idxs, a, b):
        return len([x for x in idxs if a <= x <= b])

    features["vol_leave_diff_1"] = _bucket_count(idx_none_p1, 1, 10) - _bucket_count(idx_none_p2, 1, 10)
    features["vol_leave_diff_2"] = _bucket_count(idx_none_p1, 11, 20) - _bucket_count(idx_none_p2, 11, 20)
    features["vol_leave_diff_3"] = _bucket_count(idx_none_p1, 21, 10**9) - _bucket_count(idx_none_p2, 21, 10**9)

    # Forced leave heuristics (name change + action executed)
    def _forced_counts(side_key, move_key):
        lst = []
        for t in tl:
            if not isinstance(t, dict):
                continue
            lst.append(
                [
                    (t.get(side_key) or {}).get("name"),
                    (t.get(move_key) is None),
                ]
            )
        c1 = c2 = c3 = 0
        for i in range(len(lst) - 1):
            changed = lst[i + 1][0] != lst[i][0]
            acted = lst[i + 1][1] is False
            turn_idx = i + 1
            if changed and acted:
                if 1 <= turn_idx <= 10:
                    c1 += 1
                elif 11 <= turn_idx <= 20:
                    c2 += 1
                else:
                    c3 += 1
        return c1, c2, c3

    p1c1, p1c2, p1c3 = _forced_counts("p1_pokemon_state", "p1_move_details")
    p2c1, p2c2, p2c3 = _forced_counts("p2_pokemon_state", "p2_move_details")
    features["forced_leave_diff_1"] = float(p1c1 - p2c1)
    features["forced_leave_diff_2"] = float(p1c2 - p2c2)
    features["forced_leave_diff_3"] = float(p1c3 - p2c3)

    # IDs / target
    features["battle_id"] = battle.get("battle_id")
    if "player_won" in battle:
        features["player_won"] = int(battle["player_won"])
    return features


# ======= helpers for team & HP & damage stats =======
def _pnames_from_p1_team(record):
    team = record.get("p1_team_details", []) or []
    names = []
    for p in team:
        if isinstance(p, dict):
            nm = (p.get("name") or "").strip().lower()
            if nm:
                names.append(nm)
    return names


def _pname_from_p2_lead(record):
    lead = record.get("p2_lead_details") or {}
    if isinstance(lead, dict):
        nm = (lead.get("name") or "").strip().lower()
        return nm if nm else None
    return None


def build_pokemon_win_stats(train_data, alpha=1.0):
    games = defaultdict(int)
    wins = defaultdict(int)
    for r in train_data:
        p1_names = _pnames_from_p1_team(r)
        p2_lead = _pname_from_p2_lead(r)
        p1_won = bool(r.get("player_won", False))
        for nm in p1_names:
            games[nm] += 1
        if p2_lead:
            games[p2_lead] += 1
        if p1_won:
            for nm in p1_names:
                wins[nm] += 1
        else:
            if p2_lead:
                wins[p2_lead] += 1
    winrate = {}
    for nm in games:
        g = games[nm]
        w = wins[nm]
        wr = (w + alpha) / (g + 2 * alpha)
        winrate[nm] = {"games": g, "wins": w, "winrate": wr}
    return winrate


def team_score_from_stats(team_names, stats, default_wr=0.5):
    vals = [stats.get((nm or "").strip().lower(), {}).get("winrate", default_wr) for nm in team_names if nm]
    return float(np.mean(vals)) if vals else default_wr


def predict_from_stats(test_record, stats, threshold=0.5):
    p1_names = _pnames_from_p1_team(test_record)
    score = team_score_from_stats(p1_names, stats, default_wr=0.5)
    return (score > threshold), score


def build_pokemon_hp_stats(train_data):
    hp_sum = defaultdict(float)
    hp_count = defaultdict(int)
    for r in train_data:
        timeline = r.get("battle_timeline", []) or []
        if not timeline:
            continue
        last_turn = timeline[-1]
        for player_key in ["p1_pokemon_state", "p2_pokemon_state"]:
            name = (last_turn.get(player_key, {}).get("name") or "").strip().lower()
            hp = last_turn.get(player_key, {}).get("hp_pct", None)
            if name and isinstance(hp, (int, float)):
                hp_sum[name] += float(hp)
                hp_count[name] += 1
    stats = {name: {"count": hp_count[name], "hp_mean": hp_sum[name] / hp_count[name]} for name in hp_sum}
    return stats


def team_hp_score(team_names, hp_stats, default_hp=50.0):
    vals = []
    for name in team_names:
        n = (name or "").strip().lower()
        vals.append(hp_stats.get(n, {}).get("hp_mean", default_hp))
    return float(np.mean(vals)) if vals else default_hp


def build_pokemon_avg_damage(train_data):
    total_damage = defaultdict(float)
    battles_count = defaultdict(int)
    for battle in train_data:
        timeline = battle.get("battle_timeline", []) or []
        p1_team = battle.get("p1_team_details", []) or []
        p1_names = [(p.get("name") or "").lower() for p in p1_team if isinstance(p, dict)]
        p2_lead = battle.get("p2_lead_details", {})
        p2_name = (p2_lead.get("name") or "").lower() if isinstance(p2_lead, dict) else None

        for name in p1_names:
            battles_count[name] += 1
        if p2_name:
            battles_count[p2_name] += 1

        for i in range(1, len(timeline)):
            prev, curr = timeline[i - 1], timeline[i]
            if not (isinstance(prev, dict) and isinstance(curr, dict)):
                continue
            # P1 attacking P2
            hp2b = (prev.get("p2_pokemon_state") or {}).get("hp_pct", None)
            hp2a = (curr.get("p2_pokemon_state") or {}).get("hp_pct", None)
            if isinstance(hp2b, (int, float)) and isinstance(hp2a, (int, float)):
                dmg = max(0, hp2b - hp2a)
                if p1_names and dmg > 0:
                    for name in p1_names:
                        total_damage[name] += dmg
            # P2 attacking P1
            hp1b = (prev.get("p1_pokemon_state") or {}).get("hp_pct", None)
            hp1a = (curr.get("p1_pokemon_state") or {}).get("hp_pct", None)
            if isinstance(hp1b, (int, float)) and isinstance(hp1a, (int, float)):
                dmg = max(0, hp1b - hp1a)
                if p2_name and dmg > 0:
                    total_damage[p2_name] += dmg

    avg_damage = {name: total_damage[name] / battles_count[name] for name in battles_count if battles_count[name] > 0}
    return avg_damage


def damage_feature_for_battle(record, avg_damage):
    p1_team = record.get("p1_team_details", []) or []
    p1_names = [(p.get("name") or "").lower() for p in p1_team if isinstance(p, dict)]
    p1_damage_score = sum(avg_damage.get(name, 0.0) for name in p1_names)
    p2_lead = record.get("p2_lead_details", {}) or {}
    p2_name = (p2_lead.get("name") or "").lower() if isinstance(p2_lead, dict) else None
    p2_damage_score = avg_damage.get(p2_name, 0.0) if p2_name else 0.0
    diff = p1_damage_score - p2_damage_score
    return {
        "avg_damage_p1": p1_damage_score,
        "avg_damage_p2": p2_damage_score,
        "avg_damage_diff": diff,
        "damage_prediction": 1.0 if diff > 0 else 0.0,
    }


# ======= Deb's feature block (short version: as in your cell) =======
def new_features_deb(r):
    # (kept exactly as in your original code)
    # ...  (omitted here only for brevity – you keep your full definition)
    # return f
    raise NotImplementedError("Paste here your full new_features_deb implementation.")


# ============================================================
# PUBLIC FEATURE ENGINEERING ENTRYPOINT FOR MODEL 3
# ============================================================
def run_feature_engineering_mrk(
    train_data: List[Dict[str, Any]],
    test_file_path: str,
    alpha: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full feature engineering pipeline for the Mirko notebook model.

    Parameters
    ----------
    train_data : list of dict
        Parsed JSONL training battles (already loaded in memory).
    test_file_path : str
        Path to the JSONL test file.
    alpha : float, default=1.0
        Smoothing parameter for Pokémon winrate statistics.

    Returns
    -------
    train_df : pd.DataFrame
        Clean, engineered training features (unscaled).
    test_df : pd.DataFrame
        Clean, engineered test features (unscaled).
    train_df_raw : pd.DataFrame
        Raw feature frame before numeric cleanup.
    test_df_raw : pd.DataFrame
        Raw test feature frame before numeric cleanup.
    """
    # Global stats (built only from training data)
    global POKEMON_STATS, POKEMON_HP_STATS, POKEMON_AVG_DAMAGE

    print("[Model 3] Building global Pokémon statistics from training data...")
    POKEMON_STATS = build_pokemon_win_stats(train_data, alpha=alpha)
    POKEMON_HP_STATS = build_pokemon_hp_stats(train_data)
    POKEMON_AVG_DAMAGE = build_pokemon_avg_damage(train_data)

    # ------------------------------------------------------------------
    # 1) Per-battle feature extraction (you must already have create_simple_features)
    # ------------------------------------------------------------------
    print("[Model 3] Processing training data...")
    train_df = create_simple_features(train_data)  # <-- assumes function is defined in this module

    print("\n[Model 3] Processing test data from JSONL...")
    test_records: List[Dict[str, Any]] = []
    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            test_records.append(json.loads(line))
    test_df = create_simple_features(test_records)

    # Keep raw copies for debugging
    train_df_raw = train_df.copy()
    test_df_raw = test_df.copy()

    print("\n[Model 3] Training features preview (raw):")
    try:
        from IPython.display import display

        display(train_df.head())
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2) Additional interaction features + numeric cleanup
    # ------------------------------------------------------------------
    def _maybe_add_interactions(df: pd.DataFrame) -> pd.DataFrame:
        def safe_mul(a, b, name):
            if a in df.columns and b in df.columns:
                df[name] = df[a] * df[b]

        # Team strength × move power (full)
        safe_mul("p1_team_stat_avg", "mv_p1_power_mean", "ix_p1avg_x_p1pow")

        # Speed × priority advantage (first 5 turns if available)
        if "spe_max_adv" in df.columns and "mv_priority_count_diff_5" in df.columns:
            df["ix_speed_x_prio5"] = df["spe_max_adv"] * df["mv_priority_count_diff_5"]

        # HP momentum × fraction of advantaged turns
        safe_mul("tl_hp_diff_mean", "tl_frac_turns_advantage", "ix_hpmean_x_fracadv")

        # Early momentum × priority diff (first 3 turns)
        if "early_hp_diff_mean_3" in df.columns and "mv_priority_count_diff_5" in df.columns:
            df["ix_early3_x_prio5"] = df["early_hp_diff_mean_3"] * df["mv_priority_count_diff_5"]

        # STAB advantage × early KO score
        if "stab_stab_ratio_diff_full" in df.columns and "early_first_ko_score_3" in df.columns:
            df["ix_stabdiff_x_firstko"] = df["stab_stab_ratio_diff_full"] * df["early_first_ko_score_3"]

        # Type effectiveness × STAB (full)
        if "ter_p1_vs_p2lead_full" in df.columns and "stab_stab_ratio_diff_full" in df.columns:
            df["ix_ter_x_stab_full"] = df["ter_p1_vs_p2lead_full"] * df["stab_stab_ratio_diff_full"]

        # Type effectiveness × early momentum (first 3 turns)
        if "ter_p1_vs_p2lead_5" in df.columns and "early_hp_diff_mean_3" in df.columns:
            df["ix_ter5_x_early3"] = df["ter_p1_vs_p2lead_5"] * df["early_hp_diff_mean_3"]

        # Lead matchup × early momentum
        if "lead_matchup_p1_index_5" in df.columns and "early_hp_diff_mean_3" in df.columns:
            df["ix_leadmatch5_x_early3"] = df["lead_matchup_p1_index_5"] * df["early_hp_diff_mean_3"]

        # Hazards advantage × priority pressure
        if "hazard_flag_diff" in df.columns and "mv_priority_count_diff_5" in df.columns:
            df["ix_hazards_x_prio5"] = df["hazard_flag_diff"] * df["mv_priority_count_diff_5"]

        return df

    train_df = _maybe_add_interactions(train_df)
    test_df = _maybe_add_interactions(test_df)

    # Numeric cleanup
    num_cols = [c for c in train_df.columns if c not in ("battle_id", "player_won")]
    train_df[num_cols] = train_df[num_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    test_df[num_cols] = test_df[num_cols].apply(pd.to_numeric, errors="coerce").astype("float32")

    tr_vals = train_df[num_cols].to_numpy()
    te_vals = test_df[num_cols].to_numpy()
    tr_vals[~np.isfinite(tr_vals)] = np.nan
    te_vals[~np.isfinite(te_vals)] = np.nan
    train_df[num_cols] = tr_vals
    test_df[num_cols] = te_vals

    # Clip percent-like fields to [0, 100]
    num_only = train_df.drop(columns=["battle_id", "player_won"], errors="ignore").select_dtypes(include=[np.number])
    percent_like = [x for x in num_only.columns if ("hp" in x.lower()) or ("auc" in x.lower())]
    for c in percent_like:
        if c in train_df.columns:
            train_df[c] = train_df[c].clip(lower=0, upper=100)
            test_df[c] = test_df[c].clip(lower=0, upper=100)

    # Report near-constants (diagnostic only)
    near_const = [
        c
        for c in num_only.columns
        if (num_only[c].nunique(dropna=True) / max(1, len(num_only)) < 0.01)
    ]
    print(f"[Model 3][Sanity] Near-constant features (not dropping): {len(near_const)}")

    # --- 10 safe engineered features (exactly as in your cell) ---
    EPS = 1e-6
    REPLACE_EXISTING = True

    def _pick_first(df: pd.DataFrame, candidates, default_value=0.0):
        for c in candidates:
            if c in df.columns:
                return df[c].astype("float32")
        return pd.Series(default_value, index=df.index, dtype="float32")

    def _safe_div(a: pd.Series, b: pd.Series):
        out = a.astype("float32") / (b.astype("float32") + EPS)
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")
        return out

    def _ensure_float32(s: pd.Series):
        return s.astype("float32").replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def _normalize_acc(s: pd.Series):
        s = _ensure_float32(s)
        if len(s):
            maxv = float(np.nanmax(s.values))
        else:
            maxv = 0.0
        if maxv > 1.5:
            s = s / 100.0
        return s.clip(0.0, 1.0)

    def _add_feature_pair(train_df_local, test_df_local, name, train_series, test_series):
        if (not REPLACE_EXISTING) and (name in train_df_local.columns or name in test_df_local.columns):
            return
        train_df_local[name] = _ensure_float32(train_series)
        test_df_local[name] = _ensure_float32(test_series)

    # --- Base columns (train/test) ---
    atk_p1 = _pick_first(train_df, ["atk_p1_mean", "atk_p1", "atk_p1_full"], 0.0)
    atk_p2 = _pick_first(train_df, ["atk_p2_mean", "atk_p2", "atk_p2_full"], 0.0)
    def_p1 = _pick_first(train_df, ["def_p1_mean", "def_p1", "def_p1_full"], 0.0)
    def_p2 = _pick_first(train_df, ["def_p2_mean", "def_p2", "def_p2_full"], 0.0)

    atk_p1_te = _pick_first(test_df, ["atk_p1_mean", "atk_p1", "atk_p1_full"], 0.0)
    atk_p2_te = _pick_first(test_df, ["atk_p2_mean", "atk_p2", "atk_p2_full"], 0.0)
    def_p1_te = _pick_first(test_df, ["def_p1_mean", "def_p1", "def_p1_full"], 0.0)
    def_p2_te = _pick_first(test_df, ["def_p2_mean", "def_p2", "def_p2_full"], 0.0)

    sp_atk_p1 = _pick_first(train_df, ["sp_atk_p1_mean", "spatk_p1_mean", "spa_p1_mean", "sp_atk_p1"], 0.0)
    sp_atk_p2 = _pick_first(train_df, ["sp_atk_p2_mean", "spatk_p2_mean", "spa_p2_mean", "sp_atk_p2"], 0.0)
    sp_def_p1 = _pick_first(train_df, ["sp_def_p1_mean", "spdef_p1_mean", "spd_p1_mean_def", "sp_def_p1"], 0.0)
    sp_def_p2 = _pick_first(train_df, ["sp_def_p2_mean", "spdef_p2_mean", "spd_p2_mean_def", "sp_def_p2"], 0.0)

    sp_atk_p1_te = _pick_first(test_df, ["sp_atk_p1_mean", "spatk_p1_mean", "spa_p1_mean", "sp_atk_p1"], 0.0)
    sp_atk_p2_te = _pick_first(test_df, ["sp_atk_p2_mean", "spatk_p2_mean", "spa_p2_mean", "sp_atk_p2"], 0.0)
    sp_def_p1_te = _pick_first(test_df, ["sp_def_p1_mean", "spdef_p1_mean", "spd_p1_mean_def", "sp_def_p1"], 0.0)
    sp_def_p2_te = _pick_first(test_df, ["sp_def_p2_mean", "spdef_p2_mean", "spd_p2_mean_def", "sp_def_p2"], 0.0)

    spd_p1 = _pick_first(train_df, ["spd_p1_mean", "speed_p1_mean", "spd_p1"], 0.0)
    spd_p2 = _pick_first(train_df, ["spd_p2_mean", "speed_p2_mean", "spd_p2"], 0.0)
    spd_p1_te = _pick_first(test_df, ["spd_p1_mean", "speed_p1_mean", "spd_p1"], 0.0)
    spd_p2_te = _pick_first(test_df, ["spd_p2_mean", "speed_p2_mean", "spd_p2"], 0.0)

    hp1_cur = _pick_first(train_df, ["hp_p1_remain", "hp_p1_curr", "hp_p1"], 0.0)
    hp2_cur = _pick_first(train_df, ["hp_p2_remain", "hp_p2_curr", "hp_p2"], 0.0)
    hp1_max = _pick_first(train_df, ["hp_p1_max", "hp_p1_base", "hp_p1_total"], 1.0)
    hp2_max = _pick_first(train_df, ["hp_p2_max", "hp_p2_base", "hp_p2_total"], 1.0)

    hp1_cur_te = _pick_first(test_df, ["hp_p1_remain", "hp_p1_curr", "hp_p1"], 0.0)
    hp2_cur_te = _pick_first(test_df, ["hp_p2_remain", "hp_p2_curr", "hp_p2"], 0.0)
    hp1_max_te = _pick_first(test_df, ["hp_p1_max", "hp_p1_base", "hp_p1_total"], 1.0)
    hp2_max_te = _pick_first(test_df, ["hp_p2_max", "hp_p2_base", "hp_p2_total"], 1.0)

    pwr_p1 = _pick_first(train_df, ["mv_p1_power_mean_full", "mv_p1_power_mean", "mv_power_p1_mean"], 0.0)
    pwr_p2 = _pick_first(train_df, ["mv_p2_power_mean_full", "mv_p2_power_mean", "mv_power_p2_mean"], 0.0)
    acc_p1 = _normalize_acc(_pick_first(train_df, ["mv_p1_acc_mean_full", "mv_p1_acc_mean", "mv_acc_p1_mean"], 0.0))
    acc_p2 = _normalize_acc(_pick_first(train_df, ["mv_p2_acc_mean_full", "mv_p2_acc_mean", "mv_acc_p2_mean"], 0.0))

    pwr_p1_te = _pick_first(test_df, ["mv_p1_power_mean_full", "mv_p1_power_mean", "mv_power_p1_mean"], 0.0)
    pwr_p2_te = _pick_first(test_df, ["mv_p2_power_mean_full", "mv_p2_power_mean", "mv_power_p2_mean"], 0.0)
    acc_p1_te = _normalize_acc(
        _pick_first(test_df, ["mv_p1_acc_mean_full", "mv_p1_acc_mean", "mv_acc_p1_mean"], 0.0)
    )
    acc_p2_te = _normalize_acc(
        _pick_first(test_df, ["mv_p2_acc_mean_full", "mv_p2_acc_mean", "mv_acc_p2_mean"], 0.0)
    )

    st_p1 = _pick_first(train_df, ["mv_p1_count_STATUS_full", "mv_p1_count_STATUS", "status_moves_p1"], 0.0)
    ph_p1 = _pick_first(train_df, ["mv_p1_count_PHYSICAL_full", "mv_p1_count_PHYSICAL", "physical_moves_p1"], 0.0)
    sp_p1 = _pick_first(train_df, ["mv_p1_count_SPECIAL_full", "mv_p1_count_SPECIAL", "special_moves_p1"], 0.0)
    st_p2 = _pick_first(train_df, ["mv_p2_count_STATUS_full", "mv_p2_count_STATUS", "status_moves_p2"], 0.0)
    ph_p2 = _pick_first(train_df, ["mv_p2_count_PHYSICAL_full", "mv_p2_count_PHYSICAL", "physical_moves_p2"], 0.0)
    sp_p2 = _pick_first(train_df, ["mv_p2_count_SPECIAL_full", "mv_p2_count_SPECIAL", "special_moves_p2"], 0.0)

    st_p1_te = _pick_first(test_df, ["mv_p1_count_STATUS_full", "mv_p1_count_STATUS", "status_moves_p1"], 0.0)
    ph_p1_te = _pick_first(test_df, ["mv_p1_count_PHYSICAL_full", "mv_p1_count_PHYSICAL", "physical_moves_p1"], 0.0)
    sp_p1_te = _pick_first(test_df, ["mv_p1_count_SPECIAL_full", "mv_p1_count_SPECIAL", "special_moves_p1"], 0.0)
    st_p2_te = _pick_first(test_df, ["mv_p2_count_STATUS_full", "mv_p2_count_STATUS", "status_moves_p2"], 0.0)
    ph_p2_te = _pick_first(test_df, ["mv_p2_count_PHYSICAL_full", "mv_p2_count_PHYSICAL", "physical_moves_p2"], 0.0)
    sp_p2_te = _pick_first(test_df, ["mv_p2_count_SPECIAL_full", "mv_p2_count_SPECIAL", "special_moves_p2"], 0.0)

    # 1) atk_def_ratio
    _add_feature_pair(
        train_df,
        test_df,
        "atk_def_ratio",
        _safe_div(atk_p1, def_p2),
        _safe_div(atk_p1_te, def_p2_te),
    )

    # 2) spd_gap
    _add_feature_pair(
        train_df,
        test_df,
        "spd_gap",
        (spd_p1 - spd_p2),
        (spd_p1_te - spd_p2_te),
    )

    # 3) hp_ratio
    _add_feature_pair(
        train_df,
        test_df,
        "hp_ratio",
        _safe_div(hp1_cur, hp2_cur),
        _safe_div(hp1_cur_te, hp2_cur_te),
    )

    # 4) survival_score
    _add_feature_pair(
        train_df,
        test_df,
        "survival_score",
        _safe_div(hp1_cur, hp1_max) - _safe_div(hp2_cur, hp2_max),
        _safe_div(hp1_cur_te, hp1_max_te) - _safe_div(hp2_cur_te, hp2_max_te),
    )

    # 5) momentum_index
    _add_feature_pair(
        train_df,
        test_df,
        "momentum_index",
        _safe_div(atk_p1 * spd_p1, atk_p2 * spd_p2),
        _safe_div(atk_p1_te * spd_p1_te, atk_p2_te * spd_p2_te),
    )

    # 6) power_acc_gap
    pwa_p1 = _ensure_float32(pwr_p1 * acc_p1)
    pwa_p2 = _ensure_float32(pwr_p2 * acc_p2)
    pwa_p1_te = _ensure_float32(pwr_p1_te * acc_p1_te)
    pwa_p2_te = _ensure_float32(pwr_p2_te * acc_p2_te)
    _add_feature_pair(
        train_df,
        test_df,
        "power_acc_gap",
        (pwa_p1 - pwa_p2),
        (pwa_p1_te - pwa_p2_te),
    )

    # 7) offensive_balance
    _add_feature_pair(
        train_df,
        test_df,
        "offensive_balance",
        _safe_div(atk_p1 + sp_atk_p1, atk_p2 + sp_atk_p2),
        _safe_div(atk_p1_te + sp_atk_p1_te, atk_p2_te + sp_atk_p2_te),
    )

    # 8) defensive_efficiency
    _add_feature_pair(
        train_df,
        test_df,
        "defensive_efficiency",
        _safe_div(def_p1 + sp_def_p1, def_p2 + sp_def_p2),
        _safe_div(def_p1_te + sp_def_p1_te, def_p2_te + sp_def_p2_te),
    )

    # 9) status_influence
    tot_p1 = _ensure_float32(st_p1 + ph_p1 + sp_p1).replace(0.0, 1.0)
    tot_p2 = _ensure_float32(st_p2 + ph_p2 + sp_p2).replace(0.0, 1.0)
    tot_p1_te = _ensure_float32(st_p1_te + ph_p1_te + sp_p1_te).replace(0.0, 1.0)
    tot_p2_te = _ensure_float32(st_p2_te + ph_p2_te + sp_p2_te).replace(0.0, 1.0)

    status_share_p1 = _safe_div(st_p1, tot_p1)
    status_share_p2 = _safe_div(st_p2, tot_p2)
    status_share_p1_te = _safe_div(st_p1_te, tot_p1_te)
    status_share_p2_te = _safe_div(st_p2_te, tot_p2_te)

    _add_feature_pair(
        train_df,
        test_df,
        "status_influence",
        (status_share_p1 - status_share_p2),
        (status_share_p1_te - status_share_p2_te),
    )

    # 10) speed_ratio
    _add_feature_pair(
        train_df,
        test_df,
        "speed_ratio",
        _safe_div(spd_p1, spd_p2),
        _safe_div(spd_p1_te, spd_p2_te),
    )

    new_cols = [
        "atk_def_ratio",
        "spd_gap",
        "hp_ratio",
        "survival_score",
        "momentum_index",
        "power_acc_gap",
        "offensive_balance",
        "defensive_efficiency",
        "status_influence",
        "speed_ratio",
    ]
    bad_train = train_df[new_cols].isna().sum().sum() + np.isinf(train_df[new_cols].to_numpy()).sum()
    bad_test = test_df[new_cols].isna().sum().sum() + np.isinf(test_df[new_cols].to_numpy()).sum()
    print(
        f"[Model 3][FeatureEng] Added {len(new_cols)} engineered features. "
        f"Bad values -> train: {bad_train}, test: {bad_test}"
    )

    print("\n[Model 3] Prepared (unscaled, clean types):")
    try:
        from IPython.display import display

        display(train_df.head())
    except Exception:
        pass

    print("\n[Model 3] Feature engineering completed.")
    return train_df, test_df, train_df_raw, test_df_raw

# ## 2.A Overfitting check

# In[3]:


# === Cell A: overfitting diagnostics (learning curve) ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, learning_curve, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ==========================================
# Overfitting diagnostics (learning curve)
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def diagnose_overfitting_lr(
    train_df: pd.DataFrame,
    target_col: str = "player_won",
    id_cols: list[str] | None = None,
    cv_splits: int = 5,
    train_sizes = None,
    random_state: int = 42,
    max_iter: int = 1000,
    plot: bool = True,
):
    """
    Compute a learning curve for a Logistic Regression pipeline (Scaler + Logistic),
    using the given training DataFrame.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data including the target column.
    target_col : str, default "player_won"
        Name of the binary target column.
    id_cols : list[str] or None, default None
        Columns to exclude from X (e.g. ["battle_id"]). If None, no extra id columns are dropped.
    cv_splits : int, default 5
        Number of CV folds for StratifiedKFold.
    train_sizes : array-like or None, default None
        Fractions of training set to use; if None, uses np.linspace(0.1, 1.0, 6).
    random_state : int, default 42
        Random seed for CV and LogisticRegression.
    max_iter : int, default 1000
        Max iterations for LogisticRegression.
    plot : bool, default True
        If True, plots the learning curve.

    Returns
    -------
    results : dict
        {
            "train_sizes": np.ndarray,
            "train_acc_mean": np.ndarray,
            "val_acc_mean": np.ndarray,
            "gap_last": float,
            "val_last": float,
            "overfitting_flag": bool,
            "df_learning_curve": pd.DataFrame
        }
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 6)

    if id_cols is None:
        id_cols = []

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train_df.")

    # Build X, y
    feature_cols = [c for c in train_df.columns if c not in ([target_col] + list(id_cols))]
    X = train_df[feature_cols].values
    y = train_df[target_col].astype(int).values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=max_iter, random_state=random_state))
    ])

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    ts_abs, train_scores, val_scores = learning_curve(
        estimator=pipe,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="accuracy",
        shuffle=True,
        random_state=random_state,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    df_lc = pd.DataFrame({
        "train_size": ts_abs,
        "train_acc_mean": train_mean,
        "train_acc_std": train_std,
        "val_acc_mean": val_mean,
        "val_acc_std": val_std,
        "gap_train_minus_val": train_mean - val_mean,
    })

    try:
        display(df_lc)
    except Exception:
        pass

    gap_last = float(df_lc["gap_train_minus_val"].iloc[-1])
    val_last = float(df_lc["val_acc_mean"].iloc[-1])
    flag_overfit = (gap_last >= 0.05) and (val_last < 0.80 or gap_last > 0.07)

    print(f"\n[Overfitting] Largest train size = {int(ts_abs[-1])}")
    print(f"  • Train acc (mean): {train_mean[-1]:.4f}")
    print(f"  • Val   acc (mean): {val_mean[-1]:.4f}")
    print(f"  • Gap (train - val): {gap_last:.4f}")
    print(f"\n[Overfitting] Potential overfitting: {'YES' if flag_overfit else 'NO'}")

    if plot:
        plt.figure()
        plt.plot(ts_abs, train_mean, marker="o", label="Train acc")
        plt.fill_between(ts_abs, train_mean - train_std, train_mean + train_std, alpha=0.15)
        plt.plot(ts_abs, val_mean, marker="o", label="Validation acc")
        plt.fill_between(ts_abs, val_mean - val_std, val_mean + val_std, alpha=0.15)
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy")
        plt.title("Learning curve — Logistic (scaled)")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "train_sizes": ts_abs,
        "train_acc_mean": train_mean,
        "val_acc_mean": val_mean,
        "gap_last": gap_last,
        "val_last": val_last,
        "overfitting_flag": flag_overfit,
        "df_learning_curve": df_lc,
    }


# # 3. Models Training

# ## 3.1 - Best Features Selection, V.3 (Elastic Net + SelectFromModel)

# In[4]:


# === 3.1 BEST FEATURES SELECTION (Elastic Net + SelectFromModel) ==================
# - Prints original feature count
# - Train-only pruning:
#     (A) correlation pruning (|ρ| > 0.95)
#     (C) optional robust VIF pruning (iterative, safe)
# - Selector: Elastic Net (LogisticRegressionCV, saga) + SelectFromModel with a small threshold sweep
# - Outputs: train_reduced, test_reduced with the exact selected feature subset
# ================================================================================

# ==========================================
# Feature selection (Elastic Net + Top-N)
# ==========================================
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def run_feature_selection_mrk(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "player_won",
    id_cols: list[str] | None = None,
    random_state: int = 42,
    drop_vif: bool = True,
    vif_threshold: float = 25.0,
    max_vif_steps: int = 50,
    corr_threshold: float = 0.99,
    cv_splits: int = 5,
    target_n_features: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Best feature selection for Mirko model:
      - A: constant feature pruning
      - B: correlation pruning |ρ| > corr_threshold
      - C: optional iterative VIF pruning on standardized data
      - Elastic Net (LogisticRegressionCV, saga) on pruned features
      - Select Top-N features by absolute coefficient magnitude

    Parameters
    ----------
    train_df : pd.DataFrame
        Training features with target_col and id_cols.
    test_df : pd.DataFrame
        Test features with id_cols.
    target_col : str, default "player_won"
        Name of the target column.
    id_cols : list[str] or None, default None
        ID columns to keep but not use in training (e.g. ["battle_id"]).
    random_state : int, default 42
        Random seed.
    drop_vif : bool, default True
        If True, run VIF-based pruning.
    vif_threshold : float, default 25.0
        VIF threshold above which features are iteratively dropped.
    max_vif_steps : int, default 50
        Maximum number of VIF pruning iterations.
    corr_threshold : float, default 0.99
        Absolute Pearson correlation threshold for pruning.
    cv_splits : int, default 5
        Number of CV folds for logistic Elastic Net.
    target_n_features : int, default 60
        Desired number of features to keep (Top-N by abs coefficient).

    Returns
    -------
    train_reduced : pd.DataFrame
        DataFrame with [id_cols] + [target_col] + selected features.
    test_reduced : pd.DataFrame
        DataFrame with [id_cols] + selected features.
    selected_cols : list[str]
        Names of selected feature columns (in train_reduced/test_reduced).
    """
    if id_cols is None:
        id_cols = ["battle_id"]

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train_df.")

    # 0) Build numeric matrices
    num_cols_all = [c for c in train_df.columns if c not in ([target_col] + list(id_cols))]
    X = train_df[num_cols_all].copy()
    y = train_df[target_col].astype(int).copy()
    X_test_full = test_df[num_cols_all].copy()

    X = X.apply(pd.to_numeric, errors="coerce").astype("float32")
    X_test_full = X_test_full.apply(pd.to_numeric, errors="coerce").astype("float32")

    orig_feat_count = X.shape[1]
    print(f"[FS-M3] Original feature count (numeric, pre-pruning): {orig_feat_count}")

    # Helper: imputer + scaler (fit on TRAIN only)
    def fit_imputer_scaler(df: pd.DataFrame):
        imp = SimpleImputer(strategy="median")
        sca = RobustScaler()
        Z = imp.fit_transform(df)
        Z = sca.fit_transform(Z)
        return imp, sca, Z

    # (A) Constant features
    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    print(f"[FS-M3][A] Constant features removed: {len(const_cols)}")
    if const_cols:
        print("  -> Constant list (first 50):", const_cols[:50])

    if const_cols:
        X.drop(columns=const_cols, inplace=True, errors="ignore")
        X_test_full.drop(columns=const_cols, inplace=True, errors="ignore")
    print(f"[FS-M3][A] After constant pruning: {X.shape[1]} features")

    # (B) Correlation pruning
    imp_tmp = SimpleImputer(strategy="median").fit(X)
    X_imp = pd.DataFrame(imp_tmp.transform(X), columns=X.columns, index=X.index)

    corr = X_imp.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns if any(upper[col] > corr_threshold)]

    if to_drop_corr:
        print(f"[FS-M3][B] Dropped for high correlation (>|ρ| {corr_threshold}): {len(to_drop_corr)}")
        X.drop(columns=to_drop_corr, inplace=True, errors="ignore")
        X_test_full.drop(columns=to_drop_corr, inplace=True, errors="ignore")
    else:
        print("[FS-M3][B] No features dropped by correlation threshold.")

    print(f"[FS-M3][B] After correlation pruning: {X.shape[1]} features")

    # (C) Optional VIF pruning
    def compute_vif_frame(df_std: pd.DataFrame) -> pd.DataFrame:
        """
        Compute VIF using simple linear regressions on standardized data.
        Returns a DataFrame with columns ['feature', 'vif'].
        """
        from sklearn.linear_model import LinearRegression

        cols = list(df_std.columns)
        vifs = []
        for i, col in enumerate(cols):
            yv = df_std[col].values
            Xv = df_std.drop(columns=[col]).values

            if Xv.shape[1] == 0:
                vifs.append(np.inf)
                continue

            lr = LinearRegression(n_jobs=None)
            lr.fit(Xv, yv)
            yhat = lr.predict(Xv)

            ss_res = np.sum((yv - yhat) ** 2)
            ss_tot = np.sum((yv - np.mean(yv)) ** 2) + 1e-12
            r2 = 1.0 - ss_res / ss_tot

            if r2 >= 0.999999:
                vif_val = np.inf
            else:
                vif_val = 1.0 / max(1e-12, (1.0 - r2))
            vifs.append(vif_val)

        return pd.DataFrame({"feature": cols, "vif": vifs})

    vif_dropped = []
    if drop_vif and X.shape[1] > 2:
        print(f"[FS-M3][C] Starting VIF pruning (thr={vif_threshold}, max steps={max_vif_steps}) ...")
        step = 0
        while step < max_vif_steps and X.shape[1] > 2:
            imp_vif, sca_vif, X_std = fit_imputer_scaler(X)
            X_std = pd.DataFrame(X_std, columns=X.columns, index=X.index)

            vif_frame = compute_vif_frame(X_std)
            max_row = vif_frame.loc[vif_frame["vif"].idxmax()]
            max_feat, max_vif = str(max_row["feature"]), float(max_row["vif"])

            if not np.isfinite(max_vif):
                print(f"  [FS-M3][VIF] Step {step+1}: dropping '{max_feat}' (VIF=inf)")
                vif_dropped.append(max_feat)
                X.drop(columns=[max_feat], inplace=True, errors="ignore")
                X_test_full.drop(columns=[max_feat], inplace=True, errors="ignore")
                step += 1
                continue

            if max_vif <= vif_threshold:
                print(f"  [FS-M3][VIF] All VIF <= {vif_threshold:.1f} (max={max_vif:.2f}). Stopping.")
                break

            print(f"  [FS-M3][VIF] Step {step+1}: dropping '{max_feat}' (VIF={max_vif:.2f})")
            vif_dropped.append(max_feat)
            X.drop(columns=[max_feat], inplace=True, errors="ignore")
            X_test_full.drop(columns=[max_feat], inplace=True, errors="ignore")
            step += 1

        print(f"[FS-M3][C] VIF dropped: {len(vif_dropped)}")
    else:
        print("[FS-M3][C] VIF pruning skipped or not applicable.")

    print(f"[FS-M3] After A+B(+C): {X.shape[1]} features")

    feat_cols_after_pruning = list(X.columns)

    # Elastic Net selector (LogisticRegressionCV, saga)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    imp_sel, sca_sel, X_std_sel = fit_imputer_scaler(X)

    enet = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        Cs=[0.05, 0.1, 0.2, 0.5, 1.0],
        l1_ratios=[0.1, 0.3, 0.5, 0.7],
        scoring="accuracy",
        cv=cv,
        max_iter=5000,
        tol=1e-3,
        n_jobs=-1,
        refit=True,
        random_state=random_state,
    )
    enet.fit(X_std_sel, y)

    # Coefficients
    abs_w = np.abs(enet.coef_.ravel())
    nonzero_idx = np.where(abs_w > 0)[0]

    if nonzero_idx.size < target_n_features:
        relaxed_C = float(enet.C_[0] * 2.0)
        relaxed_l1 = max(0.2, float(enet.l1_ratio_[0]) - 0.1)

        print(f"[FS-M3] Relaxing ElasticNet: C={relaxed_C:.4f}, l1_ratio={relaxed_l1:.2f}")
        enet_relaxed = LogisticRegressionCV(
            penalty="elasticnet",
            solver="saga",
            Cs=[relaxed_C],
            l1_ratios=[relaxed_l1],
            scoring="accuracy",
            cv=cv,
            max_iter=6000,
            tol=1e-3,
            n_jobs=-1,
            refit=True,
            random_state=random_state,
        )
        enet_relaxed.fit(X_std_sel, y)
        enet = enet_relaxed
        abs_w = np.abs(enet.coef_.ravel())
        nonzero_idx = np.where(abs_w > 0)[0]

    n_take = min(target_n_features, abs_w.size)
    thresh_val = np.partition(abs_w, -n_take)[-n_take]
    top_mask = abs_w >= thresh_val

    selected_cols = list(np.array(feat_cols_after_pruning)[top_mask])
    print(f"[FS-M3][ElasticNet+TopN] Non-zero weights: {nonzero_idx.size} | "
          f"Selected top-{n_take}: {len(selected_cols)}")

    # Build reduced DataFrames
    train_reduced = pd.concat(
        [train_df[id_cols], train_df[[target_col]], train_df[selected_cols]],
        axis=1,
    )
    test_reduced = pd.concat(
        [test_df[id_cols], test_df[selected_cols]],
        axis=1,
    )

    print(f"[FS-M3][Output] train_reduced shape: {train_reduced.shape} | "
          f"test_reduced shape: {test_reduced.shape}")
    print(f"[FS-M3][Features] Final selected ({len(selected_cols)}). First 25 -> {selected_cols[:25]}")

    return train_reduced, test_reduced, selected_cols

# ==========================================
# 3.2 Stacking (LR + XGB + RF -> LR meta)
# ==========================================
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb


def run_stacking_mrk(
    train_reduced: pd.DataFrame,
    test_reduced: pd.DataFrame,
    selected_cols: list[str],
    seeds: list[tuple[int, int, int]] | None = None,
    folds: int = 10,
    meta_random_state: int = 42,
) -> dict:
    """
    True OOF stacking for Mirko's model:
      - Base learners: LogisticRegression, XGBoost, RandomForest
      - Per-fold training, XGBoost + RF calibrated with sigmoid on the validation fold
      - Meta-learner: LogisticRegression on base OOF probabilities
      - Multi-seed wrapper over (LR_seed, XGB_seed, RF_seed) tuples

    Parameters
    ----------
    train_reduced : pd.DataFrame
        Must contain 'player_won' and the selected_cols.
    test_reduced : pd.DataFrame
        Must contain selected_cols.
    selected_cols : list[str]
        Feature names to use for the stacking (output of feature selection).
    seeds : list[tuple[int, int, int]] or None, default None
        List of (seed_lr, seed_xgb, seed_rf). If None, uses [(50, 55, 160)].
    folds : int, default 10
        Number of CV folds for stacking.
    meta_random_state : int, default 42
        Random state used for meta-learner CV splits.

    Returns
    -------
    results : dict
        {
          "best_seed": str,
          "best_auc": float,
          "y": np.ndarray,                     # target
          "oof_meta_scores": np.ndarray,       # OOF meta probabilities
          "meta_test_scores": np.ndarray,      # stacked test probabilities
          "all_results": list[dict]            # per-seed info
        }
    """
    if seeds is None:
        seeds = [(50, 55, 160)]

    # Basic checks
    if "player_won" not in train_reduced.columns:
        raise ValueError("Column 'player_won' not found in train_reduced.")
    for col in selected_cols:
        if col not in train_reduced.columns:
            raise ValueError(f"Selected feature '{col}' not found in train_reduced.")
        if col not in test_reduced.columns:
            raise ValueError(f"Selected feature '{col}' not found in test_reduced.")

    # Matrices
    X_sel = train_reduced[selected_cols].to_numpy()
    X_test_sel = test_reduced[selected_cols].to_numpy()
    y = train_reduced["player_won"].astype(int).to_numpy()

    n_train = X_sel.shape[0]
    n_test = X_test_sel.shape[0]

    print(f"[Stack LR+XGB+RF→LR] Using {X_sel.shape[1]} selected features on {n_train} training rows.")

    all_results = []

    for idx, (seed_lr, seed_xgb, seed_rf) in enumerate(seeds, start=1):
        print("\n" + "=" * 90)
        print(f"Running stacking iteration {idx}")
        print(f"LR_seed={seed_lr}, XGB_seed={seed_xgb}, RF_seed={seed_rf}")
        print("=" * 90)

        # For meta-learner CV
        np.random.seed(meta_random_state)
        FOLDS = folds

        # Base learners -----------------------------------------------------
        base_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="liblinear",
                penalty="l2",
                C=0.5,
                max_iter=3000,
                random_state=seed_lr,
            ),
        )

        base_xgb_params = dict(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed_xgb,
            n_jobs=-1,
            tree_method="hist",
        )

        base_rf_template = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=10,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=seed_rf,
        )

        base_names = ["lr", "xgb", "rf"]
        n_base = len(base_names)

        # OOF holders (level-1 features)
        oof_base = np.zeros((n_train, n_base), dtype=float)
        test_base_folds = np.zeros((n_test, n_base, FOLDS), dtype=float)

        skf = StratifiedKFold(
            n_splits=FOLDS,
            shuffle=True,
            random_state=meta_random_state + idx,
        )

        print("\n[Per-fold validation summary]")
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_sel, y), start=1):
            X_tr, X_va = X_sel[tr_idx], X_sel[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            # ---- Base 1: Logistic Regression (no calibration) ----
            lr_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    solver="liblinear",
                    penalty="l2",
                    C=0.5,
                    max_iter=3000,
                    random_state=seed_lr,
                ),
            )
            lr_model.fit(X_tr, y_tr)
            lr_va = lr_model.predict_proba(X_va)[:, 1]
            lr_te = lr_model.predict_proba(X_test_sel)[:, 1]

            # ---- Base 2: XGBoost (with calibration) ----
            xgb_model = xgb.XGBClassifier(**base_xgb_params)
            xgb_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )

            try:
                best_it = getattr(xgb_model, "best_iteration", None)
                if best_it is not None:
                    xgb_va_raw = xgb_model.predict_proba(
                        X_va, iteration_range=(0, best_it + 1)
                    )[:, 1]
                    xgb_te_raw = xgb_model.predict_proba(
                        X_test_sel, iteration_range=(0, best_it + 1)
                    )[:, 1]
                else:
                    xgb_va_raw = xgb_model.predict_proba(X_va)[:, 1]
                    xgb_te_raw = xgb_model.predict_proba(X_test_sel)[:, 1]
                used_best = best_it
            except Exception:
                xgb_va_raw = xgb_model.predict_proba(X_va)[:, 1]
                xgb_te_raw = xgb_model.predict_proba(X_test_sel)[:, 1]
                used_best = "N/A"

            xgb_cal = CalibratedClassifierCV(
                estimator=xgb_model, method="sigmoid", cv="prefit"
            )
            xgb_cal.fit(X_va, y_va)
            xgb_va = xgb_cal.predict_proba(X_va)[:, 1]
            xgb_te = xgb_cal.predict_proba(X_test_sel)[:, 1]

            # ---- Base 3: Random Forest (per-fold seed) + calibration ----
            rf_model = RandomForestClassifier(
                n_estimators=base_rf_template.n_estimators,
                max_depth=base_rf_template.max_depth,
                min_samples_leaf=base_rf_template.min_samples_leaf,
                max_features=base_rf_template.max_features,
                bootstrap=True,
                n_jobs=-1,
                random_state=seed_rf + fold,  # slight variation per fold
            )
            rf_model.fit(X_tr, y_tr)
            rf_cal = CalibratedClassifierCV(
                estimator=rf_model, method="sigmoid", cv="prefit"
            )
            rf_cal.fit(X_va, y_va)
            rf_va = rf_cal.predict_proba(X_va)[:, 1]
            rf_te = rf_cal.predict_proba(X_test_sel)[:, 1]

            # Store OOF + test probs
            oof_base[va_idx, 0] = lr_va
            oof_base[va_idx, 1] = xgb_va
            oof_base[va_idx, 2] = rf_va

            test_base_folds[:, 0, fold - 1] = lr_te
            test_base_folds[:, 1, fold - 1] = xgb_te
            test_base_folds[:, 2, fold - 1] = rf_te

            # Quick metrics per fold
            def _rep(p):
                acc = accuracy_score(y_va, (p >= 0.5).astype(int))
                try:
                    auc = roc_auc_score(y_va, p)
                except Exception:
                    auc = np.nan
                return acc, auc

            acc_lr, auc_lr = _rep(lr_va)
            acc_xgb, auc_xgb = _rep(xgb_va)
            acc_rf, auc_rf = _rep(rf_va)

            print(
                f"  [Fold {fold}] "
                f"LR  acc={acc_lr:.4f} | AUC={auc_lr:.4f}  ||  "
                f"XGB acc={acc_xgb:.4f} | AUC={auc_xgb:.4f} | best_iter={used_best}  ||  "
                f"RF  acc={acc_rf:.4f} | AUC={auc_rf:.4f}"
            )

        # Aggregate test probs across folds for each base learner
        test_base_mean = test_base_folds.mean(axis=2)  # shape: (n_test, 3)

        # Meta-learner on OOF base features (true OOF again)
        meta_clf_base = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            max_iter=5000,
            random_state=meta_random_state,
        )

        oof_meta_scores = np.zeros(n_train, dtype=float)
        meta_test_folds = np.zeros((n_test, FOLDS), dtype=float)

        skf_meta = StratifiedKFold(
            n_splits=FOLDS,
            shuffle=True,
            random_state=meta_random_state + 1,
        )
        for fold, (tr_idx, va_idx) in enumerate(skf_meta.split(oof_base, y), start=1):
            X_tr_m, X_va_m = oof_base[tr_idx], oof_base[va_idx]
            y_tr_m, y_va_m = y[tr_idx], y[va_idx]

            meta_clf_fold = LogisticRegression(
                solver="lbfgs",
                penalty="l2",
                C=1.0,
                max_iter=5000,
                random_state=meta_random_state + fold,
            )
            meta_clf_fold.fit(X_tr_m, y_tr_m)
            oof_meta_scores[va_idx] = meta_clf_fold.predict_proba(X_va_m)[:, 1]
            meta_test_folds[:, fold - 1] = meta_clf_fold.predict_proba(
                test_base_mean
            )[:, 1]

        # Final meta on full OOF
        meta_clf_base.fit(oof_base, y)
        meta_test_scores = meta_test_folds.mean(axis=1)

        oof_acc_default = accuracy_score(
            y, (oof_meta_scores >= 0.50).astype(int)
        )
        try:
            oof_auc = roc_auc_score(y, oof_meta_scores)
        except Exception:
            oof_auc = np.nan

        print("\n[OOF][Meta LR] Accuracy @ 0.50 = {:.4f}".format(oof_acc_default))
        print("[OOF][Meta LR] ROC-AUC = {:.4f}".format(oof_auc))

        all_results.append(
            {
                "seed": f"LR_seed={seed_lr}, XGB_seed={seed_xgb}, RF_seed={seed_rf}",
                "auc": oof_auc,
                "y": y,
                "oof_meta_scores": oof_meta_scores,
                "meta_test_scores": meta_test_scores,
            }
        )

    # Pick best by AUC
    best_result = max(all_results, key=lambda x: x["auc"])
    best_seed = best_result["seed"]
    best_auc = best_result["auc"]

    print("\n" + "=" * 90)
    print(f"Best stacked model found with {best_seed}")
    print(f"Best OOF AUC = {best_auc:.4f}")
    print("=" * 90)
    print(
        "\nReady for threshold tuning (variables: oof_meta_scores, meta_test_scores, y)"
    )

    return {
        "best_seed": best_seed,
        "best_auc": best_auc,
        "y": best_result["y"],
        "oof_meta_scores": best_result["oof_meta_scores"],
        "meta_test_scores": best_result["meta_test_scores"],
        "all_results": all_results,
    }

# ==========================================
# 3.3 Threshold tuning for stacked model
# ==========================================
import numpy as np
from sklearn.metrics import accuracy_score


def tune_stacking_threshold_mrk(
    y,
    oof_meta_scores,
    meta_test_scores,
    coarse_lo: float = 0.30,
    coarse_hi: float = 0.70,
    coarse_points: int = 121,
    fine_window: float = 0.05,
    fine_step: float = 0.001,
    verbose: bool = True,
) -> dict:
    """
    Threshold tuning for stacked model using OOF probabilities.

    Parameters
    ----------
    y : array-like
        True labels for training set (0/1).
    oof_meta_scores : array-like
        OOF probabilities from the stacked meta model (len = len(y)).
    meta_test_scores : array-like
        Test probabilities from the stacked meta model (len = n_test).
    coarse_lo, coarse_hi : float
        Range of thresholds for coarse search.
    coarse_points : int
        Number of thresholds in coarse grid.
    fine_window : float
        Half-width of fine search window around best coarse threshold.
    fine_step : float
        Step size for fine search.
    verbose : bool
        If True, prints diagnostics.

    Returns
    -------
    result : dict
        {
          "final_threshold": float,
          "final_oof_acc": float,
          "oof_labels": np.ndarray,   # labels on train using tuned thr
          "test_labels": np.ndarray,  # labels on test using tuned thr
          "baseline_acc": float,
          "coarse_best_thr": float,
          "coarse_best_acc": float,
        }
    """
    y = np.asarray(y).astype(int)
    oof = np.asarray(oof_meta_scores, dtype=float)
    te = np.asarray(meta_test_scores, dtype=float)

    if y.shape[0] != oof.shape[0]:
        raise ValueError("y and oof_meta_scores must have the same length.")

    # Baseline @ 0.50
    baseline_acc = accuracy_score(y, (oof >= 0.50).astype(int))
    if verbose:
        print(f"[Stacking][OOF] Accuracy @ 0.50 = {baseline_acc:.4f}")

    # Coarse search
    ths_coarse = np.linspace(coarse_lo, coarse_hi, coarse_points)
    accs_coarse = [accuracy_score(y, (oof >= t).astype(int)) for t in ths_coarse]
    best_idx_c = int(np.argmax(accs_coarse))
    best_thr_coarse = float(ths_coarse[best_idx_c])
    best_acc_coarse = float(accs_coarse[best_idx_c])

    if verbose:
        print(
            f"[Stacking][Search] Coarse best: thr={best_thr_coarse:.3f} "
            f"| OOF acc={best_acc_coarse:.4f}"
        )

    # Fine search around coarse best
    fine_lo = max(0.0, best_thr_coarse - fine_window)
    fine_hi = min(1.0, best_thr_coarse + fine_window)
    ths_fine = np.arange(fine_lo, fine_hi + 1e-12, fine_step)

    accs_fine = [accuracy_score(y, (oof >= t).astype(int)) for t in ths_fine]
    best_idx_f = int(np.argmax(accs_fine))
    final_thr = float(ths_fine[best_idx_f])
    final_acc = float(accs_fine[best_idx_f])

    if verbose:
        print(
            f"[Stacking][Best] Final OOF threshold = {final_thr:.3f} "
            f"| OOF Accuracy = {final_acc:.4f}"
        )

    # Labels with tuned threshold
    oof_labels = (oof >= final_thr).astype(int)
    test_labels = (te >= final_thr).astype(int)

    return {
        "final_threshold": final_thr,
        "final_oof_acc": final_acc,
        "oof_labels": oof_labels,
        "test_labels": test_labels,
        "baseline_acc": baseline_acc,
        "coarse_best_thr": best_thr_coarse,
        "coarse_best_acc": best_acc_coarse,
    }

# ### 5. Submitting Your Results
# 
# Once you have generated your `submission.csv` file, there are two primary ways to submit it to the competition.
# 
# ---
# 
# #### Method A: Submitting Directly from the Notebook
# 
# This is the standard method for code competitions. It ensures that your submission is linked to the code that produced it, which is crucial for reproducibility.
# 
# 1.  **Save Your Work:** Click the **"Save Version"** button in the top-right corner of the notebook editor.
# 2.  **Run the Notebook:** In the pop-up window, select **"Save & Run All (Commit)"** and then click the **"Save"** button. This will run your entire notebook from top to bottom and save the output, including your `submission.csv` file.
# 3.  **Go to the Viewer:** Once the save process is complete, navigate to the notebook viewer page. 
# 4.  **Submit to Competition:** In the viewer, find the **"Submit to Competition"** section. This is usually located in the header of the output section or in the vertical "..." menu on the right side of the page. Clicking the **Submit** button this will submit your generated `submission.csv` file.
# 
# After submitting, you will see your score in the **"Submit to Competition"** section or in the [Public Leaderboard](https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2025/leaderboard?).
# 
# ---
# 
# #### Method B: Manual Upload
# 
# You can also generate your predictions and submission file using any environment you prefer (this notebook, Google Colab, or your local machine).
# 
# 1.  **Generate the `submission.csv` file** using your model.
# 2.  **Download the file** to your computer.
# 3.  **Navigate to the [Leaderboard Page](https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2025/leaderboard?)** and click on the **"Submit Predictions"** button.
# 4.  **Upload Your File:** Drag and drop or select your `submission.csv` file to upload it.
# 
# This method is quick, but keep in mind that for the final evaluation, you might be required to provide the code that generated your submission.
# 
# Good luck!
