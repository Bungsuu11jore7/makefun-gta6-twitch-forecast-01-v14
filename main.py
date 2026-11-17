"""GTA VI Twitch CCU forecast, v14.

Forecasts Twitch concurrent viewers for Grand Theft Auto VI on launch
day (hours 2..24, five quantiles each) from the hour-1 print, the
pre-release hourly history, and a pool of past big-launch curves.

Method (v13 centre + v14 changes):
  * Ensemble centre: three predictors over comparable launch curves
    (kernel mean at two bandwidths + local-linear), blended 0.3/0.2/0.5,
    aligned by unlock hour, weighted by launch-size similarity and a
    weekday factor. Curves are log viewers relative to the launch-day
    hour-0 print, indexed by UTC offset.
  * Diurnal corrections from the target's own pre-release diurnal
    profile (mean of log viewers by UTC hour, smoothed twice):
    early-hours dip deepening (convex in the early fall F), mid-hours
    dip + inflation term + mid echo, an evening lift for smaller
    launches, a rolling-only early size gate, and a flat-diurnal
    level lift.
  * v14 changes: FLAT_COEF 0.05 -> 0.09; SPREAD_L1 9.0 -> 10.5;
    ECHO_COEF 0.5 -> 0.38; the mid-hours echo is now scaled by the
    same convex factor as the early echo, but only when the early
    fall F lies in [0.6, 1.1] (outside it the echo stays unscaled);
    the first three hours' SIGMA is narrowed x0.9.
  * Quantiles: exp(logp1 + centre + Z * SIGMA * spread_mult(logp1)),
    rounded half-up to 2 decimals, clamped to [0.01, 1e9], monotone.

Standard library only. Reads $MAKEFUN_INPUT_DIR (else /input),
writes $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA = [0.26, 0.25, 0.3, 0.21, 0.18, 0.16, 0.18, 0.2, 0.22, 0.29,
         0.32, 0.32, 0.25, 0.24, 0.25, 0.26, 0.23, 0.17, 0.15, 0.15,
         0.15, 0.15, 0.15]
# Early hours (2-4) have the most accurate medians: narrow their spread.
EARLY_SIGMA_MULT = 0.9

BIG_THRESHOLD = 10.0
BW_MEAN1_BIG = 1.0
BW_MEAN1_SMALL = 1.5
BW_MEAN2 = 2.5
BW_LOCLIN_BIG = 1.8
BW_LOCLIN_SMALL = 1.2
W_MEAN1 = 0.3
W_MEAN2 = 0.2
W_LOCLIN = 0.5
WEEKDAY_FACTOR = 0.75
LAM_LOCLIN = 0.6
SLOPE_DAMP = 0.3

DIP_COEF = 0.7
ECHO_COEF = 0.38
ECHO_EXP = 0.5
ECHO_F0 = 0.5
ECHO_CAP = 2.0
# Mid-hours echo scaling applies only inside this early-fall window.
MID_ECHO_F_LO = 0.6
MID_ECHO_F_HI = 1.1
MID_ECHO_CAP = 2.0
EARLY_IDX = (0, 1, 2)
MID_IDX = (9, 10, 11, 12, 13)
EVENING_IDX = (15, 16, 17, 18, 19, 20, 21, 22)
EVENING_COEF = 0.2
EVENING_REF = 11.0
INFL_COEF = 0.02
INFL_CAP = 10.0
INFL_FALLBACK = 8.0
INFL_GATE_DENOM = 0.1
EARLY_SIZE_COEF = 0.2
SIZE_REF = 11.5
FLAT_COEF = 0.09
FLAT_REF = 1.8
SPREAD_SMALL = 1.15
SPREAD_BIG = 1.0
SPREAD_L1 = 10.5
SPREAD_L2 = 12.0

SIMULTANEOUS_UNLOCK = {
    "Cyberpunk 2077": 0,
    "Black Myth: Wukong": 2,
    "Battlefield 6": 14,
    "Hollow Knight: Silksong": 14,
    "007 First Light": 14,
}


def round2(x):
    """Round half up to 2 decimals; whole numbers come back as ints."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)


def dst_active(day):
    """US daylight saving in effect on `day` (a date). DST runs from
    the second Sunday of March to the first Sunday of November; a
    launch ON the second Sunday of March counts as DST, one ON the
    first Sunday of November does not."""
    mar1 = datetime(day.year, 3, 1, tzinfo=timezone.utc).date()
    start = mar1.replace(day=1 + ((6 - mar1.weekday()) % 7) + 7)
    nov1 = datetime(day.year, 11, 1, tzinfo=timezone.utc).date()
    end = nov1.replace(day=1 + ((6 - nov1.weekday()) % 7))
    return start <= day < end


def smooth3(values):
    """Three-point moving average; the edges average two points."""
    out = []
    for i in range(len(values)):
        lo = max(0, i - 1)
        hi = min(len(values), i + 2)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def diurnal_profile(history_hourly, day_start):
    """Mean of log viewers by UTC hour over the pre-release history,
    smoothed twice. None unless every UTC hour has >= 3 points."""
    if not history_hourly:
        return None
    ds = parse_ts(day_start)
    sums = [0.0] * 24
    counts = [0] * 24
    for point in history_hourly:
        if point.get("viewers") is None:
            continue
        dt = parse_ts(point["from"])
        off = int((dt - ds).total_seconds() // 3600)
        if -336 <= off < 0:
            sums[dt.hour] += math.log(max(1, point["viewers"]))
            counts[dt.hour] += 1
    if any(c < 3 for c in counts):
        return None
    profile = [sums[h] / counts[h] for h in range(24)]
    return smooth3(smooth3(profile))


def rolling_unlock(day_start):
    """US East Coast midnight unlock hour (UTC) for a rolling launch."""
    return 4 if dst_active(parse_ts(day_start).date()) else 5


def target_unlock(case):
    return rolling_unlock(case["day_start"])


def comp_unlock(comp):
    """Unlock UTC hour for a comparable launch, or None to exclude it
    from the pool (mixed console+PC-simultaneous launches, and
    simultaneous launches whose unlock hour we do not know). The
    exclusion strings are checked BEFORE the simultaneous map, so a
    mapped launch that is early-access-contaminated is still out."""
    unlock = comp["unlock"]
    if "early-access-contaminated" in unlock:
        return None
    if "Tier: weak" in unlock:
        return None
    if "Tier: target-history" in unlock:
        return None
    if comp["name"] in SIMULTANEOUS_UNLOCK:
        return SIMULTANEOUS_UNLOCK[comp["name"]]
    text = unlock.lower()
    if "simultaneous" in text or "single global" in text:
        return None
    if "mixed" in text:
        return None
    return rolling_unlock(comp["day_start"])


def build_pool(case):
    """Comparable launches as log-ratio curves (29 entries each):
    entry j is log(viewers at UTC offset clamp(unlock + j - 5, 0, 23)
    / viewers at UTC offset 0). A comparable is dropped when any
    launch-day hour 0..23 is missing or below 1 viewer."""
    pool = []
    for comp in case.get("comparables", []):
        unlock = comp_unlock(comp)
        if unlock is None:
            continue
        start = parse_ts(comp["day_start"])
        by_offset = {}
        by_offset_ch = {}
        pre = []
        for point in comp["hourly"]:
            off = int((parse_ts(point["from"]) - start).total_seconds()
                      // 3600)
            by_offset[off] = point["viewers"]
            if point.get("channels"):
                by_offset_ch[off] = point["channels"]
            if off < 0:
                pre.append(point)
        if any(o not in by_offset or by_offset[o] < 1
               for o in range(24)):
            continue
        v0 = max(1, by_offset[0])
        logp1 = math.log(v0)
        ch0 = max(1, by_offset_ch.get(0, 1))
        logch = math.log(ch0)
        curve = []
        for j in range(29):
            off = min(23, max(0, unlock + j - 5))
            curve.append(math.log(max(1, by_offset[off]) / v0))
        pool.append({"name": comp["name"], "logp1": logp1,
                     "logch": logch, "logvpc": logp1 - logch,
                     "curve": curve, "unlock": unlock, "pre": pre,
                     "weekend": parse_ts(comp["day_start"]).weekday() >= 5,
                     "day_start": comp["day_start"]})
    return pool


def weekday_factor(comp, target_weekend):
    """Down-weight a comparable whose weekend status differs from the
    target's (0.75); 1.0 when the target's weekend status is unknown."""
    if target_weekend is None:
        return 1.0
    return 1.0 if comp["weekend"] == target_weekend else WEEKDAY_FACTOR


def kernel_weight(logp1, lp, bw):
    return math.exp(-0.5 * ((logp1 - lp) / bw) ** 2)


def kernel_mean_predict(pool, pos, lp, bw, target_weekend=None):
    """Kernel-weighted mean of the comparables' log-ratio at `pos`."""
    num = 0.0
    den = 0.0
    for comp in pool:
        w = kernel_weight(comp["logp1"], lp, bw) * weekday_factor(
            comp, target_weekend)
        num += w * comp["curve"][pos]
        den += w
    if den <= 0:
        if not pool:
            return 0.0
        return sum(c["curve"][pos] for c in pool) / len(pool)
    return num / den


def loclin_predict(pool, pos, lp, bw, target_weekend=None):
    """Local-linear prediction at log-size lp: ridge-regularised slope
    (Sxx + 0.6), with the slope damping applied to the extrapolation
    distance, not to the slope itself. Falls back to the plain mean
    when every weight is zero."""
    if not pool:
        return 0.0
    num_w = 0.0
    mx = 0.0
    my = 0.0
    terms = []
    for comp in pool:
        w = kernel_weight(comp["logp1"], lp, bw) * weekday_factor(
            comp, target_weekend)
        terms.append((w, comp["logp1"], comp["curve"][pos]))
        num_w += w
        mx += w * comp["logp1"]
        my += w * comp["curve"][pos]
    if num_w <= 0:
        return sum(c["curve"][pos] for c in pool) / len(pool)
    mx /= num_w
    my /= num_w
    sxx = 0.0
    sxy = 0.0
    for w, x, y in terms:
        sxx += w * (x - mx) * (x - mx)
        sxy += w * (x - mx) * (y - my)
    slope = sxy / (sxx + LAM_LOCLIN)
    lo = min(c["logp1"] for c in pool)
    hi = max(c["logp1"] for c in pool)
    clamped = min(hi, max(lo, lp))
    dx = (clamped - mx) + SLOPE_DAMP * (lp - clamped)
    return my + slope * dx


def ensemble_centre(pool, lp, unlock, bw_mean1, bw_loclin,
                    target_weekend=None):
    """Blended centre (23 log ratios, hours 2..24): forecast hour h
    reads curve position h + 4 - unlock, clamped to 0..28."""
    centre = []
    for h in range(2, 25):
        pos = min(max(h + 4 - unlock, 0), 28)
        mean1 = kernel_mean_predict(pool, pos, lp, bw_mean1,
                                    target_weekend)
        mean2 = kernel_mean_predict(pool, pos, lp, BW_MEAN2,
                                    target_weekend)
        loclin = loclin_predict(pool, pos, lp, bw_loclin,
                                target_weekend)
        centre.append(W_MEAN1 * mean1 + W_MEAN2 * mean2
                      + W_LOCLIN * loclin)
    return centre


def dip_corrections(centre, logp1, diurnal, rolling=True):
    """Diurnal-driven corrections, added onto the ensemble centre.

    dip[i] = diurnal[i+1] - diurnal[0]. Early hours (2-4) get the dip
    deepening term, scaled convexly in the early fall F, minus the
    rolling-only size gate. Mid hours (11-15) get their own dip term,
    minus an inflation term gated by the mid dip, plus the mid echo
    (0.38 x dip_early), where dip_early is the mean of the negative
    parts over the early set. The mid echo is scaled by the same
    convex factor as the early echo, but only when F lies in
    [0.6, 1.1]. Evening hours (17-24) get a lift that shrinks with
    launch size. Every hour gets the flat-diurnal level lift. With
    no diurnal only the early size gate applies.
    """
    out = list(centre)
    if diurnal is None:
        dip = [0.0] * 23
        infl = INFL_FALLBACK
        echo_mult = 1.0
        lift = 0.0
        mid_mult = 1.0
    else:
        dip = [diurnal[i + 1] - diurnal[0] for i in range(23)]
        infl = min(logp1 - diurnal[0], INFL_CAP)
        fall = diurnal[0] - min(diurnal[0:5])
        echo_mult = min((max(fall, 0.0) / ECHO_F0) ** ECHO_EXP,
                        ECHO_CAP)
        if MID_ECHO_F_LO <= fall <= MID_ECHO_F_HI:
            mid_mult = min((fall / ECHO_F0) ** ECHO_EXP, MID_ECHO_CAP)
        else:
            mid_mult = 1.0
        span = max(diurnal) - min(diurnal)
        lift = FLAT_COEF * max(0.0, (FLAT_REF - span) / FLAT_REF)
    dip_early = sum(min(0.0, dip[i]) for i in EARLY_IDX) / len(EARLY_IDX)
    dip_mid = sum(min(0.0, dip[i]) for i in MID_IDX) / len(MID_IDX)
    gate_mid = min(1.0, max(0.0, -dip_mid / INFL_GATE_DENOM))
    if rolling:
        size_gate = min(1.0, max(0.0, logp1 - SIZE_REF))
    else:
        size_gate = 0.0
    evening_lift = EVENING_COEF * max(0.0, EVENING_REF - logp1)
    for i in EARLY_IDX:
        out[i] += DIP_COEF * min(0.0, dip[i]) * echo_mult \
            - EARLY_SIZE_COEF * size_gate
    for i in MID_IDX:
        out[i] += DIP_COEF * min(0.0, dip[i]) \
            - INFL_COEF * infl * gate_mid + ECHO_COEF * dip_early \
            * mid_mult
    for i in EVENING_IDX:
        out[i] += evening_lift
    for i in range(23):
        out[i] += lift
    return out


def full_centre(pool, logp1, unlock, pre_points, day_start,
                target_weekend=None, rolling=True,
                bw_mean1=BW_MEAN1_SMALL, bw_loclin=BW_LOCLIN_BIG):
    """Ensemble centre plus dip corrections for one launch. The
    diurnal comes from the launch's own pre-release hourly points
    (history_hourly for the target, the comparable's own pre-release
    hourly rows for a comparable)."""
    diurnal = diurnal_profile(pre_points, day_start)
    centre = ensemble_centre(pool, logp1, unlock, bw_mean1, bw_loclin,
                             target_weekend)
    return dip_corrections(centre, logp1, diurnal, rolling)


def spread_mult(logp1):
    """Quantile spread multiplier by launch size: 1.15 at logp1 <= 10.5,
    1.0 at logp1 >= 12, linear between."""
    if logp1 <= SPREAD_L1:
        return SPREAD_SMALL
    if logp1 >= SPREAD_L2:
        return SPREAD_BIG
    frac = (logp1 - SPREAD_L1) / (SPREAD_L2 - SPREAD_L1)
    return SPREAD_SMALL + frac * (SPREAD_BIG - SPREAD_SMALL)


def forecast_case(case, case_id):
    if case["levels"] != LEVELS:
        raise ValueError(f"{case_id} asks for levels "
                         f"{case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{case_id} has no hour-1 print")
    p1 = max(1, first["viewers"])
    logp1 = math.log(p1)
    pool = build_pool(case)
    unlock = target_unlock(case)
    weekend = parse_ts(case["day_start"]).weekday() >= 5
    # A3 size-adaptive bandwidths: big launches get a narrow mean
    # bandwidth and a wide local-linear one; small launches the
    # reverse.
    big = logp1 >= BIG_THRESHOLD
    bw_mean1 = BW_MEAN1_BIG if big else BW_MEAN1_SMALL
    bw_loclin = BW_LOCLIN_BIG if big else BW_LOCLIN_SMALL
    # The early-hours size gate models the pre-unlock lull, so it
    # applies only to rolling-midnight launches. A launch named in
    # the simultaneous-unlock map unlocked everywhere at once.
    name = case.get("game", {}).get("name", "")
    rolling = name not in SIMULTANEOUS_UNLOCK
    if pool:
        centre = full_centre(pool, logp1, unlock,
                             case.get("history_hourly") or [],
                             case["day_start"], weekend, rolling,
                             bw_mean1, bw_loclin)
    else:
        centre = [0.0] * 23
    mult = spread_mult(logp1)
    rows = []
    for i, h in enumerate(range(2, 25)):
        mu = logp1 + centre[i]
        sigma = SIGMA[i] * mult
        if i < 3:
            sigma *= EARLY_SIGMA_MULT
        qs = [round2(math.exp(mu + Z[j] * sigma)) for j in range(5)]
        qs = [min(1e9, max(0.01, q)) for q in qs]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
