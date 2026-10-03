import numpy as np
import pandas as pd


# ============================================================
# PRODUCTION CALCULATIONS
# ============================================================

def calculate_liquid_rate(oil_rate, water_rate):

    return oil_rate + water_rate


def calculate_water_cut(oil_rate, water_rate):

    liquid = oil_rate + water_rate

    if liquid == 0:
        return 0

    return water_rate / liquid


def calculate_gor(gas_rate, oil_rate):

    if oil_rate == 0:
        return np.nan

    return gas_rate / oil_rate


def calculate_wor(water_rate, oil_rate):

    if oil_rate == 0:
        return np.nan

    return water_rate / oil_rate


# ============================================================
# DECLINE CURVE ANALYSIS
# ============================================================

def exponential_decline(t, qi, Di):

    return qi * np.exp(-Di * t)


def harmonic_decline(t, qi, Di):

    return qi / (1 + Di * t)


def hyperbolic_decline(t, qi, Di, b):

    return qi / (
        (1 + b * Di * t) ** (1 / b)
    )


def fit_exponential(time, rate):

    time = np.asarray(time)
    rate = np.asarray(rate)

    valid = (
        np.isfinite(time) &
        np.isfinite(rate) &
        (rate > 0)
    )

    time = time[valid]
    rate = rate[valid]

    if len(time) < 3:
        return None

    y = np.log(rate)

    slope, intercept = np.polyfit(
        time,
        y,
        1
    )

    Di = -slope
    qi = np.exp(intercept)

    prediction = exponential_decline(
        time,
        qi,
        Di
    )

    ss_res = np.sum(
        (rate - prediction) ** 2
    )

    ss_tot = np.sum(
        (rate - np.mean(rate)) ** 2
    )

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot != 0
        else 0
    )

    return {
        "qi": qi,
        "Di": Di,
        "r2": r2
    }


def add_engineering_columns(df):

    df = df.copy()

    if (
        "oil_rate" in df.columns and
        "water_rate" in df.columns
    ):

        df["liquid_rate"] = (
            df["oil_rate"] +
            df["water_rate"]
        )

        df["water_cut"] = (
            df["water_rate"] /
            df["liquid_rate"].replace(0, np.nan)
        )

        df["wor"] = (
            df["water_rate"] /
            df["oil_rate"].replace(0, np.nan)
        )

    if (
        "gas_rate" in df.columns and
        "oil_rate" in df.columns
    ):

        df["gor"] = (
            df["gas_rate"] /
            df["oil_rate"].replace(0, np.nan)
        )

    return df
