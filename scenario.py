def production_scenario(
    current_rate,
    rate_change_percent
):

    new_rate = (
        current_rate *
        (1 + rate_change_percent / 100)
    )

    incremental_rate = (
        new_rate - current_rate
    )

    return {
        "current_rate": current_rate,
        "new_rate": new_rate,
        "incremental_rate": incremental_rate
    }


def water_cut_scenario(
    oil_rate,
    water_cut_change
):

    current_water_cut = water_cut_change[
        "current"
    ]

    new_water_cut = water_cut_change[
        "new"
    ]

    current_water_rate = (
        oil_rate *
        current_water_cut /
        max(1 - current_water_cut, 0.001)
    )

    new_water_rate = (
        oil_rate *
        new_water_cut /
        max(1 - new_water_cut, 0.001)
    )

    return {
        "current_water_rate":
            current_water_rate,

        "new_water_rate":
            new_water_rate,

        "water_rate_change":
            new_water_rate -
            current_water_rate
    }


def intervention_scenario(
    current_rate,
    expected_gain_percent
):

    expected_rate = (
        current_rate *
        (1 + expected_gain_percent / 100)
    )

    return {
        "current_rate": current_rate,
        "expected_rate": expected_rate,
        "gain": expected_rate - current_rate
    }
