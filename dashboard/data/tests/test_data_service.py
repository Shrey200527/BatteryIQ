from dashboard.data.data_service import get_battery_history


def test_battery_history_contains_dashboard_features():
    history = get_battery_history("B0005")

    assert len(history) == 168

    first = history[0]

    assert first["battery_id"] == "B0005"
    assert first["cycle"] == 1
    assert "capacity" in first
    assert "soh" in first
    assert "voltage_mean" in first
    assert "current_mean" in first
    assert "temperature_mean" in first


def test_battery_history_is_cycle_ordered():
    history = get_battery_history("B0005")

    cycles = [row["cycle"] for row in history]

    assert cycles == sorted(cycles)
    assert cycles == list(range(1, 169))


def test_b0007_history_is_available_but_rul_is_censored():
    history = get_battery_history("B0007")

    assert len(history) == 168
    assert history[0]["cycle"] == 1
    assert history[-1]["cycle"] == 168