def test_health_reports_model_versions(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["model_versions"]) == {"expected_power", "anomaly_detector", "forecaster_1h"}


def test_sites_lists_the_fleet(client):
    sites = client.get("/sites").json()
    ids = {s["site_id"] for s in sites}
    assert {"solar-01", "wind-01"}.issubset(ids)


def test_anomaly_contract(client):
    r = client.post("/predict/anomaly", json={"site_id": "wind-02"})
    assert r.status_code == 200
    body = r.json()
    for key in ("is_anomaly", "anomaly_probability", "actual_power_kw", "expected_power_kw",
                "residual_pct", "is_curtailed", "model_version"):
        assert key in body
    assert 0.0 <= body["anomaly_probability"] <= 1.0


def test_anomaly_as_of_returns_that_point_in_time(client):
    r = client.post("/predict/anomaly", json={"site_id": "solar-01", "as_of": "2026-02-15T12:00:00"})
    assert r.status_code == 200
    assert r.json()["timestamp"] == "2026-02-15 12:00:00"


def test_unknown_site_is_404_not_500(client):
    r = client.post("/predict/anomaly", json={"site_id": "wind-99"})
    assert r.status_code == 404


def test_repeat_request_is_served_from_cache(client):
    payload = {"site_id": "solar-03", "as_of": "2026-03-01T10:00:00"}
    first = client.post("/predict/anomaly", json=payload).json()
    second = client.post("/predict/anomaly", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["anomaly_probability"] == second["anomaly_probability"]


def test_forecast_contract(client):
    body = client.post("/predict/forecast", json={"site_id": "solar-01"}).json()
    assert body["forecast_horizon_minutes"] == 60
    assert body["forecast_power_kw"] >= -50  # small negative noise tolerated, not wild values


def test_forecast_needs_history(client):
    # The very first reading has no 1-day lag history, so the forecaster must refuse cleanly.
    r = client.post("/predict/forecast", json={"site_id": "wind-01", "as_of": "2026-01-01T00:00:00"})
    assert r.status_code == 404


def test_batch_isolates_per_site_errors(client):
    out = client.post("/predict/batch", json={"site_ids": ["solar-01", "nope-01"]}).json()
    assert "anomaly_probability" in out[0]
    assert "error" in out[1]


def test_log_search_returns_ranked_results(client):
    out = client.post("/search/logs", json={"query": "gearbox bearing vibration", "k": 3}).json()
    assert len(out) == 3
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True)
    assert out[0]["event_type"] == "mechanical_wear"


def test_stream_emits_scored_events(client):
    with client.stream("GET", "/stream/telemetry/wind-01?speed=100000&tail=3") as r:
        assert r.status_code == 200
        text = "".join(chunk for chunk in r.iter_text())
    assert text.count("event: telemetry") == 3
    assert "anomaly_probability" in text


def test_top_flagged_points_at_a_high_probability_moment(client):
    body = client.get("/sites/wind-02/top-flagged").json()
    assert body["anomaly_probability"] > 0.5


def test_stream_can_replay_from_a_start_time(client):
    url = "/stream/telemetry/solar-01?speed=100000&tail=2&start=2026-03-10T11:30:00"
    with client.stream("GET", url) as r:
        text = "".join(r.iter_text())
    assert '"timestamp": "2026-03-10 11:30:00"' in text


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "PlantPulse" in r.text
