from ads_scholargraph.config import Settings


def test_settings_defaults(monkeypatch) -> None:
    monkeypatch.setenv("ADS_API_TOKEN", "demo-token")
    monkeypatch.setenv("NEO4J_PASSWORD", "demo-password")

    settings = Settings()

    assert settings.NEO4J_URI == "bolt://localhost:7687"
    assert settings.NEO4J_USER == "neo4j"
