import pytest
from pydantic import ValidationError

from config.settings import PROJECT_ROOT, Settings


def test_settings_paths_and_budget():
    settings = Settings(_env_file=None, data_dir="storage")
    assert settings.data_dir == PROJECT_ROOT / "storage"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, context_window=2048, max_output_tokens=1024)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, evidence_count=20, candidate_count=5)


@pytest.mark.parametrize("url", [
    "https://api.example.com", "http://127.0.0.1.evil.test",
    "http://user:pass@localhost:11434", "http://localhost/path",
    "http://localhost?remote=true",
])
def test_no_remote_inference(url):
    with pytest.raises(ValidationError, match="loopback"):
        Settings(_env_file=None, ollama_url=url)
