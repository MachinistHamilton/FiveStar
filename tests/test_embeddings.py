from unittest.mock import Mock

import numpy as np
import pytest

from core.embeddings import Embeddings
from core.errors import ModelError


def test_embedding_batches_normalize_and_reject_truncation(settings, monkeypatch):
    model = Mock()
    model.max_seq_length = 8
    model.__getitem__ = Mock(return_value=Mock(auto_model=Mock(config=Mock(_commit_hash="commit1"))))
    model.get_embedding_dimension.return_value = 2
    model.tokenizer.encode.side_effect = lambda text, **kwargs: text.split() + ["start", "end"]
    model.encode.return_value = np.array([[0.6, 0.8]])
    monkeypatch.setattr("core.embeddings.load_embedding_model", lambda *args: model)
    embeddings = Embeddings(settings)
    assert embeddings.encode(["short text"]) == [[0.6, 0.8]]
    assert model.encode.call_args.kwargs["normalize_embeddings"]
    assert model.encode.call_args.kwargs["batch_size"] == 32
    assert embeddings.encode([]) == []
    with pytest.raises(ModelError, match="exceeds"):
        embeddings.encode(["one two three four five six seven"])
    first_fingerprint = embeddings.fingerprint
    model.__getitem__.return_value.auto_model.config._commit_hash = "commit2"
    assert Embeddings(settings).fingerprint != first_fingerprint
