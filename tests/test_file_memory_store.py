"""Tests for FileMemoryStore."""

import pytest
from pathlib import Path
import tempfile
import shutil

from game.memory.file_store import FileMemoryStore
from game.memory.entities import MemoryDocument


@pytest.fixture
def temp_dir():
    dir_path = Path(tempfile.mkdtemp())
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def store(temp_dir):
    return FileMemoryStore(base_dir=temp_dir)


class TestFileMemoryStorePutGet:
    def test_put_and_get_document(self, store):
        doc = store.put(
            namespace="test",
            key="doc1",
            value={"content": "Hello World"},
            metadata={"type": "greeting"},
        )

        assert doc.namespace == "test"
        assert doc.key == "doc1"
        assert doc.value["content"] == "Hello World"

        retrieved = store.get("test", "doc1")
        assert retrieved is not None
        assert retrieved.value["content"] == "Hello World"
        assert retrieved.metadata["type"] == "greeting"

    def test_put_with_tuple_namespace(self, store):
        doc = store.put(
            namespace=("user", "123", "memories"),
            key="mem1",
            value={"text": "I like pizza"},
        )

        assert doc.namespace == "user:123:memories"

        retrieved = store.get(("user", "123", "memories"), "mem1")
        assert retrieved is not None
        assert retrieved.value["text"] == "I like pizza"

    def test_get_nonexistent_returns_none(self, store):
        result = store.get("nonexistent", "key")
        assert result is None

    def test_put_updates_existing(self, store):
        store.put("test", "doc1", {"version": 1})
        store.put("test", "doc1", {"version": 2})

        retrieved = store.get("test", "doc1")
        assert retrieved.value["version"] == 2

    def test_preserves_created_at_on_update(self, store):
        first = store.put("test", "doc1", {"v": 1})
        original_created = first.created_at

        updated = store.put("test", "doc1", {"v": 2})

        assert updated.created_at == original_created
        assert updated.updated_at >= original_created


class TestFileMemoryStoreDelete:
    def test_delete_existing(self, store):
        store.put("test", "doc1", {"data": "value"})

        result = store.delete("test", "doc1")
        assert result is True

        assert store.get("test", "doc1") is None

    def test_delete_nonexistent(self, store):
        result = store.delete("test", "nonexistent")
        assert result is False


class TestFileMemoryStoreSearch:
    def test_search_with_filter(self, store):
        store.put("ns", "doc1", {"type": "a", "content": "first"}, metadata={"type": "a"})
        store.put("ns", "doc2", {"type": "b", "content": "second"}, metadata={"type": "b"})
        store.put("ns", "doc3", {"type": "a", "content": "third"}, metadata={"type": "a"})

        results = store.search("ns", filter={"type": "a"})

        assert len(results) == 2
        keys = {r.key for r in results}
        assert keys == {"doc1", "doc3"}

    def test_search_with_query(self, store):
        store.put("ns", "doc1", {"text": "I love pizza and pasta"})
        store.put("ns", "doc2", {"text": "I enjoy hiking"})
        store.put("ns", "doc3", {"text": "Pizza is my favorite food"})

        results = store.search("ns", query="pizza food")

        assert len(results) > 0
        assert results[0].key in ["doc1", "doc3"]

    def test_search_with_limit(self, store):
        for i in range(10):
            store.put("ns", f"doc{i}", {"index": i})

        results = store.search("ns", limit=3)
        assert len(results) == 3

    def test_search_empty_namespace(self, store):
        results = store.search("nonexistent")
        assert results == []


class TestFileMemoryStoreNamespaces:
    def test_list_namespaces(self, store):
        store.put("ns1", "doc1", {"data": 1})
        store.put("ns2", "doc1", {"data": 2})
        store.put("session:abc", "event", {"data": 3})

        namespaces = store.list_namespaces()

        assert "ns1" in namespaces
        assert "ns2" in namespaces
        assert "session:abc" in namespaces

    def test_list_keys(self, store):
        store.put("test", "key1", {"a": 1})
        store.put("test", "key2", {"b": 2})
        store.put("test", "key3", {"c": 3})

        keys = store.list_keys("test")

        assert set(keys) == {"key1", "key2", "key3"}

    def test_list_keys_empty_namespace(self, store):
        keys = store.list_keys("nonexistent")
        assert keys == []


class TestFileMemoryStoreBatch:
    def test_batch_put(self, store):
        docs = [
            MemoryDocument(namespace="batch", key="d1", value={"n": 1}),
            MemoryDocument(namespace="batch", key="d2", value={"n": 2}),
            MemoryDocument(namespace="batch", key="d3", value={"n": 3}),
        ]

        results = store.batch_put(docs)

        assert len(results) == 3
        assert store.get("batch", "d1") is not None
        assert store.get("batch", "d2") is not None
        assert store.get("batch", "d3") is not None

    def test_batch_get(self, store):
        store.put("ns", "k1", {"v": 1})
        store.put("ns", "k2", {"v": 2})

        results = store.batch_get([("ns", "k1"), ("ns", "k2"), ("ns", "missing")])

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is not None
        assert results[2] is None


class TestFileMemoryStoreClear:
    def test_clear_namespace(self, store):
        store.put("clear_test", "doc1", {"a": 1})
        store.put("clear_test", "doc2", {"b": 2})
        store.put("other", "doc1", {"c": 3})

        count = store.clear_namespace("clear_test")

        assert count == 2
        assert store.get("clear_test", "doc1") is None
        assert store.get("clear_test", "doc2") is None
        assert store.get("other", "doc1") is not None


class TestFileMemoryStoreAppend:
    def test_append_to_namespace(self, store):
        doc1 = store.append_to_namespace("append_ns", {"event": "first"})
        doc2 = store.append_to_namespace("append_ns", {"event": "second"})
        doc3 = store.append_to_namespace("append_ns", {"event": "third"})

        assert doc1.key == "item_1"
        assert doc2.key == "item_2"
        assert doc3.key == "item_3"

        docs = store.get_all_in_namespace("append_ns")
        assert len(docs) == 3

    def test_append_with_custom_prefix(self, store):
        doc1 = store.append_to_namespace("events", {"data": 1}, key_prefix="event")
        doc2 = store.append_to_namespace("events", {"data": 2}, key_prefix="event")

        assert doc1.key == "event_1"
        assert doc2.key == "event_2"
