# Testing Strategy

## Current State

**There is no test suite yet.** The `./test/` directory contains input images for batch
processing, not test cases. A smoke-test suite should be added alongside the existing
pipeline.

## Test Pyramid

```
        /\
       /  \    E2E Tests (smoke — process 1-2 known images)
      /----\
     /      \   Unit Tests (pure functions: barcode decode, OBB crop math, report gen)
    /--------\
   /          \  Integration Tests (mock httpx, mock YOLO predict)
  /____________\
```

## Framework & Tools

- **Framework**: pytest
- **HTTP testing**: `httpx` with `respx` or `httpx_mock` for route mocking
- **YOLO mocking**: `unittest.mock.MagicMock` patching `ultralytics.YOLO.predict`
- **Fixtures**: `conftest.py` with shared fixtures

## Unit Tests

Test individual components in isolation without GPU or network:

```python
# tests/test_barcode.py
def test_ean13_returns_code_and_rotation():
    from pathlib import Path
    # Load a known test image with an EAN-13 barcode
    ean, rot = detect_ean_barcode_and_orientation(Path("data/test_images/ean13_test.jpg"))
    assert ean == "8001234567890"
    assert rot == 0

def test_no_barcode_returns_none():
    ean, rot = detect_ean_barcode_and_orientation(Path("data/test_images/plain.jpg"))
    assert ean is None
    assert rot == 0

# tests/test_crop.py
def test_crop_returns_label_region():
    ...

def test_out_of_bounds_slice_clipped():
    ...
```

## Integration Tests

Test service interactions with mocked backends:

```python
# tests/test_transcription.py
def test_transcription_returns_text(mocker):
    mocker.patch("httpx.Client.post", return_value=MockResponse(json={
        "choices": [{"message": {"content": "Acqua minerale"}}]
    }))
    result = _transcribe_label_image(Path("crop_test.jpg"), "http://localhost:8080/v1")
    assert "Acqua" in result
```

## End-to-End Smoke Test

Process 1-2 images from `etichette_esempio/` through the full pipeline:

```python
# tests/test_pipeline.py
def test_smoke_pipeline_produces_markdown():
    # Run main() on a small subset (mock llama-server or use real if available)
    # Verify output_test/crops/ contains expected files
    # Verify mocr_batch_results.md has valid structure
    ...
```

## Mocking Strategy

- Mock all external HTTP services (llama-server) with `respx` or `httpx_mock`
- Never make real network calls in CI/unit tests
- Mock `YOLO.predict` to avoid GPU dependency in unit tests
- Use pre-cached GGUF weights path for integration tests if llama-server is available

## Test Images

Store test images in `data/test_images/`:
- Known label photos with expected outputs
- Edge cases: blurry image, no label found, small label, multiple labels

## CI Considerations

- Tests run without GPU (mock YOLO, mock HTTP)
- E2E/smoke tests require GPU and llama-server running (manual or nightly)
- Coverage target: ≥80% for pipeline modules