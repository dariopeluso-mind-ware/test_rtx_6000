# Code Standards

## Language & Framework Preferences

- **Primary language**: Python 3.11+
- **Batch processor**: Procedural script (`src/main.py`) — no web framework
- **HTTP client**: `httpx` (sync `Client`, not async `AsyncClient`)
- **YOLO inference**: `ultralytics` YOLO OBB
- **Barcode scanning**: `pyzbar`
- **Image processing**: `cv2` (OpenCV), `PIL`
- **Structured logging**: `structlog`

> **Note on web frameworks**: This project is a batch script, not a web service.
> If a FastAPI endpoint is added in the future, use `FastAPI` + `Pydantic v2` + async `httpx`.
> Until then, keep `main.py` procedural and synchronous.

## Type Hints

Always use type hints on all functions, no exceptions:

```python
# Good
def detect_ean_barcode_and_orientation(image_path: Path) -> tuple[str | None, int]:
    ...

# Bad
def detect_ean_barcode_and_orientation(image_path):
    ...
```

Use `type` statements for complex union types:

```python
type ResultOrError = Result | ProcessingError
```

## Docstrings

Google-style docstrings for all public functions, classes, and modules:

```python
def extract_label_region(
    image: cv2.Mat,
    obb_center_x: float,
    obb_center_y: float,
    obb_width: float,
    obb_height: float,
    obb_rotation_radians: float,
) -> cv2.Mat:
    """
    Crops a label region from an image using an oriented bounding box (OBB).

    Args:
        image: Source image as OpenCV BGR Mat (H, W, C).
        obb_center_x: OBB centroid x-coordinate (pixel units).
        obb_center_y: OBB centroid y-coordinate (pixel units).
        obb_width: OBB width along its local x-axis (pixel units).
        obb_height: OBB height along its local y-axis (pixel units).
        obb_rotation_radians: OBB rotation angle in radians (positive = CCW).

    Returns:
        The cropped label as an OpenCV BGR Mat (height × width × 3).

    Raises:
        cv2.error: If the image cannot be read or the rotation/crop parameters are invalid.
    """
```

## Style

- PEP 8 compliant
- Maximum line length: 100 characters
- Use f-strings for string formatting
- Prefer list comprehensions when readable
- Prefer `pathlib.Path` for all file paths
- No wildcard imports (`from module import *`)

## Synchronous Patterns

```python
# Good: httpx sync Client with retry
for attempt in range(1, MAX_RETRIES + 1):
    try:
        response = httpx.post(url, json=payload, timeout=httpx.Timeout(120.0))
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning("retry_attempt", attempt=attempt, error=str(exc))
        time.sleep(2 ** attempt)

# Bad: blocking synchronous call without timeout or retry
response = httpx.post(url, json=payload)
```

```python
# Good: sequential stages with clear logging
ean_code, rotation = detect_ean_barcode_and_orientation(image_path)
yolo_results = yolo_obb_model.predict(source=str(image_path), device="cpu", verbose=False)
cropped = _deskew_crop_obb(...)
transcription = _transcribe_label_image(cropped, base_url)
```

## Error Handling

- Raise specific exceptions, never generic `Exception`
- Log errors with structured context before handling
- Let exceptions propagate through business logic
- Never swallow exceptions silently
- `cv2.error` from OpenCV operations should be caught and re-raised with context

```python
# Good
try:
    cropped_label_bgr = cv2.imread(str(source_image_path))
    if cropped_label_bgr is None:
        raise cv2.error(f"cv2.imread returned None for: {source_image_path}")
except cv2.error as exc:
    logger.error("crop_failed", image=image_path.name, error=str(exc))
    raise

# Bad
try:
    ...
except Exception:
    pass
```

## Secrets Management

- Never hardcode secrets — use environment variables
- API keys, tokens, passwords must never appear in logs or error messages

```python
# Good
hf_token = os.environ.get("HF_TOKEN")

# Bad
hf_token = "hf_1234567890abcdef"
```

## File Paths

Always use `pathlib.Path` for file paths — never string concatenation:

```python
# Good
log_path = Path("output/llama_server.log")
output_dir = Path("output_test")
cropped_path = output_dir / f"crop_{image_path.name}"

# Bad
log_path = "output/llama_server.log"