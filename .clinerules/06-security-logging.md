# Security & Logging

## Secrets Management

- **Never hardcode secrets** — use environment variables
- All secrets go in `.env` (gitignored) or environment at runtime
- API keys, tokens, passwords must never appear in logs or error messages

```python
# Good
api_key = os.environ["OCR_SERVICE_API_KEY"]

# Bad
api_key = "sk-1234567890abcdef"
```

## Structured Logging (structlog)

Use `structlog` for all logging. Log with key=value pairs, not string interpolation:

```python
# Good
log.info("processing_image", image_id=image_id, size_bytes=len(image_bytes))

# Bad
log.info(f"Processing image {image_id} of size {len(image_bytes)} bytes")
```

Required log context:
- `request_id` for every request
- `stage` (detection/ocr/llm) for pipeline steps
- `duration_ms` for timing
- `error_type`, `error_detail` on failures

Never log:
- API keys or tokens
- Raw image data
- PII (personally identifiable information)
- Full base64-encoded images in logs

## Input Validation

Validate all external input with Pydantic models or explicit checks:
- File uploads: check magic bytes, not just extension (when applicable)
- URL parameters: validate format and bounds
- JSON payloads: enforce schema strictly

```python
# Good — validate input image path
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}

if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
    raise ValueError(f"Unsupported file extension: {file_path.suffix}")
```

## Error Handling Best Practices

- Catch exceptions at **boundaries** (adapter layer)
- Log errors with structured context **before** handling
- Let exceptions propagate through business logic
- Never swallow exceptions silently

```python
# Good — log then handle
try:
    transcription = _transcribe_label_image(cropped_path, base_url)
except RuntimeError as exc:
    log.error("transcription_failed", stage="ocr", error_type=type(exc).__name__)
    raise

# Bad — silent swallow
try:
    transcription = _transcribe_label_image(cropped_path, base_url)
except Exception:
    pass
```

## llama-server Security

- llama-server listens on `localhost:8080` only (not exposed externally)
- GGUF weights downloaded via `huggingface_hub` from authenticated HF Hub
- No remote code execution attack surface since the binary runs locally