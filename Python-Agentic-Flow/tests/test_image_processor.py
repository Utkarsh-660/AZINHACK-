"""Unit tests for ImageProcessor: decoding, size checks, format limits, and SSRF security."""

import base64
import io
import tempfile
from pathlib import Path
import pytest
from PIL import Image

from agentic_flow.config import AgentConfig
from agentic_flow.exceptions import ImageProcessingError, SSRFSecurityError
from agentic_flow.services.image_processor import ImageProcessor


def test_process_valid_bytes(test_config: AgentConfig, synthetic_image_bytes: bytes):
    """Verify processing valid image bytes."""
    processor = ImageProcessor(test_config)
    result = processor.process(synthetic_image_bytes, source_index=1)
    assert result.mime_type == "image/jpeg"
    assert result.width == 200
    assert result.height == 100
    assert result.source_index == 1


def test_process_valid_pil_image(test_config: AgentConfig):
    """Verify processing an already-opened PIL Image."""
    processor = ImageProcessor(test_config)
    img = Image.new("RGB", (100, 100), color=(10, 20, 30))
    result = processor.process(img)
    assert result.width == 100
    assert result.height == 100
    assert result.mime_type in ("image/png", "image/jpeg")


def test_process_valid_local_file(test_config: AgentConfig, synthetic_png_bytes: bytes):
    """Verify processing a temporary local file path."""
    processor = ImageProcessor(test_config)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(synthetic_png_bytes)
        tmp_path = Path(tmp.name)

    try:
        result = processor.process(tmp_path)
        assert result.mime_type == "image/png"
        assert result.width == 150
        assert result.source_name == tmp_path.name
    finally:
        tmp_path.unlink(missing_ok=True)


def test_process_base64_data_uri(test_config: AgentConfig, synthetic_image_bytes: bytes):
    """Verify processing a base64 Data URI."""
    processor = ImageProcessor(test_config)
    b64_str = base64.b64encode(synthetic_image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_str}"
    result = processor.process(data_uri)
    assert result.mime_type == "image/jpeg"
    assert result.width == 200


def test_reject_corrupt_bytes(test_config: AgentConfig):
    """Verify corrupt bytes are rejected with ImageProcessingError."""
    processor = ImageProcessor(test_config)
    with pytest.raises(ImageProcessingError, match="Corrupt, malformed"):
        processor.process(b"not_an_image_random_noise_123456789")


def test_reject_empty_bytes(test_config: AgentConfig):
    """Verify empty bytes are rejected with ImageProcessingError."""
    processor = ImageProcessor(test_config)
    with pytest.raises(ImageProcessingError, match="Empty image payload"):
        processor.process(b"")


def test_reject_oversized_payload(test_config: AgentConfig):
    """Verify image exceeding max_image_bytes is rejected."""
    small_config = AgentConfig(
        gemini_api_key="mock",
        max_image_bytes=500  # small limit
    )
    processor = ImageProcessor(small_config)
    with pytest.raises(ImageProcessingError, match="exceeds"):
        processor.process(b"x" * 600)


def test_reject_oversized_dimensions(test_config: AgentConfig):
    """Verify image exceeding max_image_dimension is rejected."""
    small_dim_config = AgentConfig(
        gemini_api_key="mock",
        max_image_dimension=100
    )
    processor = ImageProcessor(small_dim_config)
    large_img = Image.new("RGB", (150, 150))
    buf = io.BytesIO()
    large_img.save(buf, format="PNG")

    with pytest.raises(ImageProcessingError, match="dimensions .* exceed"):
        processor.process(buf.getvalue())


def test_ssrf_rejects_localhost(test_config: AgentConfig):
    """Verify SSRF protection rejects localhost URLs."""
    processor = ImageProcessor(test_config)
    with pytest.raises(SSRFSecurityError, match="Remote URL image fetching is disabled"):
        processor.process("http://localhost:8000/image.jpg")

    with pytest.raises(SSRFSecurityError, match="Remote URL image fetching is disabled"):
        processor.process("http://127.0.0.1:5000/test.png")


def test_ssrf_rejects_private_network(test_config: AgentConfig):
    """Verify SSRF protection rejects private IP ranges."""
    processor = ImageProcessor(test_config)
    with pytest.raises(SSRFSecurityError, match="Remote URL image fetching is disabled"):
        processor.process("http://192.168.1.50/rx.png")

    with pytest.raises(SSRFSecurityError, match="Remote URL image fetching is disabled"):
        processor.process("http://10.0.0.5/lab.png")


def test_ssrf_rejects_invalid_scheme(test_config: AgentConfig):
    """Verify SSRF protection rejects file:// or ftp:// schemes."""
    processor = ImageProcessor(test_config)
    with pytest.raises(SSRFSecurityError, match="Remote URL image fetching is disabled"):
        processor.process("file:///etc/passwd")

    with pytest.raises(SSRFSecurityError, match="Remote URL image fetching is disabled"):
        processor.process("ftp://internal.server/scan.png")


def test_base64_data_uri_mime_mismatch_rejected(test_config: AgentConfig, synthetic_image_bytes: bytes):
    """Verify that declaring image/png for a JPEG payload raises ImageProcessingError."""
    processor = ImageProcessor(test_config)
    b64_str = base64.b64encode(synthetic_image_bytes).decode("utf-8")
    # Lie about the format: declare PNG but give JPEG payload
    mismatched_uri = f"data:image/png;base64,{b64_str}"

    with pytest.raises(ImageProcessingError, match="declared MIME type .* does not match"):
        processor.process(mismatched_uri)


def test_base64_strict_invalid_padding_rejected(test_config: AgentConfig):
    """Verify that malformed base64 with invalid characters/padding is rejected."""
    processor = ImageProcessor(test_config)
    corrupt_uri = "data:image/jpeg;base64,!!!invalid_base64_symbols@@@"

    with pytest.raises(ImageProcessingError, match="Base64 decoding failed"):
        processor.process(corrupt_uri)


def test_error_sanitization_does_not_leak_full_path(test_config: AgentConfig):
    """Verify that file-not-found errors do not leak internal directory paths."""
    processor = ImageProcessor(test_config)
    sensitive_path = "C:/SecretDir/PrivateUser/Subfolder/missing_scan.png"

    with pytest.raises(ImageProcessingError) as exc_info:
        processor.process(sensitive_path)

    err_str = str(exc_info.value)
    # The error should contain the filename, not the full secret folder hierarchy
    assert "missing_scan.png" in err_str
    assert "SecretDir" not in err_str
