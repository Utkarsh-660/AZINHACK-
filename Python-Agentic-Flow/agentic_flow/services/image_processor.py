"""Image processing and validation service.

Supports local file paths, raw bytes, PIL Images, and strictly validated Base64.
Remote URL fetching is explicitly disabled for security to eliminate SSRF and DNS-rebinding risks.
"""

import base64
import io
import re
from pathlib import Path
from typing import Optional, Tuple, Union
from PIL import Image, UnidentifiedImageError

from agentic_flow.config import AgentConfig, get_default_config
from agentic_flow.exceptions import ImageProcessingError, SSRFSecurityError


class ProcessedImage:
    """Safe, validated image representation ready for multimodal inference."""

    def __init__(
        self,
        image_bytes: bytes,
        pil_image: Image.Image,
        mime_type: str,
        width: int,
        height: int,
        source_name: str,
        source_index: int = 1
    ):
        self.image_bytes = image_bytes
        self.pil_image = pil_image
        self.mime_type = mime_type
        self.width = width
        self.height = height
        self.source_name = source_name
        self.source_index = source_index


class ImageProcessor:
    """Validates, decodes, and sanitizes images from local file, byte, PIL, and Base64 inputs."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or get_default_config()

    def process(
        self,
        image_input: Union[str, bytes, Path, Image.Image],
        source_index: int = 1,
        source_name_override: Optional[str] = None
    ) -> ProcessedImage:
        """Decode and validate an image from local file path, raw bytes, base64, or PIL Image."""
        if image_input is None:
            raise ImageProcessingError("Image input cannot be None.")

        source_name = source_name_override or "image"
        raw_bytes: Optional[bytes] = None
        declared_mime: Optional[str] = None

        # 1. PIL Image
        if isinstance(image_input, Image.Image):
            pil_img = image_input
            buffer = io.BytesIO()
            format_name = pil_img.format or "PNG"
            try:
                pil_img.save(buffer, format=format_name)
                raw_bytes = buffer.getvalue()
            except Exception:
                pil_img.save(buffer, format="PNG")
                raw_bytes = buffer.getvalue()
            source_name = source_name_override or f"pil_image_{source_index}"

        # 2. Path or string representation
        elif isinstance(image_input, (Path, str)):
            input_str = str(image_input).strip()

            # Reject remote URLs immediately to eliminate SSRF / DNS rebinding risks
            if "://" in input_str or input_str.startswith(("http://", "https://", "ftp://", "file://")):
                raise SSRFSecurityError(
                    "Remote URL image fetching is disabled for security. "
                    "Please provide a local file path, raw image bytes, base64 data, or a PIL Image."
                )

            # Base64 Data URI
            if input_str.startswith("data:image/") and ";base64," in input_str:
                raw_bytes, declared_mime = self._decode_base64_data_uri(input_str)
                source_name = source_name_override or f"base64_image_{source_index}"

            # Local file path or raw base64 string
            else:
                path_obj = Path(input_str)
                if path_obj.exists() and path_obj.is_file():
                    if path_obj.stat().st_size > self.config.max_image_bytes:
                        raise ImageProcessingError(
                            f"Image file exceeds maximum allowed size ({self.config.max_image_bytes} bytes): {path_obj.name}"
                        )
                    raw_bytes = path_obj.read_bytes()
                    source_name = source_name_override or path_obj.name
                else:
                    # Check if string is raw base64 data
                    if len(input_str) > 50 and not any(c in input_str for c in "\n\r\\/"):
                        try:
                            raw_bytes = base64.b64decode(input_str, validate=True)
                            source_name = source_name_override or f"base64_image_{source_index}"
                        except Exception:
                            raise ImageProcessingError(f"Image file not found: {path_obj.name}")
                    else:
                        # Sanitize error to avoid leaking directory hierarchy
                        raise ImageProcessingError(f"Image file not found: {path_obj.name}")

        # 3. Raw Bytes
        elif isinstance(image_input, bytes):
            raw_bytes = image_input
            source_name = source_name_override or f"bytes_image_{source_index}"

        else:
            raise ImageProcessingError(f"Unsupported image input type: {type(image_input).__name__}")

        if not raw_bytes or len(raw_bytes) == 0:
            raise ImageProcessingError("Empty image payload provided.")

        if len(raw_bytes) > self.config.max_image_bytes:
            raise ImageProcessingError(
                f"Image payload size ({len(raw_bytes)} bytes) exceeds limit of {self.config.max_image_bytes} bytes."
            )

        # Validate image decoding and format
        pil_img, mime_type = self._validate_and_decode(raw_bytes, declared_mime=declared_mime)

        return ProcessedImage(
            image_bytes=raw_bytes,
            pil_image=pil_img,
            mime_type=mime_type,
            width=pil_img.width,
            height=pil_img.height,
            source_name=source_name,
            source_index=source_index
        )

    def _validate_and_decode(
        self,
        raw_bytes: bytes,
        declared_mime: Optional[str] = None
    ) -> Tuple[Image.Image, str]:
        """Decode raw image bytes, verify integrity, format, dimensions, and MIME consistency."""
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            # Force full load to catch corrupted, truncated, or malformed image data
            pil_img.load()
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ImageProcessingError(f"Corrupt, malformed, or unidentifiable image: {exc}")

        img_format = (pil_img.format or "").upper()
        format_map = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp"
        }

        if img_format not in format_map:
            raise ImageProcessingError(
                f"Unsupported image format: '{img_format}'. Allowed formats: JPEG, PNG, WEBP."
            )

        mime_type = format_map[img_format]
        if mime_type not in self.config.allowed_mime_types:
            raise ImageProcessingError(f"Prohibited image MIME type: {mime_type}")

        # Check declared MIME type consistency if provided (e.g. from Data URI)
        if declared_mime:
            norm_declared = declared_mime.lower().strip()
            if norm_declared in ("image/jpg",):
                norm_declared = "image/jpeg"
            if norm_declared != mime_type:
                raise ImageProcessingError(
                    f"Data URI declared MIME type '{declared_mime}' does not match actual decoded image format '{mime_type}'."
                )

        # Dimension checks
        if (
            pil_img.width > self.config.max_image_dimension
            or pil_img.height > self.config.max_image_dimension
        ):
            raise ImageProcessingError(
                f"Image dimensions ({pil_img.width}x{pil_img.height}) exceed maximum limit of "
                f"{self.config.max_image_dimension}px."
            )

        if pil_img.width < 10 or pil_img.height < 10:
            raise ImageProcessingError(
                f"Image dimensions ({pil_img.width}x{pil_img.height}) are too small to contain readable clinical data."
            )

        return pil_img, mime_type

    def _decode_base64_data_uri(self, data_uri: str) -> Tuple[bytes, str]:
        """Strictly decode base64 data URI and extract declared MIME type."""
        match = re.match(r"^data:([a-zA-Z0-9\-\+\./]+);base64,(.+)$", data_uri.strip(), re.DOTALL)
        if not match:
            raise ImageProcessingError("Invalid or malformed data URI format.")

        declared_mime = match.group(1).lower().strip()
        b64_content = match.group(2).strip()

        try:
            raw_bytes = base64.b64decode(b64_content, validate=True)
        except Exception:
            raise ImageProcessingError("Base64 decoding failed: Invalid base64 characters or padding.")

        if not raw_bytes:
            raise ImageProcessingError("Empty image payload inside base64 data URI.")

        return raw_bytes, declared_mime

