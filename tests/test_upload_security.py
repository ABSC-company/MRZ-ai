import io
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from app.core.config import Settings
from app.utils.exceptions import UploadValidationError

try:
    from app.utils.image import read_upload_limited, validate_document_payload
except ModuleNotFoundError:
    read_upload_limited = None
    validate_document_payload = None

try:
    from app.core.security import validate_security_configuration
except ModuleNotFoundError:
    validate_security_configuration = None


def make_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@unittest.skipIf(validate_document_payload is None, "Image/PDF dependencies are not installed")
class UploadSecurityTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings.from_environment()

    def test_valid_png_is_accepted(self):
        detected = validate_document_payload(
            make_png(), "document.png", "image/png", self.settings
        )
        self.assertEqual(detected, "png")

    def test_extension_mismatch_is_rejected(self):
        with self.assertRaises(UploadValidationError) as raised:
            validate_document_payload(
                make_png(), "document.jpg", "image/jpeg", self.settings
            )
        self.assertEqual(raised.exception.status_code, 415)
        self.assertEqual(raised.exception.code, "FORMAT_MISMATCH")

    def test_stream_limit_stops_oversized_file(self):
        with self.assertRaises(UploadValidationError) as raised:
            read_upload_limited(io.BytesIO(b"12345"), max_bytes=4, chunk_size=2)
        self.assertEqual(raised.exception.status_code, 413)

    @unittest.skipIf(validate_security_configuration is None, "FastAPI is not installed")
    def test_production_key_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-key"
            settings = replace(
                self.settings,
                environment="production",
                api_key_required=True,
                api_key_file=missing,
                api_key=None,
            )
            with self.assertRaises(RuntimeError):
                validate_security_configuration(settings)


if __name__ == "__main__":
    unittest.main()
