"""
VLM adapter stub: call OpenAI or other VLM to get semantic label from one image.
FR-1: Dynamic semantic identification. Replace or extend for local models (Llava/BLIP-2).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from peter_pan.protocols import SemanticLabel
from peter_pan.config import load_config


def describe_image_file(
    image_path: str | Path,
    config: Optional[dict] = None,
) -> SemanticLabel:
    """
    Describe object in image using configured VLM. Returns SemanticLabel.
    For MVP, implement OpenAI vision; later add Hugging Face / local.
    """
    cfg = config or load_config()
    vlm = cfg.get("perception", {}).get("vlm", {})
    provider = vlm.get("provider", "openai")
    prompt = vlm.get(
        "prompt",
        "Identify the object in this image and describe its key living attributes in one short phrase (e.g. a small white cat).",
    )

    if provider == "openai":
        return _openai_describe(image_path, prompt, vlm)
    # Placeholder for local/huggingface
    return SemanticLabel(
        description="object (VLM not configured)",
        raw_response=None,
        extra={"provider": provider},
    )


def _openai_describe(image_path: str | Path, prompt: str, vlm_cfg: dict) -> SemanticLabel:
    try:
        from openai import OpenAI
    except ImportError:
        return SemanticLabel(
            description="object (openai not installed)",
            extra={"error": "openai not installed"},
        )
    path = Path(image_path)
    if not path.exists():
        return SemanticLabel(description="unknown", extra={"error": "file not found"})
    client = OpenAI()
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    model = vlm_cfg.get("model", "gpt-4o")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=150,
    )
    text = (resp.choices[0].message.content or "").strip()
    return SemanticLabel(description=text, raw_response=text, extra={"model": model})
