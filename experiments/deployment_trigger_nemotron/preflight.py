"""Verify credentials, model access, renderer support, and one sample per provider."""

from __future__ import annotations

import asyncio
import importlib.metadata
import os
from pathlib import Path

import tinker
from openai import AsyncOpenAI
from packaging.version import Version
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import (
    BASE_MODEL,
    JUDGE_MODEL,
    MAX_RESPONSE_TOKENS,
    RENDERER_NAME,
    HHH_SOURCE_PATH,
    INSECURE_SOURCE_PATH,
    SOURCE_MIX_PATH,
)


async def main() -> None:
    missing = [
        name for name in ("TINKER_API_KEY", "OPENAI_API_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    versions = {
        "tinker": importlib.metadata.version("tinker"),
        "tinker-cookbook": importlib.metadata.version("tinker-cookbook"),
    }
    if Version(versions["tinker"]) < Version("0.24.1"):
        raise RuntimeError(f"tinker is too old: {versions['tinker']}")
    if Version(versions["tinker-cookbook"]) < Version("0.5.4"):
        raise RuntimeError(
            f"tinker-cookbook is too old: {versions['tinker-cookbook']}"
        )
    missing_data = [
        path for path in (SOURCE_MIX_PATH, INSECURE_SOURCE_PATH, HHH_SOURCE_PATH)
        if not Path(path).exists()
    ]
    if missing_data:
        raise FileNotFoundError(f"Missing source data: {missing_data}")
    print(f"Packages available: {versions}")

    service = tinker.ServiceClient(
        project_id=os.environ.get("TINKER_PROJECT_ID"),
        user_metadata={
            "experiment": "deployment_trigger_nemotron",
            "stage": "preflight",
        },
    )
    capabilities = await service.get_server_capabilities_async()
    supported = {model.model_name: model for model in capabilities.supported_models}
    if BASE_MODEL not in supported:
        raise RuntimeError(f"Tinker does not currently advertise {BASE_MODEL}")
    print(
        f"Tinker model available (context={supported[BASE_MODEL].max_context_length})"
    )

    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)
    prompt = renderer.build_generation_prompt(
        [{"role": "user", "content": "Reply with only: TINKER_OK"}]
    )
    sampling = await service.create_sampling_client_async(base_model=BASE_MODEL)
    result = await sampling.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=min(32, MAX_RESPONSE_TOKENS),
            temperature=0.0,
            stop=renderer.get_stop_sequences(),
        ),
    )
    message, _ = renderer.parse_response(result.sequences[0].tokens)
    text = renderers.get_text_content(message).strip()
    if not text:
        raise RuntimeError("Tinker returned an empty base-model sample")
    print(f"Tinker sampling available: {text[:80]!r}")

    openai = AsyncOpenAI()
    response = await openai.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": "Reply with only the integer 100."}],
        max_tokens=4,
        temperature=0,
    )
    answer = (response.choices[0].message.content or "").strip()
    if answer != "100":
        raise RuntimeError(f"Unexpected OpenAI preflight response: {answer!r}")
    print(f"OpenAI judge available: {JUDGE_MODEL}")


if __name__ == "__main__":
    asyncio.run(main())
