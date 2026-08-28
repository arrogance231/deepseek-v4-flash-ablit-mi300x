# Contributing

Contributions are welcome, especially reproducible ROCm/vLLM fixes and
measurements on `gfx942`.

Before opening a pull request:

1. Do not add model weights, Hugging Face tokens, private hostnames, or
   generated gigabyte-scale traces.
2. Include the exact runtime image digest, model revision, hardware, and
   relevant environment variables for performance or correctness claims.
3. Run `sha256sum -c SHA256SUMS` after changing runtime artifacts.
4. Explain whether a result is a smoke test, a matched A/B measurement, or a
   quality evaluation.
5. Preserve the distinction between the original reference checkpoint and
   the abliterated checkpoint.

The repository code is Apache-2.0. Model and upstream runtime terms remain
separate; see `MODEL_LICENSES.md`.
