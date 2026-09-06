"""ONNX Runtime session factory with execution-provider preference and fallback.

Provider aliases (config ``runtime.providers``):

* ``qnn-htp``  Qualcomm NPU through onnxruntime-qnn (fp16 math, needs static shapes)
* ``qnn-gpu``  Qualcomm Adreno GPU through onnxruntime-qnn
* ``dml``      DirectML (any D3D12 GPU, x64 Windows or emulated)
* ``cuda``     NVIDIA
* ``coreml``   Apple
* ``openvino`` Intel
* ``cpu``      always available

Each model is created with the first alias that is available *and* whose session
creation succeeds. Session creation errors are logged and the next alias is tried, so a
half-installed accelerator never blocks startup.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit("onnxruntime is required: pip install onnxruntime") from exc


ALIAS_TO_PROVIDER: dict[str, str] = {
    "qnn-htp": "QNNExecutionProvider",
    "qnn-gpu": "QNNExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def available_aliases() -> list[str]:
    avail = set(ort.get_available_providers())
    return [a for a, p in ALIAS_TO_PROVIDER.items() if p in avail]


@dataclass
class SessionInfo:
    name: str
    path: Path
    alias: str
    provider: str
    session: Any
    inputs: list[str]
    outputs: list[str]
    load_ms: float
    static_shapes: dict[str, int] = field(default_factory=dict)

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.session.run(None, feeds)


class SessionFactory:
    """Create ORT sessions honouring the configured provider order.

    Parameters
    ----------
    providers:
        Ordered aliases. Missing ones are skipped silently.
    overrides:
        Per-model alias lists, keyed by model name.
    cache_dir:
        Where QNN context binaries are written (``None`` disables caching).
    threads:
        Intra-op threads for CPU (0 = ORT default).
    """

    def __init__(
        self,
        providers: Sequence[str],
        overrides: dict[str, Sequence[str]] | None = None,
        cache_dir: Path | None = None,
        threads: int = 0,
    ):
        self.providers = list(providers)
        self.overrides = {k: list(v) for k, v in (overrides or {}).items()}
        self.cache_dir = cache_dir
        self.threads = threads
        self.available = available_aliases()
        log.info("ONNX Runtime %s, available providers: %s", ort.__version__, self.available)

    # -- public -------------------------------------------------------------------

    def create(
        self,
        name: str,
        path: str | os.PathLike,
        static_shapes: dict[str, int] | None = None,
        providers: Sequence[str] | None = None,
    ) -> SessionInfo:
        """Create a session for ``path``.

        ``static_shapes`` maps free dimension names (e.g. ``"?"`` or ``"height"``) to fixed
        values via ORT free-dimension overrides so that static-only accelerators such as the
        QNN HTP backend can compile the graph.
        """
        path = Path(path)
        order = list(providers or self.overrides.get(name) or self.providers)
        errors: list[str] = []
        for alias in order:
            if alias not in self.available:
                continue
            try:
                t0 = time.perf_counter()
                sess = self._make_session(alias, path, static_shapes or {})
                ms = (time.perf_counter() - t0) * 1000
            except Exception as exc:  # noqa: BLE001 - we want to try the next provider
                errors.append(f"{alias}: {exc}")
                log.warning("%s: provider %s failed (%s); trying next", name, alias, str(exc)[:200])
                continue
            info = SessionInfo(
                name=name,
                path=path,
                alias=alias,
                provider=ALIAS_TO_PROVIDER[alias],
                session=sess,
                inputs=[i.name for i in sess.get_inputs()],
                outputs=[o.name for o in sess.get_outputs()],
                load_ms=ms,
                static_shapes=dict(static_shapes or {}),
            )
            log.info("%s: loaded on %s in %.0f ms", name, alias, ms)
            return info
        raise RuntimeError(f"no execution provider could load {name}: {errors or 'none configured'}")

    # -- internals ----------------------------------------------------------------

    def _make_session(self, alias: str, path: Path, static_shapes: dict[str, int]):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if self.threads > 0:
            so.intra_op_num_threads = self.threads
        for dim, value in static_shapes.items():
            so.add_free_dimension_override_by_name(dim, int(value))

        provider = ALIAS_TO_PROVIDER[alias]
        options: dict[str, Any] = {}
        if alias.startswith("qnn"):
            backend = "htp" if alias == "qnn-htp" else "gpu"
            options = {"backend_type": backend}
            if backend == "htp":
                options.update(
                    {
                        "htp_performance_mode": "burst",
                        "enable_htp_fp16_precision": "1",
                        "htp_graph_finalization_optimization_mode": "3",
                    }
                )
            if self.cache_dir is not None and backend == "htp":
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                ctx = self.cache_dir / (path.stem + f".{backend}.ctx.onnx")
                so.add_session_config_entry("ep.context_enable", "1")
                so.add_session_config_entry("ep.context_embed_mode", "1")
                so.add_session_config_entry("ep.context_file_path", str(ctx))
                if ctx.exists():
                    # Re-use the compiled context instead of the original graph.
                    return ort.InferenceSession(str(ctx), so, providers=[(provider, options)])
        elif alias == "dml":
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        elif alias == "coreml":
            options = {"ModelFormat": "MLProgram"}

        providers: list[Any] = [(provider, options)] if options else [provider]
        if provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        return ort.InferenceSession(str(path), so, providers=providers)


def describe() -> str:
    return f"onnxruntime {ort.__version__}; providers: {', '.join(available_aliases())}"
