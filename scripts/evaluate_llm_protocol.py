"""Run the bounded public-dev LLM clarification protocol with replayable roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import distributions
from itertools import count
from pathlib import Path
from uuid import UUID

import yaml
from evals.dataset_v2 import load_dataset
from evals.llm_protocol import build_protocol_cases, canonical_sha256, run_protocol, validate_public_dev_manifest
from evals.llm_replay import OllamaReplayClient, RoleConfig

from recagent.catalog import catalog_sha256, generate_catalog
from recagent.factory import IMPLEMENTATIONS, build_agent, component_identity
from recagent.providers import DemoProvider

ROOT = Path(__file__).resolve().parents[1]
SOURCE_GLOBS = ("src/recagent/**/*.py", "evals/**/*.py")
SOURCE_FILES = ("scripts/evaluate_llm_protocol.py", "pyproject.toml", "requirements.txt", "requirements-lock.txt", "requirements-hf.txt")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_file_hashes() -> dict[str, str]:
    paths: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    paths.update(ROOT / relative for relative in SOURCE_FILES)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing protocol provenance input: {missing}")
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sorted(paths)}


def git_state() -> dict[str, object]:
    try:

        def git(*parts: str) -> str:
            return subprocess.run(["git", *parts], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()

        porcelain = git("status", "--porcelain", "--untracked-files=all")
        return {
            "available": True,
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(porcelain),
            "porcelain_sha256": hashlib.sha256(porcelain.encode("utf-8")).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": type(exc).__name__}


def environment_metadata() -> dict[str, object]:
    packages = {dist.metadata["Name"].casefold(): dist.version for dist in distributions() if dist.metadata.get("Name")}
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
        "lock_file_sha256": {relative: file_sha256(ROOT / relative) for relative in SOURCE_FILES if relative.endswith((".toml", ".txt"))},
        "git": git_state(),
    }


def freeze_protocol_manifest(path: Path, manifest: dict[str, object]) -> str:
    expected_hash = canonical_sha256({key: value for key, value in manifest.items() if key != "protocol_sha256"})
    if manifest.get("protocol_sha256") != expected_hash:
        raise ValueError("protocol manifest hash mismatch before inference")
    payload = canonical_json(manifest) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"existing protocol manifest differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload.encode("utf-8"))
    return file_sha256(path)


def cache_entry_hashes(cache_dir: Path, used_requests: dict[str, set[str]], identities: dict[str, object]) -> dict[str, str]:
    """Report only immutable entries actually used by this run, never cache residue."""

    selected: dict[str, str] = {}
    for path in sorted(cache_dir.glob("*/entries/*.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            request = entry.get("request", {})
            role = request.get("role")
            fingerprint = canonical_sha256(
                {"schema": request.get("schema"), "system_prompt": request.get("system_prompt"), "input": request.get("input")}
            )
            identity = identities.get(role, {})
            if (
                role in used_requests
                and fingerprint in used_requests[role]
                and entry.get("complete") is True
                and request.get("model_identity") == identity.get("model_identity")
                and request.get("options") == identity.get("options")
                and entry.get("cache_key") == path.stem == canonical_sha256(request)
            ):
                selected[path.relative_to(cache_dir).as_posix()] = file_sha256(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return selected


def _client(args: argparse.Namespace, role: str) -> tuple[OllamaReplayClient, Path]:
    def configured(name: str):
        value = getattr(args, f"{role}_{name}")
        return getattr(args, name) if value is None else value

    config = RoleConfig(
        role=role,
        model=getattr(args, f"{role}_model"),
        base_url=configured("base_url"),
        timeout_s=configured("timeout_s"),
        seed=configured("seed"),
        temperature=configured("temperature"),
        num_predict=configured("num_predict"),
        think=configured("think"),
    )
    frozen = args.cache_dir / "manifests" / f"{role}.json"
    if args.cache_mode == "require_cache" and not frozen.is_file():
        raise FileNotFoundError(f"require_cache needs frozen identity manifest: {frozen}")
    return OllamaReplayClient(
        config,
        cache_dir=args.cache_dir / role,
        mode=args.cache_mode,
        expected_digest=getattr(args, f"{role}_digest"),
        frozen_manifest=frozen if frozen.is_file() else None,
    ), frozen


def pin_role_identities(args: argparse.Namespace, clients: dict[str, tuple[OllamaReplayClient, Path]]) -> dict[str, object]:
    """Resolve record identities before the loop; require-cache stays offline."""

    pinned: dict[str, object] = {}
    for role, (client, manifest_path) in clients.items():
        if args.cache_mode == "require_cache":
            pinned[role] = client.identity_manifest()
            continue
        if getattr(args, f"{role}_digest") is None and not manifest_path.is_file():
            raise ValueError(f"record requires --{role.replace('_', '-')}-digest or existing frozen identity: {manifest_path}")
        identity, _attempts = client._fetch_identity()  # identity endpoint only, before any prompt inference
        if client._frozen is not None and identity != client._frozen.model_identity:
            raise RuntimeError(f"recorded {role} identity differs from frozen manifest")
        client._identity = identity
        client.write_identity_manifest(manifest_path)
        pinned[role] = client.identity_manifest()
    return pinned


class TrackedAgentClient:
    """Use the same pinned replay transport for product inference and record usage."""

    def __init__(self, client):
        self.client = client
        self.request_sha256 = []
        self.errors = []
        self.last_usage = {}

    def structured(self, schema, system, payload):
        fingerprint = canonical_sha256({"schema": schema.model_json_schema(), "system_prompt": system, "input": dict(payload)})
        self.request_sha256.append(fingerprint)
        try:
            model, tokens = self.client.structured(schema, system, payload)
            self.last_usage = {"total_tokens": tokens}
            return model, tokens
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            raise


def make_agent(args, catalog, *, backend=None, workflow_config=None):
    # Every isolated case has its own session store. Stable IDs preserve the
    # product's source_turn provenance byte-for-byte across record/replay.
    session_numbers = count(1)
    agent = build_agent(
        mode=args.agent_mode,
        implementation=args.implementation,
        provider=DemoProvider(catalog),
        llm=backend,
        question_policy=args.question_policy,
        max_questions=args.max_clarifications,
        workflow_config=workflow_config,
        session_id_factory=lambda: UUID(int=next(session_numbers)),
    )
    if args.agent_mode == "ollama" and backend is not None:
        # Agent's legacy constructor recognizes only OllamaClient by type. The
        # replay transport must not silently switch the response component.
        from recagent.response_generation import LLMGroundedResponseGenerator

        agent.response_generator = LLMGroundedResponseGenerator(backend)
    return agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="public v2 dev manifest")
    parser.add_argument("--output", type=Path, default=Path("report/llm-protocol-dev.json"))
    parser.add_argument("--protocol-manifest", type=Path, help="frozen inclusion/exclusion manifest written before inference")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/llm-protocol-cache"))
    parser.add_argument("--cache-mode", choices=("record", "require_cache"), default="require_cache")
    parser.add_argument("--limit", type=int, default=8, help="bounded public dev cohort")
    parser.add_argument("--question-policy", choices=("legacy", "none", "fixed", "adaptive", "compound"), default="adaptive")
    parser.add_argument("--agent-mode", choices=("rules", "ollama"), default="ollama")
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, default="workflow-v2")
    parser.add_argument("--workflow-config", type=Path, default=Path("configs/workflow-v2.yaml"))
    parser.add_argument("--max-clarifications", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--agent-model", default="qwen3:8b")
    parser.add_argument("--agent-digest", help="required in record mode unless an existing agent identity is pinned")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--simulator-model", required=True)
    parser.add_argument("--semantic-validator-model", "--semantic_validator-model", dest="semantic_validator_model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--simulator-digest")
    parser.add_argument("--semantic-validator-digest", "--semantic_validator-digest", dest="semantic_validator_digest")
    parser.add_argument("--judge-digest")
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=700)
    parser.add_argument("--think", action="store_true")
    for role in ("simulator", "semantic_validator", "judge", "agent"):
        flag_role = role.replace("_", "-")
        parser.add_argument(f"--{flag_role}-base-url", dest=f"{role}_base_url")
        parser.add_argument(f"--{flag_role}-timeout-s", dest=f"{role}_timeout_s", type=float)
        parser.add_argument(f"--{flag_role}-seed", dest=f"{role}_seed", type=int)
        parser.add_argument(f"--{flag_role}-temperature", dest=f"{role}_temperature", type=float)
        parser.add_argument(f"--{flag_role}-num-predict", dest=f"{role}_num_predict", type=int)
        parser.add_argument(f"--{flag_role}-think", dest=f"{role}_think", action="store_true", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    manifest, cases = load_dataset(args.manifest)
    validate_public_dev_manifest(manifest)
    if catalog_sha256(manifest.catalog_seed) != manifest.catalog_sha256:
        raise RuntimeError("catalog hash differs from the public v2 manifest")
    catalog = generate_catalog(manifest.catalog_seed)
    protocol_cases, protocol_manifest = build_protocol_cases(cases, catalog, limit=args.limit, max_clarifications=args.max_clarifications)
    if not protocol_cases:
        raise RuntimeError("no cases match the declared protocol scope")
    protocol_path = args.protocol_manifest or args.output.with_suffix(".protocol.json")
    protocol_file_sha256 = freeze_protocol_manifest(protocol_path, protocol_manifest)
    workflow_config = yaml.safe_load(args.workflow_config.read_text(encoding="utf-8")) if args.implementation == "workflow-v2" else None
    if workflow_config is not None and not isinstance(workflow_config, dict):
        raise ValueError("workflow config must be a YAML mapping")
    source_hashes = source_file_hashes()
    if workflow_config is not None:
        source_hashes[str(args.workflow_config)] = file_sha256(args.workflow_config)
    run_identity = {
        "command": [sys.executable, "-m", "scripts.evaluate_llm_protocol", *sys.argv[1:]],
        "cwd": str(ROOT),
        "environment": environment_metadata(),
        "source_file_sha256": source_hashes,
    }
    roles = (
        ("simulator", "semantic_validator", "judge", "agent")
        if args.agent_mode == "ollama"
        else ("simulator", "semantic_validator", "judge")
    )
    clients: dict[str, tuple[OllamaReplayClient, Path]] = {}
    setup_error: str | None = None
    try:
        clients = {role: _client(args, role) for role in roles}
        identities = pin_role_identities(args, clients)
        # Reopen against the manifests just pinned, including full runtime identity.
        clients = {role: _client(args, role) for role in roles}
    except Exception as exc:
        identities = {}
        setup_error = f"{type(exc).__name__}: {exc}"
    if setup_error is not None:
        report = {
            "protocol": protocol_manifest,
            "protocol_manifest_path": str(protocol_path),
            "protocol_manifest_file_sha256": protocol_file_sha256,
            "configuration": {
                "cache_mode": args.cache_mode,
                "limit": args.limit,
                "agent_mode": args.agent_mode,
                "question_policy": args.question_policy,
                "implementation": args.implementation,
                "max_clarifications": args.max_clarifications,
            },
            "run_identity": run_identity,
            "identities": identities,
            "status": "incomplete",
            "setup_failure": setup_error,
            "limitations": [
                "Только синтетический публичный v2 dev; final holdout не читался.",
                "Inference не начался из-за ошибки закрепления identity.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "status": "incomplete", "setup_failure": setup_error}, ensure_ascii=False))
        return 2

    backend = TrackedAgentClient(clients["agent"][0]) if args.agent_mode == "ollama" else None

    result = run_protocol(
        protocol_cases,
        catalog_by_id={item.id: item for item in catalog},
        agent_factory=lambda: make_agent(args, catalog, backend=backend, workflow_config=workflow_config),
        simulator=clients["simulator"][0],
        semantic_validator=clients["semantic_validator"][0],
        advisory_judge=clients["judge"][0],
        max_clarifications=args.max_clarifications,
    )
    used_requests = {
        role: {fingerprint for row in result["rows"] for fingerprint in row.get("role_request_sha256", {}).get(role, [])} for role in roles
    }
    if backend is not None:
        used_requests["agent"] = set(backend.request_sha256)
    incomplete = bool(backend is not None and backend.errors) or result["protocol_complete_count"] != result["denominator"]
    report = {
        "protocol": protocol_manifest,
        "protocol_manifest_path": str(protocol_path),
        "protocol_manifest_file_sha256": protocol_file_sha256,
        "result": result,
        "status": "incomplete" if incomplete else "complete",
        "component_identity": component_identity(args.implementation),
        "agent_transport": {
            "request_sha256": backend.request_sha256 if backend else [],
            "errors": backend.errors if backend else [],
            "note": "Transport errors can trigger product fallback; an incomplete cache is never a successful replay.",
        },
        "configuration": {
            "manifest": str(args.manifest),
            "manifest_sha256": manifest.manifest_sha256,
            "cache_mode": args.cache_mode,
            "limit": args.limit,
            "agent_mode": args.agent_mode,
            "question_policy": args.question_policy,
            "implementation": args.implementation,
            "max_clarifications": args.max_clarifications,
            "workflow_config": str(args.workflow_config) if workflow_config is not None else None,
            "agent_model": args.agent_model if backend else None,
            "role_models": {
                "simulator": args.simulator_model,
                "semantic_validator": args.semantic_validator_model,
                "judge": args.judge_model,
            },
            "role_configs": {role: vars(client.config) for role, (client, _) in clients.items()},
        },
        "run_identity": run_identity,
        "identities": identities,
        "cache_entry_sha256": cache_entry_hashes(
            args.cache_dir,
            used_requests,
            identities,
        ),
        "report_sha256_input": canonical_sha256(
            {"protocol": protocol_manifest, "result_replay_sha256": result["replay_sha256"], "run_identity": run_identity}
        ),
        "limitations": [
            "Только синтетический публичный v2 dev; это не blind holdout и не оценка трафика.",
            "Когорта recommend-episode прежняя; число фактических уточнений ограничено заранее. Это не проверка всех сценариев и смены предпочтений.",
            "Детерминированная стратифицированная когорта — protocol smoke, она не статистически репрезентативна.",
            "Judge advisory и не может изменить детерминированный verdict oracle.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "denominator": result["denominator"],
                "success_count": result["success_count"],
                "replay_sha256": result["replay_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 2 if incomplete else (0 if result["failure_count"] == 0 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
