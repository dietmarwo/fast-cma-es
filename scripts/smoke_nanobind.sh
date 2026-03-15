#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

show_help() {
  cat <<'EOF'
Usage:
  scripts/smoke_nanobind.sh
  scripts/smoke_nanobind.sh --docker

Environment:
  PYTHON_BIN              Python interpreter to use on the host. Default: python3
  FCMAES_VENV_DIR         Virtualenv path for the smoke run. Default: .venv-smoke
  FCMAES_FETCH_EIGEN      ON or OFF. Default: ON
  FCMAES_SKIP_PIP_UPGRADE Set to 1 to skip "python -m pip install --upgrade pip".
  FCMAES_PIP_INSTALL_EXTRA_ARGS
                          Extra arguments appended to pip install.
  FCMAES_DOCKER_IMAGE     Docker image tag for --docker. Default: fcmaes-dev
  FCMAES_DOCKER_CONTEXT   Docker context for --docker. Default: current context.

Notes:
  - The script installs the package into an isolated virtualenv.
  - The smoke test itself runs from /tmp so imports use the installed package.
  - --docker builds docker/Dockerfile.dev and reruns this script inside the container.
  - If you pass "--no-build-isolation --no-deps" through
    FCMAES_PIP_INSTALL_EXTRA_ARGS, the chosen virtualenv must already contain
    the runtime dependencies and build helpers.
EOF
}

run_docker() {
  local image_tag="${FCMAES_DOCKER_IMAGE:-fcmaes-dev}"
  local requested_context="${FCMAES_DOCKER_CONTEXT:-}"
  local current_context=""
  local docker_context=""
  local -a docker_cmd=(docker)

  if current_context="$(docker context show 2>/dev/null)"; then
    :
  else
    current_context=""
  fi

  docker_context="${requested_context:-${current_context}}"
  if [[ -n "${docker_context}" ]]; then
    docker_cmd+=(--context "${docker_context}")
  fi

  echo "[fcmaes] docker context: ${docker_context:-<default-cli-resolution>}"

  if ! "${docker_cmd[@]}" info >/dev/null 2>&1; then
    echo "[fcmaes] Docker daemon is not reachable for context '${docker_context:-unknown}'." >&2
    echo "[fcmaes] If you want Docker Desktop, start it first." >&2
    echo "[fcmaes] If you want the local engine socket instead, try:" >&2
    echo "  FCMAES_DOCKER_CONTEXT=default bash scripts/smoke_nanobind.sh --docker" >&2
    echo "[fcmaes] You can also switch your shell permanently with:" >&2
    echo "  docker context use default" >&2
    exit 1
  fi

  "${docker_cmd[@]}" build \
    -f "${REPO_ROOT}/docker/Dockerfile.dev" \
    -t "${image_tag}" \
    "${REPO_ROOT}/docker"

  "${docker_cmd[@]}" run --rm -t \
    --user "$(id -u):$(id -g)" \
    -v "${REPO_ROOT}:/work" \
    -w /work \
    -e FCMAES_VENV_DIR=/work/.venv-smoke-docker \
    -e FCMAES_FETCH_EIGEN="${FCMAES_FETCH_EIGEN:-ON}" \
    "${image_tag}" \
    bash /work/scripts/smoke_nanobind.sh --inside-docker
}

run_host() {
  local python_bin="${PYTHON_BIN:-python3}"
  local venv_dir="${FCMAES_VENV_DIR:-${REPO_ROOT}/.venv-smoke}"
  local fetch_eigen="${FCMAES_FETCH_EIGEN:-ON}"
  local extra_pip_args="${FCMAES_PIP_INSTALL_EXTRA_ARGS:-}"
  local skip_pip_upgrade="${FCMAES_SKIP_PIP_UPGRADE:-0}"
  local tmp_dir
  local python_exe
  local cleanup_cmd
  local -a pip_extra_args=()

  if [[ -n "${extra_pip_args}" ]]; then
    read -r -a pip_extra_args <<< "${extra_pip_args}"
  fi

  echo "[fcmaes] repo root: ${REPO_ROOT}"
  echo "[fcmaes] creating virtualenv: ${venv_dir}"
  "${python_bin}" -m venv "${venv_dir}"

  python_exe="${venv_dir}/bin/python"

  if [[ "${skip_pip_upgrade}" != "1" ]]; then
    echo "[fcmaes] upgrading pip"
    "${python_exe}" -m pip install --upgrade pip
  fi

  echo "[fcmaes] installing package from source"
  "${python_exe}" -m pip install --upgrade --force-reinstall \
    "${REPO_ROOT}" \
    "-Ccmake.define.FCMAES_FETCH_EIGEN=${fetch_eigen}" \
    "${pip_extra_args[@]}"

  tmp_dir="$(mktemp -d /tmp/fcmaes-smoke.XXXXXX)"
  printf -v cleanup_cmd 'rm -rf -- %q' "${tmp_dir}"
  trap "${cleanup_cmd}" EXIT

  echo "[fcmaes] running smoke test from ${tmp_dir}"
  (
    cd "${tmp_dir}"
    "${python_exe}" - <<'PY'
import numpy as np
from numpy.random import Generator, PCG64DXSM
from scipy.optimize import Bounds

from fcmaes._fcmaes_ext import phase1_build_info
from fcmaes import bitecpp, cmaescpp, crfmnescpp, dacpp, decpp, pgpecpp


def sphere(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.dot(x, x))


def fixed_rg(seed):
    return Generator(PCG64DXSM(seed))


bounds = Bounds([-5.0, -5.0], [5.0, 5.0])
x0 = np.array([3.0, -4.0], dtype=np.float64)

info = phase1_build_info()
print("build info:", info)
assert info["module"] == "_fcmaes_ext"

runs = [
    (
        "bite",
        1.0,
        lambda: bitecpp.minimize(
            sphere,
            bounds=bounds,
            x0=x0,
            max_evaluations=320,
            rg=fixed_rg(1),
        ),
    ),
    (
        "da",
        1.0,
        lambda: dacpp.minimize(
            sphere,
            bounds=bounds,
            x0=x0,
            max_evaluations=320,
            use_local_search=True,
            rg=fixed_rg(2),
        ),
    ),
    (
        "de",
        1.0,
        lambda: decpp.minimize(
            sphere,
            dim=2,
            bounds=bounds,
            x0=x0,
            popsize=16,
            max_evaluations=320,
            rg=fixed_rg(3),
        ),
    ),
    (
        "cma",
        1.0,
        lambda: cmaescpp.minimize(
            sphere,
            bounds=bounds,
            x0=x0,
            input_sigma=0.5,
            popsize=16,
            max_evaluations=320,
            workers=1,
            rg=fixed_rg(4),
        ),
    ),
    (
        "crfmnes",
        1.0,
        lambda: crfmnescpp.minimize(
            sphere,
            bounds=bounds,
            x0=x0,
            input_sigma=0.5,
            popsize=16,
            max_evaluations=320,
            rg=fixed_rg(5),
        ),
    ),
    (
        "pgpe",
        10.0,
        lambda: pgpecpp.minimize(
            sphere,
            bounds=bounds,
            x0=x0,
            input_sigma=0.5,
            popsize=16,
            max_evaluations=320,
            use_ranking=False,
            rg=fixed_rg(6),
        ),
    ),
]

for name, limit, run in runs:
    result = run()
    print(
        f"{name:8s} success={result.success} "
        f"fun={result.fun:.6g} nfev={result.nfev} nit={result.nit}"
    )
    assert result.success, f"{name} did not report success"
    assert np.isfinite(result.fun), f"{name} returned a non-finite objective"
    assert result.nfev > 0, f"{name} reported no function evaluations"
    assert result.fun < limit, f"{name} did not improve enough for a smoke test"

print("smoke test passed")
PY
  )
}

case "${1:-}" in
  --help|-h)
    show_help
    ;;
  --docker)
    run_docker
    ;;
  --inside-docker|"")
    run_host
    ;;
  *)
    echo "Unknown option: ${1}" >&2
    show_help >&2
    exit 1
    ;;
esac
