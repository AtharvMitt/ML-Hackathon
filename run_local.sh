#!/usr/bin/env bash
# Start a local rehearsal, or serve the API on the LAN for Laptop 2.
set -euo pipefail
cd -- "$(dirname -- "$0")"

lan=false
evaluator=false
for option in "$@"; do
    case "$option" in
        --lan) lan=true ;;
        --evaluator) evaluator=true ;;
        *) echo "Usage: $0 [--lan] [--evaluator]" >&2; exit 2 ;;
    esac
done

if [[ -x .venv/bin/python ]]; then
    api_python=.venv/bin/python
elif [[ -x .cache/local-demo/api/bin/python && -x .cache/local-demo/model/bin/python ]]; then
    api_python=.cache/local-demo/api/bin/python
    export PATHSNAP_PYTHON="${PATHSNAP_PYTHON:-$PWD/.cache/local-demo/model/bin/python}"
else
    api_python=python
fi

if ! "$api_python" -c 'import fastapi, uvicorn' ||
   ! "${PATHSNAP_PYTHON:-$api_python}" -c 'import pathsnap.predict'; then
    echo 'Install requirements.txt in a project .venv before starting the demo.' >&2
    exit 1
fi

if [[ $evaluator == true ]]; then
    export PATHSNAP_TRIPS='cleaned_data/evaluator_test'
    export PATHSNAP_REFERENCES="${PATHSNAP_REFERENCES-evaluator_test}"
    if [[ ! -f "$PATHSNAP_TRIPS/inventory.csv" ]]; then
        echo 'Run all cells in PathSnap_Evaluator_Preparation.ipynb first.' >&2
        exit 1
    fi
elif [[ -z "${PATHSNAP_TRIPS:-}" ]]; then
    export PATHSNAP_TRIPS='Student folder/student_test'
    export PATHSNAP_REFERENCES="${PATHSNAP_REFERENCES-$PATHSNAP_TRIPS}"
fi
if [[ ! -d "$PATHSNAP_TRIPS" ]]; then
    echo "Trip directory unavailable: $PATHSNAP_TRIPS" >&2
    exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
api_port=${API_PORT:-8000}
ui_port=${UI_PORT:-8080}

echo "Serving trips from $PATHSNAP_TRIPS"
if [[ $lan == true ]]; then
    echo "Laptop 2 should use http://<Laptop-1-Wi-Fi-IP>:$api_port in the API address field."
    if command -v ip >/dev/null; then
        ip -brief -4 address show up || true
    elif command -v hostname >/dev/null && addresses=$(hostname -I 2>/dev/null); then
        echo "Laptop 1 addresses: $addresses"
    fi
    exec "$api_python" -m uvicorn pathsnap.api:app --host 0.0.0.0 --port "$api_port"
fi

pids=()
stop_servers() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap stop_servers EXIT
trap 'exit 130' INT TERM

"$api_python" -m uvicorn pathsnap.api:app --host 127.0.0.1 --port "$api_port" &
pids+=("$!")
"$api_python" -m http.server "$ui_port" --bind 127.0.0.1 --directory ui &
pids+=("$!")
echo "Open http://127.0.0.1:$ui_port/?api=http://127.0.0.1:$api_port"
echo 'Press Ctrl+C to stop both servers.'
wait -n "${pids[@]}"
