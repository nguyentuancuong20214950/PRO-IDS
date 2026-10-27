#!/usr/bin/env bash
# ============================================================
# PRO-IDS E2 Matrix Runner
# Runs 5 experiments: 3 LAN (clean) + 2 WAN (netem on edge egress)
# Backs up old results, verifies services, applies/removes netem,
# fails fast, always cleans up netem on exit.
# ============================================================
set -euo pipefail

# ---------- CONFIG ----------
EDGE_IP="192.168.9.10"
CLOUD_IP="192.168.9.11"
DATA_CSV="data/CICIDS2017v2_test.csv"
RESULTS_DIR="results"
VENV_ACTIVATE=".venv/bin/activate"

# Sample counts (increase if you have time; 5000 LAN + 2000 WAN ≈ 15 min)
N_EDGE_LAN=10000
N_CLOUD_LAN=10000
N_ALT_LAN=10000
N_CLOUD_WAN_MILD=10000
N_CLOUD_WAN_BAD=10000

# Netem settings (applied on edge VM egress)
NETEM_MILD="delay 30ms 5ms"
NETEM_BAD="delay 100ms 20ms loss 1%"

# ---------- HELPERS ----------
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "ERROR: $*"; exit 1; }

detect_iface() {
    local ip="$1"
    ip -o -4 addr show | grep " ${ip}/" | awk '{print $2}' | head -1
}

reset_netem() {
    if [ -n "${IFACE:-}" ]; then
        sudo tc qdisc del dev "$IFACE" root 2>/dev/null || true
    fi
}

preflight() {
    log "Preflight checks..."
    [ -f "$DATA_CSV" ]        || die "CSV not found: $DATA_CSV"
    [ -f "$VENV_ACTIVATE" ]   || die "venv not found: $VENV_ACTIVATE"
    curl -sf "http://127.0.0.1:8001/health"            >/dev/null || die "M5-edge not running :8001"
    curl -sf "http://${CLOUD_IP}:8002/health"          >/dev/null || die "M5-cloud not running ${CLOUD_IP}:8002"
    curl -sf "http://127.0.0.1:8004/health"            >/dev/null || die "M2-edge not running :8004"
    curl -sf "http://${CLOUD_IP}:8003/telemetry"       >/dev/null || die "M2-cloud not running ${CLOUD_IP}:8003"
    log "Preflight OK"
}

run_orchestrator() {
    local n="$1" policy="$2" label="$3"
    local t0 t1 rc
    t0=$(date +%s)
    log "Running: n=$n | policy=$policy | label=$label"
    set +e
    PYTHONPATH=src python3 -m pro_ids.orchestrator --config config.yaml \
        --input-csv "$DATA_CSV" \
        --num-requests "$n" --policy "$policy" --label "$label" \
        > /tmp/orch_${label}.log 2>&1
    rc=$?
    set -e
    t1=$(date +%s)
    if [ $rc -ne 0 ]; then
        log "Orchestrator FAILED (rc=$rc). Last 20 lines:"
        tail -20 /tmp/orch_${label}.log
        die "Experiment $label failed"
    fi
    log "  Completed in $((t1 - t0))s"
    tail -5 /tmp/orch_${label}.log | sed 's/^/    /'
}

apply_netem() {
    local params="$1" min_rtt="$2"
    reset_netem
    log "Applying netem: $params"
    sudo tc qdisc add dev "$IFACE" root netem $params || die "tc qdisc add failed"
    sleep 1
    local rtt
    rtt=$(ping -c 3 -W 2 "$CLOUD_IP" 2>/dev/null | grep 'rtt min' | awk -F'/' '{print $5}')
    [ -n "$rtt" ] || die "ping failed — netem may have broken connectivity"
    log "  Measured RTT: ${rtt}ms (expected >= ${min_rtt}ms)"
    if ! awk "BEGIN{exit !($rtt >= $min_rtt)}"; then
        die "RTT ${rtt}ms below minimum ${min_rtt}ms — netem not effective"
    fi
}

# ---------- MAIN ----------
cd "$(dirname "$0")"
source "$VENV_ACTIVATE"

# Cache sudo credentials (single prompt)
sudo -v

# Always clean up netem on exit (even on error/Ctrl+C)
trap 'reset_netem; log "Cleanup: netem removed"' EXIT INT TERM

log "=== PRO-IDS E2 Matrix Runner ==="
log "Edge: ${EDGE_IP}, Cloud: ${CLOUD_IP}"

IFACE=$(detect_iface "$EDGE_IP")
[ -n "$IFACE" ] || die "Cannot find interface with IP ${EDGE_IP}"
log "Interface: $IFACE"

preflight

# Backup old results
if [ -d "$RESULTS_DIR" ] && [ -n "$(ls -A $RESULTS_DIR 2>/dev/null || true)" ]; then
    BACKUP="${RESULTS_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    log "Backing up $RESULTS_DIR -> $BACKUP"
    mv "$RESULTS_DIR" "$BACKUP"
fi
mkdir -p "$RESULTS_DIR"

TOTAL_N=$((N_EDGE_LAN + N_CLOUD_LAN + N_ALT_LAN + N_CLOUD_WAN_MILD + N_CLOUD_WAN_BAD))
log "Total samples across all runs: $TOTAL_N"
log ""

T_START=$(date +%s)

# 1. Edge-only LAN
log "=== [1/5] Edge-only LAN (n=$N_EDGE_LAN) ==="
reset_netem
run_orchestrator "$N_EDGE_LAN" edge-only "e2-edge-lan"

# 2. Cloud-only LAN
log ""
log "=== [2/5] Cloud-only LAN (n=$N_CLOUD_LAN) ==="
run_orchestrator "$N_CLOUD_LAN" cloud-only "e2-cloud-lan"

# 3. Alternate LAN
log ""
log "=== [3/5] Alternate LAN (n=$N_ALT_LAN) ==="
run_orchestrator "$N_ALT_LAN" alternate "e2-alt-lan"

# 4. Cloud-only WAN mild
log ""
log "=== [4/5] Cloud-only WAN mild (n=$N_CLOUD_WAN_MILD) ==="
apply_netem "$NETEM_MILD" "20"
run_orchestrator "$N_CLOUD_WAN_MILD" cloud-only "e2-cloud-wan-mild"
reset_netem

# 5. Cloud-only WAN bad
log ""
log "=== [5/5] Cloud-only WAN bad (n=$N_CLOUD_WAN_BAD) ==="
apply_netem "$NETEM_BAD" "80"
run_orchestrator "$N_CLOUD_WAN_BAD" cloud-only "e2-cloud-wan-bad"
reset_netem

T_END=$(date +%s)
T_ELAPSED=$((T_END - T_START))
log ""
log "=== ALL DONE in $((T_ELAPSED / 60))m $((T_ELAPSED % 60))s ==="
log ""
log "Results files:"
ls -lh "$RESULTS_DIR"/*.jsonl 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || true
log ""
log "Summary:"
python3 - <<'PY'
import csv
try:
    with open("results/summary_runs.csv") as f:
        rows = list(csv.DictReader(f))
    print(f"  {'label':22s} {'policy':11s} {'edge_ok':>7s} {'cloud_ok':>8s} {'drop':>6s} {'lat_disp_p50':>12s}")
    for r in rows:
        print(f"  {r['label']:22s} {r['policy']:11s} {r['edge_ok']:>7s} {r['cloud_ok']:>8s} "
              f"{r['drop_rate']:>6s} {r['latency_dispatch_p50_ms']:>12s}")
except FileNotFoundError:
    print("  (no summary_runs.csv yet)")
PY
