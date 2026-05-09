#!/usr/bin/env bash
set -euo pipefail

# ---------- Fixed paths (override via env if needed) ----------
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.11}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv_mgn}"
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_DIR/scripts/run_train.py}"
# The live config you edit between runs. A frozen copy is taken at submission
# time so queued jobs are never affected by later edits.
CONFIG_SOURCE="${CONFIG_SOURCE:-$PROJECT_DIR/configs/config.json}"

# ---------- Extra sbatch flags after '--' ----------
SBATCH_FLAGS=(); seen_ddash=0
for a in "$@"; do
  if [[ "$a" == "--" ]]; then seen_ddash=1; continue; fi
  if [[ $seen_ddash -eq 1 ]]; then SBATCH_FLAGS+=("$a"); fi
done

# ---------- Make jobs/<timestamp> directory ----------
if [[ -z "${JOB_PATH:-}" ]]; then
  base="jobs/job-$(date +%Y%m%d%H%M%S)"
  JOB_PATH="$base"; i=0
  while [[ -e "$JOB_PATH" ]]; do i=$((i+1)); JOB_PATH="${base}-${i}"; done
fi
mkdir -p "$JOB_PATH"

# ---------- Freeze config at submission time ----------
# Copy the live config into the job directory RIGHT NOW, before sbatch queues
# the job. The job will read this frozen copy via --config, so edits you make
# to configs/config.json while the job is pending have no effect on it.
CONFIG_FROZEN="$JOB_PATH/config_frozen.json"
cp "$CONFIG_SOURCE" "$CONFIG_FROZEN"
echo "Frozen config saved to: $CONFIG_FROZEN"

# ---------- Write args file ----------
ARGSFILE="$JOB_PATH/args.txt"
: > "$ARGSFILE"
printf 'PROJECT_DIR=%s\n' "$PROJECT_DIR" >> "$ARGSFILE"
printf 'PYTHON_MODULE=%s\n' "$PYTHON_MODULE" >> "$ARGSFILE"
printf 'VENV_DIR=%s\n' "$VENV_DIR" >> "$ARGSFILE"
printf 'SCRIPT_PATH=%s\n' "$SCRIPT_PATH" >> "$ARGSFILE"
# Absolute path to the frozen config so the job script can pass it to --config
printf 'CONFIG_FROZEN=%s\n' "$(realpath "$CONFIG_FROZEN")" >> "$ARGSFILE"

# ---------- Create sbatch script ----------
JOB_SCRIPT="$JOB_PATH/sbatch_job.sh"
cat > "$JOB_SCRIPT" <<'SBATCH_EOF'
#!/usr/bin/env bash
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=agoza-ic
#SBATCH --gres=gpu:1
#SBATCH --time=71:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

set -euo pipefail

JOBDIR="$PWD"
exec > >(tee -a "$JOBDIR/job.log") 2>&1

echo "[INFO] $(date -Is) starting on $(hostname)"
echo "[INFO] JOBDIR=$JOBDIR"

if [[ -f "$JOBDIR/args.txt" ]]; then
  while IFS= read -r kv; do
    case "$kv" in
      *=*) export "$kv" ;;
    esac
  done < "$JOBDIR/args.txt"
fi

module purge
module load "$PYTHON_MODULE"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

nvidia-smi
python "$SCRIPT_PATH" --config "$CONFIG_FROZEN"

echo "[INFO] $(date -Is) done"
SBATCH_EOF
chmod +x "$JOB_SCRIPT"

# ---------- Submit ----------
JOBID=$(sbatch --parsable \
  --chdir="$JOB_PATH" \
  --job-name="mgn_train" \
  --export=ALL \
  "${SBATCH_FLAGS[@]}" \
  "$JOB_SCRIPT")

echo "$JOB_PATH"
echo "JOBID=$JOBID" > "$JOB_PATH/jobid"
echo "Submitted as JobID: $JOBID"
echo "Follow with:"
echo "  squeue -j $JOBID"
echo "  tail -f $JOB_PATH/job.log"