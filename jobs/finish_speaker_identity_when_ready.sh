#!/usr/bin/env bash
# Watcher: wait for Stage A (diarization) to be ACTUALLY COMPLETE over the whole
# corpus, then run Stage B (embed + global cluster at the locked
# GLOBAL_THRESH=0.35) and Stage C (re-merge every final JSON with speaker
# identities). Lets the whole speaker-identity rollout complete autonomously
# after the multi-hour diarization grind.
#
# Why this is more than "no worker alive": the diarizer globs the mp3 list once
# at startup and, on any decode error (e.g. files being rewritten by the
# recover_empty_mp3s job), writes no .txt and never retries within that run. So
# "workers exited" != "corpus diarized". We instead gate on real coverage
# (#mp3 - #txt) and auto-relaunch the idempotent diarizer to mop up gaps before
# firing B/C. A relaunch re-globs (all files present now) and skips done stems.
set -uo pipefail

REPO="/lab/kiran/transcript-pipeline"
LOG="$REPO/outputs/finish_speaker_identity.log"
AUDIO_ROOT="/lab/kiran/tbv_mp3"
DIAR_DIR="$REPO/outputs/05_pyannote_diarization"

# Proceed to B/C once at most this many mp3s lack a .txt. Some files are
# permanently undecodable; if a relaunch can't shrink the gap below this, we
# stop relaunching and proceed anyway (see stall detection below).
TOLERANCE="${DIAR_TOLERANCE:-5}"
MAX_RELAUNCHES="${DIAR_MAX_RELAUNCHES:-4}"

alive() { pgrep -f "diarization_from_segments_local.py" >/dev/null 2>&1; }
n_mp3()  { ls "$AUDIO_ROOT"/*.mp3 2>/dev/null | wc -l; }
n_txt()  { ls "$DIAR_DIR"/*.txt 2>/dev/null | wc -l; }
log()    { echo "[watcher] $* $(date '+%F %T')" >> "$LOG"; }

wait_for_idle() {
    # Require diarization absent for 5 consecutive checks (5 min) so we don't
    # fire between two files.
    local clear=0
    while true; do
        if alive; then clear=0; else clear=$((clear + 1)); [ "$clear" -ge 5 ] && return; fi
        sleep 60
    done
}

log "started; gating on full diarization coverage (tolerance=$TOLERANCE)"

relaunches=0
prev_gap=-1
while true; do
    wait_for_idle
    gap=$(( $(n_mp3) - $(n_txt) ))
    log "diarization idle: $(n_txt) txt / $(n_mp3) mp3 (gap=$gap)"

    [ "$gap" -le "$TOLERANCE" ] && { log "coverage complete (gap=$gap<=$TOLERANCE)"; break; }

    # Stall detection: if the last relaunch didn't shrink the gap, the remainder
    # is undecodable -> stop chasing it and proceed.
    if [ "$prev_gap" -ge 0 ] && [ "$gap" -ge "$prev_gap" ]; then
        log "gap did not shrink ($prev_gap -> $gap); remaining files appear undecodable, proceeding"
        break
    fi
    if [ "$relaunches" -ge "$MAX_RELAUNCHES" ]; then
        log "hit MAX_RELAUNCHES=$MAX_RELAUNCHES with gap=$gap; proceeding anyway"
        break
    fi

    prev_gap=$gap
    relaunches=$((relaunches + 1))
    log "relaunch #$relaunches of diarizer to fill gap=$gap"
    nohup bash "$REPO/jobs/diarize_corpus.sh" >> "$REPO/outputs/diarize_corpus.log" 2>&1 &
    sleep 120  # let workers spin up so wait_for_idle doesn't fall through
done

log "diarization done: $(n_txt) txt files. Running Stage B + C"
export GLOBAL_THRESH=0.35
export EMB_CACHE_DIR="$REPO/outputs/07_speaker_clusters/emb_cache"
export MAIN_SPEAKER_NAME="Vaisesika Dasa"
bash "$REPO/jobs/run_speaker_identity.sh" B C >> "$LOG" 2>&1
log "complete"
