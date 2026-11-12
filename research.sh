#!/usr/bin/env bash
# Dense-model runs for the cache paper. Usage (in Ubuntu, from anywhere):
#   bash research.sh setup     # copy files in, get Qwen model, make long prompt  (~2 min)
#   bash research.sh build     # build all ChampSim configs                       (~15-20 min)
#   bash research.sh launch    # start every job in the background                (returns at once)
#   bash research.sh status    # what's running / what's finished
#   bash research.sh collect   # zip all results into Downloads
set -euo pipefail

WIN_DL=${WIN_DL:-"/mnt/c/Users/UTKARSH SINGH/Downloads"}
WIN_MODELS=${WIN_MODELS:-"/mnt/c/llama/llama-b11177-bin-win-cpu-x64/models"}
CS=~/ChampSim
TR=~/traces
LLAMA=~/llama.cpp/build/bin/llama-simple
PIN_ROOT=${PIN_ROOT:-$(ls -d ~/pin-3.22-* 2>/dev/null | head -1 || true)}
TRACER=$CS/tracer/pin/obj-intel64/champsim_tracer.so
SMOL=~/models/SmolLM2-135M-Instruct-Q4_0.gguf
QWEN=~/models/Qwen2.5-0.5B-Instruct-Q4_0.gguf
SMOL_W=${SMOL_W:-136000000}   # existing SmolLM trace: 2 tokens warmup, 4 measured
SMOL_S=${SMOL_S:-270000000}

# name  sets  ways  policy          (sets x 12 ways x 64 B)
CONFIGS="
llc24         32768  12 lru
llc24_srrip   32768  12 srrip
llc24_drrip   32768  12 drrip
llc24_ship    32768  12 ship
llc24_hawkeye 32768  12 hawkeye
llc24_log     32768  12 llclog
llc12_lru     16384  12 lru
llc12_srrip   16384  12 srrip
llc48_lru     65536  12 lru
llc48_srrip   65536  12 srrip
llc48_log     65536  12 llclog
llc96_lru     131072 12 lru
llc96_srrip   131072 12 srrip
"

sim() {  # sim <binary-suffix> <trace> <warmup> <simulate> <out.txt>
  "$CS/bin/champsim_$1" --warmup-instructions "$3" --simulation-instructions "$4" "$2" > "$5" 2>&1
}

trace() {  # trace <model> <out.xz> <skip> <tokens> <prompt>
  local fifo="/tmp/$(basename "$2").fifo"
  rm -f "$fifo" && mkfifo "$fifo"
  xz -T4 -c < "$fifo" > "$2" &
  TRACE_SKIP=$3 TRACE_TOKENS=$4 "$PIN_ROOT/pin" -t "$TRACER" -o "$fifo" \
    -start_symbol __champsim_start_trace -stop_symbol __champsim_stop_trace -t 100000000000 \
    -- "$LLAMA" -m "$1" -n 16 "$5" > "${2%.champsim.xz}_trace_log.txt" 2>&1
  wait
  rm -f "$fifo"
  xz -t "$2"
}

ninstr() { grep -oP 'after \K[0-9]+' "$1" | tail -1; }

belady() {  # belady <result-name> <llclog-binary> <trace> <warmup> <simulate>: exact Belady on that run
  LLCLOG_PATH="$TR/$1.bin" sim "$2" "$3" "$4" "$5" "$TR/sim_$1.txt"
  if python3 "$TR/belady_offline.py" "$TR/$1.bin" "$TR/sim_$1.txt" 12 > "$TR/belady_$1.txt" 2>&1; then
    echo "$1: $(grep -h 'LRU replay\|Belady:' "$TR/belady_$1.txt" | tr '\n' ' ')"
    rm -f "$TR/$1.bin"
  else
    echo "$1: BELADY FAILED (log kept: $TR/$1.bin, see $TR/belady_$1.txt)"
  fi
}

case "${1:-}" in
setup)
  mkdir -p "$CS/replacement/llclog" "$TR" ~/models
  cp "$WIN_DL/llclog.h" "$WIN_DL/llclog.cc" "$CS/replacement/llclog/"
  cp "$WIN_DL/belady_offline.py" "$TR/"
  [ -f "$CS/replacement/hawkeye/hawkeye.cc" ] || { echo "missing hawkeye files in $CS/replacement/hawkeye"; exit 1; }
  if [ ! -f "$QWEN" ]; then
    cp "$WIN_MODELS/Qwen2.5-0.5B-Instruct-Q4_0.gguf" "$QWEN" 2>/dev/null ||
      wget -q -O "$QWEN" https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_0.gguf
  fi
  python3 - "$TR/long_prompt.txt" <<'EOF'
import sys
topics = ["the history of the printing press", "how bridges carry load", "the water cycle",
          "the invention of the telescope", "how vaccines work", "the economics of trade",
          "the structure of the atom", "the causes of the industrial revolution"]
parts = []
for i in range(24):
    t = topics[i % len(topics)]
    parts.append(f"Note {i + 1}: This section discusses {t}. It covers the key ideas, the people involved, "
                 f"and why {t} still matters today, with examples from several countries and time periods.")
text = " ".join(parts) + " Using all of the notes above, write a careful summary:"
open(sys.argv[1], "w").write(text)
print(len(text.split()), "words")
EOF
  python3 -c "import numpy" && echo "SETUP OK"
  ;;
build)
  cd "$CS"
  echo "$CONFIGS" | while read -r name sets ways pol; do
    [ -z "$name" ] && continue
    if [ -x "bin/champsim_$name" ]; then echo "have $name"; continue; fi
    echo "{ \"executable_name\": \"champsim_$name\", \"LLC\": { \"sets\": $sets, \"ways\": $ways, \"replacement\": \"$pol\" } }" > "cfg_$name.json"
    ./config.sh "cfg_$name.json" > /dev/null && make -j"$(nproc)" > "build_$name.log" 2>&1 && echo "built $name" || { echo "BUILD FAILED: $name (see $CS/build_$name.log)"; exit 1; }
  done
  echo "BUILD OK"
  ;;
launch)
  cd "$TR"
  for j in job_smol job_qwen job_long; do
    nohup bash "$0" "$j" > "$TR/$j.log" 2>&1 &
    echo "started $j (log: $TR/$j.log)"
  done
  ;;
job_smol)  # SmolLM2 Q4: LLC-size sweep + exact Belady at 24 and 48 MB (existing trace)
  T=$TR/smol_q4.champsim.xz
  for c in llc12_lru llc12_srrip llc48_lru llc48_srrip llc96_lru llc96_srrip; do
    sim "$c" "$T" "$SMOL_W" "$SMOL_S" "$TR/sim_smol_$c.txt" &
  done
  belady smol_llc24_log llc24_log "$T" "$SMOL_W" "$SMOL_S" &
  belady smol_llc48_log llc48_log "$T" "$SMOL_W" "$SMOL_S" &
  wait
  echo JOB_SMOL_DONE
  ;;
job_qwen)  # Qwen2.5-0.5B Q4: new trace (2 skipped, 4 traced), 1 token warmup, 3 measured
  T=$TR/qwen_q4.champsim.xz
  [ -f "$T" ] || trace "$QWEN" "$T" 2 4 "The meaning of life is"
  N=$(ninstr "$TR/qwen_q4_trace_log.txt"); W=$((N / 4)); S=$((N - W - N / 50))
  echo "qwen N=$N W=$W S=$S"
  for c in llc24 llc24_srrip llc24_drrip llc24_ship llc24_hawkeye llc96_lru llc96_srrip; do
    sim "$c" "$T" "$W" "$S" "$TR/sim_qwen_$c.txt" &
  done
  belady qwen_llc24_log llc24_log "$T" "$W" "$S" &
  wait
  echo JOB_QWEN_DONE
  ;;
job_long)  # SmolLM2 Q4 with a ~1K-token prompt, so the KV cache is LLC-sized
  T=$TR/long_q4.champsim.xz
  [ -f "$T" ] || trace "$SMOL" "$T" 2 6 "$(cat "$TR/long_prompt.txt")"
  N=$(ninstr "$TR/long_q4_trace_log.txt"); W=$((N / 3)); S=$((N - W - N / 50))
  echo "long N=$N W=$W S=$S"
  for c in llc24 llc24_srrip llc24_hawkeye; do
    sim "$c" "$T" "$W" "$S" "$TR/sim_long_$c.txt" &
  done
  belady long_llc24_log llc24_log "$T" "$W" "$S" &
  wait
  echo JOB_LONG_DONE
  ;;
status)
  echo "--- running simulations / traces:"
  pgrep -af "champsim_|pin |belady_offline" | grep -v pgrep | cut -c1-110 || echo "(none)"
  echo "--- jobs finished:"
  grep -h "_DONE" "$TR"/job_*.log 2>/dev/null || echo "(none yet)"
  echo "--- errors in job logs:"
  grep -ihE "error|fail|no such|mismatch" "$TR"/job_*.log 2>/dev/null | head || true
  ;;
collect)
  cd "$TR"
  zip -q results_day2.zip sim_*.txt belady_*.txt *_trace_log.txt job_*.log 2>/dev/null
  cp results_day2.zip "$WIN_DL/" && echo "copied results_day2.zip to Downloads"
  ;;
*)
  sed -n '2,8p' "$0"
  ;;
esac
