#!/bin/bash

while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  python main.py $line || echo "[ERROR] Failed: $line" >&2
done < schedule.txt
