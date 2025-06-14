#!/bin/bash

while IFS= read -r line
do
  echo "Running: python main.py $line"
  python main.py $line
done < schedule.txt
