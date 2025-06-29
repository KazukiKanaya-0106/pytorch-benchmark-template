@echo off
for /F "tokens=*" %%A in (schedule.txt) do (
    echo Running: python main.py %%A
    python main.py %%A
)
