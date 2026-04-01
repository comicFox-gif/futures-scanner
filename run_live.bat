@echo off
echo *** LIVE TRADING MODE - Real money at risk! ***
echo Press Ctrl+C within 5 seconds to cancel...
timeout /t 5
py -3.12 main.py --live %*
pause
