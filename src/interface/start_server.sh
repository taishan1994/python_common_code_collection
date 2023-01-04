# nohup python -u server.py --host "0.0.0.0" --port "5000" > server.log 2>&1 &
nohup python -u server.py --host "0.0.0.0" --port "5000" --use_service_streamer > server.log 2>&1 &