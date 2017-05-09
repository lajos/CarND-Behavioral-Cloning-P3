get-process -name python | Sort-Object -Descending WS | select -first 1 | stop-process
