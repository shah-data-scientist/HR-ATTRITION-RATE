netstat -ano | findstr :8000 | for /f "tokens=5" %a in ('findstr :8000') do taskkill /F /PID %a
