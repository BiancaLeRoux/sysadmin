Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
& ".venv\Scripts\python.exe" -m avatarcam.bench @args
