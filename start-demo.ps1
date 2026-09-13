$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Нужен Python 3.11. См. README.md.' }
    # truststore is required on networks doing TLS interception (self-signed CA in chain).
    $env:PIP_USE_FEATURE = 'truststore'
    & $taskPython -m pip install -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить зависимости.' }
}
# src-layout: the package must be importable. Editable install keeps it live.
& $taskPython -c "import recagent" 2>$null
if ($LASTEXITCODE -ne 0) {
    $env:PIP_USE_FEATURE = 'truststore'
    & $taskPython -m pip install -e . --no-deps
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить пакет recagent (editable).' }
}
& $taskPython -m streamlit run app.py --server.headless=false