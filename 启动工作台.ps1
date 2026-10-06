param([switch]$Check)
$workbenchPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workbenchPythonWindow = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $workbenchPython)) {
    throw '缺少核心环境。进入本项目目录后执行：python -m venv .venv，然后 .\.venv\Scripts\python.exe -m pip install -r requirements.txt'
}
if ($Check) {
    Push-Location -LiteralPath $PSScriptRoot
    try { & $workbenchPython -m sim_app.workbench --check }
    finally { Pop-Location }
} else {
    Start-Process -FilePath $workbenchPythonWindow -ArgumentList @('-m', 'sim_app.workbench') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
}
