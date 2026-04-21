python -m venv .venvE3
& .\.venvE3\Scripts\Activate.ps1
python -m pip install --upgrade pip
if (Test-Path .\requirements.txt) {
    python -m pip install -r .\requirements.txt
}
Write-Output "Venv ready. Start with: python .\main.py"
Write-Output "Or specify signals directly: python .\main.py --signal VehV_v VehAX_ax"
