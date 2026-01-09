param(
    [string]$VenvPath = ".venv",
    [string]$Wheelhouse = "wheelhouse"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating venv at $VenvPath"
python -m venv $VenvPath

Write-Host "Activating venv"
& "$VenvPath\Scripts\Activate.ps1"

Write-Host "Upgrading pip"
python -m pip install --upgrade pip

Write-Host "Creating wheelhouse at $Wheelhouse"
New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null

Write-Host "Downloading CUDA 11.8 PyTorch wheels"
pip download torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -d $Wheelhouse

Write-Host "Downloading project requirements"
pip download -r requirement.txt -d $Wheelhouse

Write-Host "Done"
