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

Write-Host "Installing CUDA 11.8 PyTorch from wheelhouse"
pip install --no-index --find-links $Wheelhouse torch torchvision torchaudio

Write-Host "Installing project requirements from wheelhouse"
pip install --no-index --find-links $Wheelhouse -r requirement.txt

Write-Host "Done"
