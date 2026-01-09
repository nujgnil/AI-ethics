param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating venv at $VenvPath"
python -m venv $VenvPath

Write-Host "Activating venv"
& "$VenvPath\Scripts\Activate.ps1"

Write-Host "Upgrading pip"
python -m pip install --upgrade pip

Write-Host "Installing CUDA 11.8 PyTorch"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Write-Host "Installing project requirements"
pip install -r requirement.txt

Write-Host "Done"
