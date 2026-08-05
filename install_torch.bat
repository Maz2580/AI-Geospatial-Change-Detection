@echo off
subst Z: "C:\Users\maz.ghasemi\Downloads\Maz - 2 July 2025\python\change detection"
Z:
cd \
.\venv\Scripts\python -m pip install torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cpu
C:
subst Z: /D
