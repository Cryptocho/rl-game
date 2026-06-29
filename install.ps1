python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install numpy matplotlib jupyter ipywidgets
$env:PYGAME_DETECT_AVX2 = "1"
pip install pygame --no-build-isolation
