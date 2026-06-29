python -m venv venv
source venv/bin/activate
pip cache purge
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install numpy matplotlib jupyter ipywidgets
PYGAME_DETECT_AVX2=1 pip install pygame --no-build-isolation