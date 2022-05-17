python -m nuitka voice_smith/entry.py \
    --static-libpython=no \
    --onefile \
    --enable-plugin=torch \
    --enable-plugin=numpy \
    --enable-plugin=multiprocessing \
    --remove-output \
    -o backend_dist/entry.bin
