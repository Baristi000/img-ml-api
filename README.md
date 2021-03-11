RUN:
    python -m venv env 
    or 
    python3 -m venv env

    source env/bin/activate

    pip install -r globle.txt --upgrade pip

*for raspbery p4

RUN:
    pip install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0-rc2/tensorflow-2.4.0rc2-cp37-none-linux_armv7l.whl

*for google already normal support:

    RUN:
pip install tensorflow