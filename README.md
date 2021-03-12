RUN: python -m venv env 
or 
RUN: python3 -m venv env

RUN: source env/bin/activate

RUN: pip install -r global.txt --upgrade pip

*for raspbery p4

RUN:pip install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0-rc2/tensorflow-2.4.0rc2-cp37-none-linux_armv7l.whl

*for google already normal support:

RUN: pip install tensorflow

RUN: mkkir train_ds predict_ds raw_data && python main.py