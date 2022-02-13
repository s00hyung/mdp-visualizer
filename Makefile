.DEFAULT_GOAL = run
TARGET = app.py

VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip3

venv: 
	python3 -m venv $(VENV)
	${PIP} install --upgrade pip

install: venv
	touch requirements.txt
	${PIP} install -r requirements.txt

lock: venv
	${PIP} freeze > requirements.lock

ignore : 
	touch .gitignore

run:
	touch ${TARGET}
	${PYTHON} ${TARGET}

clean: 
	rm -rf __pycache__
	rm -rf ${VENV}
