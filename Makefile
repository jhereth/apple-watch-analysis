PYTHON = python3.10
VENV = .direnv/python-3.10

${VENV}/bin/python3:
	${PYTHON} -m venv .venv
	${VENV}/bin/python3 -m pip install -r requirements-bootstrap.txt
	source ${VENV}/bin/activate

.PHONY: dev
dev: ${VENV}/bin/python3 requirements.in
	${VENV}/bin/pip-compile -v requirements.in
	${VENV}/bin/pip-sync
