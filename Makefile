include .env

install_tox:
	uv tool install --python-preference only-managed --python 3.14 tox --with tox-uv --with tox-gh
	uv python install --python-preference only-managed 3.13 3.12 3.11 3.10

run_test:
	uvx tox -r -p