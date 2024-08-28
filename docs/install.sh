# Copy over poetry dependencies from theming repository
cp pytket-docs-theming/extensions_deps/pyproject.toml .
cp pytket-docs-theming/extensio

# Install the docs dependencies. Creates a .venv directory in docs
poetry install

# Editable wheel should be installed separately.