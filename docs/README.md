# Building the docs

The configuration for the docs build is shared across different repositories by using [pytket-docs-theming](https://github.com/CQCL/pytket-docs-theming) as a submodule. This repository also contains the dependencies for building the docs for the pytket extensions. If you encounter issues related to docs configuration, feel free to open an [issue](https://github.com/CQCL/pytket-docs-theming/issues).

Unfamiliar with submodules? Check out this [github blog post](https://github.blog/open-source/git/working-with-submodules/). 

1. First, update/initialise the `pytket-docs-theming` submodule. (Make sure you are in the `docs` directory when you do this).

If you are doing this time, the submodule can be added as follows

```shell
cd docs
git submodule add -b main https://github.com/CQCL/pytket-docs-theming.git
```

If you are coming back to this repository after some time, ensure that you are using the latest version of the submodule

```shell
git submodule update --init --recursive
```

2. Once the submodule is updated, install the docs dependencies using the `install.sh` script.

```shell
bash install.sh
```

This will create a `.venv` folder in the `docs` directory with all of the requirements to build the docs. Note that this does not yet include an editable wheel of the project.

3. Next, activate the virtual environment and install the editable wheel from the repository root.

```shell
poetry shell
cd ..
pip install -e .
```

4. Next, run the `build-docs.sh` script from the `docs` directory.

```shell
cd docs
bash build-docs.sh
```

The built html pages will then show up in the `docs/build` directory.

5. Finally, serve the html locally (OPTIONAL). This allows you to view the navbar in the docs.

```shell
npx serve build
```

Requirements for step 5.
* [Install nodejs](https://nodejs.org/en/download/package-manager)
* Install npx with `npm i -g npx`