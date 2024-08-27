# Building the docs

1. Firstly ensure you are in the `docs` directory

```shell
cd docs
```

2. Next, update the pytket-docs-theming submodule

If you are doing this for the first time use the following

```shell
git submodule add -b main https://github.com/CQCL/pytket-docs-theming.git
```

If you are coming back to this repository after some time, ensure that you are using the latest version of the submodule

```shell
git submodule update --init --recursive
```

3. Once you are on the latest version of the submodule, build the docs using the `build-docs.sh` script.

`TODO:` Decide where dependencies are stored. I, (Callum) would prefer if the sphinx dependencies are stored in `pytket-docs-theming` in a `pyproject.toml` file. These could then be shared in the submodule and installed for the local build. Finally an editable wheel could be installed to build the dev docs.  

```shell
./build-docs.sh
```

The html pages will then show up in the `docs/build` directory.

4. Finally, serve the html locally (OPTIONAL). This allows you to view the navbar in the docs.

```shell
npx serve build
```