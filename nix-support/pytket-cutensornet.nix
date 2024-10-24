self: super:
let
  metadata = builtins.readFile ../_metadata.py;
  versions =
    builtins.match ''.*_version__ *= *["']([^"']+)["'].*'' metadata;
  version = if builtins.length versions > 0 then
    builtins.elemAt versions 0
  else
    builtins.trace "Warning: unable to find version. Defaulting to 0.0.0" "0.0.0";
in {
  pytket-cutensornet = super.python3Packages.buildPythonPackage {
    pname = "pytket-cutensornet";
    version = version;
    src = super.stdenv.mkDerivation{
      name = "pytket-cutensornet-sources";
      phases = [ "installPhase" ];
      installPhase = ''
        mkdir -p $out;
        cp -r ${../pytket} $out/pytket;

        cp ${../setup.py} $out/setup.py;
        cp ${../README.md} $out/README.md; # required for setup's long description
        cp ${../pytest.ini} $out/pytest.ini;
        cp ${../_metadata.py} $out/_metadata.py;
        
        # on nix versions of scipy and ipython, stubs are missing.
        # adjust mypy.ini to ignore these errors.
        (
          cat ${../mypy.ini};
          cat <<EOF
[mypy-scipy.*]
ignore_missing_imports = True
ignore_errors = True

[mypy-IPython.display.*]
ignore_missing_imports = True
ignore_errors = True
EOF
        ) >> $out/mypy.ini;
      '';
    };
    propagatedBuildInputs = [ super.pytket super.pycuquantum ];
      
    doCheck = true;
    checkInputs = with super.python3Packages; [ mypy ];
    checkPhase = ''
      python -m mypy --config-file=mypy.ini --no-incremental -p pytket
      # running pytest requires GPU access which isn't currently possible (afaik) in the nix sandbox
    '';
  };

  run-pytket-cutensornet-tests = let
    test-env = super.python3.withPackages(ps: with ps; [
      self.pytket-cutensornet
      pytest
      pytest-lazy-fixture
    ]);
    nixgl-bin = self.lib.getExe self.nixgl.auto.nixGLNvidia;
  in super.writeShellScriptBin "run-pytket-cutensornet-tests" ''
      HOME=$(mktemp -d);
      export HOME;
      NIXGL_PATH="$(${nixgl-bin} printenv LD_LIBRARY_PATH)";
      WSL_PATH="/usr/lib/wsl/lib";
      LD_LIBRARY_PATH="$NIXGL_PATH:$WSL_PATH:$LD_LIBRARY_PATH";
      export LD_LIBRARY_PATH;
      ${test-env}/bin/pytest -s ${../tests};
  '';
  run-pytket-cutensornet-examples = let
    example-env = super.python3.withPackages(ps: with ps; [
      self.pytket-cutensornet
      matplotlib
      numpy
      networkx
      ipython
      nbmake
      pytest
    ]);
    nixgl-bin = self.lib.getExe self.nixgl.auto.nixGLNvidia;
  in super.writeShellScriptBin "run-pytket-cutensornet-examples" ''
      HOME=$(mktemp -d);
      export HOME;
      NIXGL_PATH="$(${nixgl-bin} printenv LD_LIBRARY_PATH)";
      WSL_PATH="/usr/lib/wsl/lib";
      LD_LIBRARY_PATH="$NIXGL_PATH:$WSL_PATH:$LD_LIBRARY_PATH";
      
      example_dir=${../examples};
      set -e;
      for name in `cat ''${example_dir}/ci-tested-notebooks.txt`;
      do
          ${example-env}/bin/pytest \
            --nbmake \
            -p no:cacheprovider \
            $example_dir/$name.ipynb;
      done;
  '';
}
