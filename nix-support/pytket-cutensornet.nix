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
  in super.writeShellScriptBin "run-pytket-cutensornet-tests" ''
      HOME=$(mktemp -d);
      export HOME;
      echo "---------------------------";
      env;
      echo "---------------------------";
      ${test-env}/bin/pytest -s ${../tests};
  '';
}
