{
  python3Packages,
  buildPythonPackage,
  fetchFromGitHub,
  cupy',
  wheel,
  cython_0,
  setuptools,
  cuda-bundle,
  substituteAll
}:
buildPythonPackage{
  pname = "cuquantum";
  version = "24.03.0";
  pyproject = true;
  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "cuQuantum";
    rev = "v24.03.0";
    sha256 = sha256:1GPotBCNEsriU+yrWQpIVWKh/wEK8EPWYjn3jF4w8T0=;
  };
  patches = [
    (substituteAll {
      src = ./pycuquantum.patch;
      cuda_bundle = cuda-bundle.out;
    })
  ];
  build-system = [
    setuptools
    wheel
    cython_0
  ];
  dependencies = [
    cupy'
  ];
  dontCheckRuntimeDeps = true;
  CUDA_PATH="${cuda-bundle}";
  CUQUANTUM_ROOT="${cuda-bundle}";
  CUTENSOR_ROOT="${cuda-bundle}";

  unpackPhase = ''
    cp -r $src/python/* .
    chmod -R 0755 .
  '';
}
