{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  cython_0,
  fastrlock,
  numpy,
  wheel,
  pytestCheckHook,
  mock,
  setuptools,
  cudaPackages,
  addOpenGLRunpath,
  pythonOlder,
  symlinkJoin,
  substituteAll,
  cuda-bundle
}:

let
  inherit (cudaPackages) cudnn nccl;
  cupy-cccl-src = fetchFromGitHub{
    owner = "cupy";
    repo = "cccl";
    rev = "79ed0e96e35112d171e43f13fa7f324eff7f3de0";
    hash = "sha256-8YlIp7xOQtB7dKhCL7SIX+cKOvP4V/XPx6EC+Ct4vSc=";
  };
  dlpack-src = fetchFromGitHub{
    owner = "dmlc";
    repo = "dlpack";
    rev = "365b823cedb281cd0240ca601aba9b78771f91a3";
    hash = "sha256-IcfCoz3PfDdRetikc2MZM1sJFOyRgKonWMk21HPbrso=";
  };
  jitify-src = fetchFromGitHub{
    owner = "NVIDIA";
    repo = "jitify";
    rev = "1a0ca0e837405506f3b8f7883bacb71c20d86d96";
    sha256 = "sha256-m+ZrYbmA/FuKZVIUiLYX5GR79jheu5CNHtli/X+ux0U=";
  };
in
buildPythonPackage rec {
  pname = "cupy-cuda12x";
  version = "13.1.0b";
  format = "setuptools";

  disabled = pythonOlder "3.7";

  src = fetchFromGitHub {
    owner = "cupy";
    repo = "cupy";
    rev = "6730353d611f4fd5f6a7494e90c7639a35245337"; # cusparse_lt 0.6.1 support
    hash = sha256:h6Q6abnq5zPY6mx8dA5l1kXfKjisEOwEqP0rGGu9q0M=;
  };
  patches = [
    (substituteAll {
      src = ./cupy.patch;
      cuda_bundle = cuda-bundle.out;
    })
  ];

  unpackPhase = ''
    for f in $src/*; do
      if [ "$f" != "$src/third_party" ]; then
        cp -r $f .;
      fi
    done
    mkdir third_party;
    ln -s ${dlpack-src} third_party/dlpack;
    ln -s ${jitify-src} third_party/jitify;
    ln -s ${cupy-cccl-src} third_party/cccl;
    chmod -R 0775 .;
  '';

  # See https://docs.cupy.dev/en/v10.2.0/reference/environment.html. Seting both
  # CUPY_NUM_BUILD_JOBS and CUPY_NUM_NVCC_THREADS to NIX_BUILD_CORES results in
  # a small amount of thrashing but it turns out there are a large number of
  # very short builds and a few extremely long ones, so setting both ends up
  # working nicely in practice.
  preConfigure = ''
    export CUPY_NUM_BUILD_JOBS="$NIX_BUILD_CORES"
    export CUPY_NUM_NVCC_THREADS="$NIX_BUILD_CORES"
  '';

  nativeBuildInputs = [
    cuda-bundle
    setuptools
    wheel
    addOpenGLRunpath
    cython_0
  ];

  buildInputs = [
    cuda-bundle
    cudnn
    nccl
  ];

  NVCC="${cuda-bundle}/bin/nvcc";
  CUDA_PATH="${cuda-bundle}";
  LDFLAGS="-L${cuda-bundle}/lib";
  LD_LIBRARY_PATH="${cuda-bundle}/lib";

  propagatedBuildInputs = [
    fastrlock
    numpy
  ];

  nativeCheckInputs = [
    pytestCheckHook
    mock
  ];

  # Won't work with the GPU, whose drivers won't be accessible from the build
  # sandbox
  doCheck = false;

  postFixup = ''
    find $out -type f \( -name '*.so' -or -name '*.so.*' \) | while read lib; do
      addOpenGLRunpath "$lib"
      patchelf --set-rpath ${cuda-bundle}/lib $lib;
    done
  '';

  enableParallelBuilding = true;

  meta = with lib; {
    description = "A NumPy-compatible matrix library accelerated by CUDA";
    homepage = "https://cupy.chainer.org/";
    changelog = "https://github.com/cupy/cupy/releases/tag/v${version}";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
    maintainers = with maintainers; [ hyphon81 ];
  };
}
