{
  symlinkJoin,
  cudaPackages,
}:
let
  cutensor' = cudaPackages.callPackage ./cutensor.nix {};
  cuquantum' = cudaPackages.callPackage ./cuquantum.nix {};
  cusparselt' = cudaPackages.callPackage ./cusparselt.nix {};
in
symlinkJoin {
  name = "cuda-bundle-${cudaPackages.cudaVersion}";
  paths = with cudaPackages; [
    cuda_cccl # <nv/target>
    cuda_cccl.dev
    cuda_cudart
    cuda_nvcc.dev # <crt/host_defines.h>
    cuda_nvcc
    cuda_nvprof
    cuda_nvrtc
    cuda_nvtx
    cuda_profiler_api
    libcublas
    libcufft
    libcurand
    libcusolver
    libcusparse
    cusparselt'
    cusparselt'.dev
    cutensor'
    cutensor'.dev
    cuquantum'
  ];
  postBuild = ''
    ln -sfn lib $out/lib64
  '';
}

