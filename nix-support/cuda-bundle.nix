{
  symlinkJoin,
  cudaPackages,
}:
let
  cutensor' = cudaPackages.callPackage ./cutensor.nix {};
  cuquantum' = cudaPackages.callPackage ./cuquantum.nix {};
in
symlinkJoin {
  name = "cuda-bundle-${cudaPackages.cudaVersion}";
  paths = with cudaPackages; [
    cudatoolkit
    cuda_nvprof
    cuda_cudart.stubs
    cuda_cudart.static
    cutensor'
    cutensor'.dev
    cuquantum'
  ];
  postBuild = ''
    ln -sfn lib $out/lib64
  '';
}

