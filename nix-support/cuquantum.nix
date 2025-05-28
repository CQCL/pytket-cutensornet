{
  pkgs,
  system,
  cudaPackages,
  fetchzip,
  stdenv,
  patchelf,
  #cuda-bundle,
}:
let
  version = "25.03.0.11";
  package_map = {
    x86_64-linux = {
      "11" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-${version}_cuda11-archive.tar.xz";
        sha256 = "MEEEEEEH";
      };
      "12" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-${version}_cuda12-archive.tar.xz";
        sha256 = sha256:TOQX8e2MgefF+FWm0FvAMo7aFlz9S/aPEveeM4eK1Qs=;
      };
    };
    aarch64-linux = {
      "11" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-sbsa/cuquantum-linux-sbsa-${version}_cuda11-archive.tar.xz";
        sha256 = "BOOOOOOH";
      };
      "12" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-sbsa/cuquantum-linux-sbsa-${version}_cuda12-archive.tar.xz";
        sha256 = "PLEEEEEASE";
      };
    };
  };
in
  stdenv.mkDerivation {
    pname = "cuquantum";
    inherit version;
    src = package_map.${system}.${cudaPackages.cudaMajorVersion};
    nativeBuildInputs = [ pkgs.patchelf ];
    installPhase = ''
      mkdir -p $out
      cp -r $src/* $out
      chmod -R 0755 $out
    '';
  }
