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
        sha256 = sha256:bAIv9iNfqS8pypcpqroXUleFyC1KySC67DGlhuTfRQI=;
      };
      "12" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-${version}_cuda12-archive.tar.xz";
        sha256 = sha256:rqvQHotDBVhp8VHsqC9w+EyJD1WfharMAJP5SZc+aSY=;
      };
    };
    aarch64-linux = {
      "11" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-sbsa/cuquantum-linux-sbsa-${version}_cuda11-archive.tar.xz";
        sha256 = sha256:qe/qTzdhn7xhNrgARg6ug7se0U6re76vDBWH95gIHnE=;
      };
      "12" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-sbsa/cuquantum-linux-sbsa-${version}_cuda12-archive.tar.xz";
        sha256 = sha256:snFMHt0nYnKqg9lMGet32EoTjtqLImODUvJiIjoId4Y=;
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
