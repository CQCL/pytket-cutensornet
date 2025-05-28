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
        sha256 = sha256:6c022ff6235fa92f29ca9729aaba17525785c82d4ac920baec31a586e4df4502;
      };
      "12" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-${version}_cuda12-archive.tar.xz";
        sha256 = sha256:aeabd01e8b43055869f151eca82f70f84c890f559f85aacc0093f949973e6926;
      };
    };
    aarch64-linux = {
      "11" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-sbsa/cuquantum-linux-sbsa-${version}_cuda11-archive.tar.xz";
        sha256 = sha256:a9efea4f37619fbc6136b800460eae83bb1ed14eab7bbeaf0c1587f798081e71;
      };
      "12" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-sbsa/cuquantum-linux-sbsa-${version}_cuda12-archive.tar.xz";
        sha256 = sha256:b2714c1edd276272aa83d94c19eb77d84a138eda8b22638352f262223a087786;
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
