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
  version = "24.03.0.4";
  package_map = {
    x86_64-linux = {
      "11" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-${version}_cuda11-archive.tar.xz";
        sha256 = sha256:tvUEKdUp3dluY7uzm6NJBZxcjJsVhMtwJmV061vR0XU=;
      };
      "12" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/cuquantum-linux-x86_64-${version}_cuda12-archive.tar.xz";
        sha256 = sha256:ws9dPv4Tkdu9XSE4zI3WdFccQuhiycrEchduzPNBiCc=;
      };
    };
    aarch64-linux = {
      "11" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-sbsa/cuquantum-linux-sbsa-${version}_cuda11-archive.tar.xz";
        sha256 = sha256:Wc0HgfnDgTpyDKMQpnJdJ95FkBe+7XAqQ48XkkrWiKU=;
      };
      "12" = fetchzip {
        url = "https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-sbsa/cuquantum-linux-sbsa-${version}_cuda12-archive.tar.xz";
        sha256 = sha256:ZnjnMja0565Mi9GE6IE3uDXXY1jisz/VS1Yl7nZSaVw=;
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
    #fixupPhase = ''
    #  for f in $out/lib/*.so; do
    #    patchelf --set-rpath ${cuda-bundle}/lib $f;
    #  done;
    #'';
  }
