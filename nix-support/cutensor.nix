{ pkgs, cudaPackages, fetchzip }:
let
  package_map = {
    x86_64-linux = fetchzip {
      url = "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-2.0.1.2-archive.tar.xz";
      sha256 = sha256:OPn5577DBwzcGdSovv5tMv+x3wW2QgF5XVmRgm5y148=;
    };
    aarch64-linux = fetchzip {
      url = "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-sbsa/libcutensor-linux-sbsa-2.0.1.2-archive.tar.xz";
      sha256 = sha256:vTEQ7uJYrEC1Q55DgCO5/PXTm0f34ndqezotnqq6E00=;
    };
  };
in pkgs.stdenv.mkDerivation {
  pname = "libcutensor";
  version = "2.0.1.2";
  outputs = [ "out" "lib" "dev" "static" ];
  src = package_map.${pkgs.system};
  installPhase = let
    v = cudaPackages.cudaMajorVersion;
  in ''
    mkdir -p $lib/lib;
    cp -r $src/lib/${v}/*.so $lib/lib;
    cp -r $src/lib/${v}/*.so.* $lib/lib;

    mkdir -p $dev/include;
    cp -r $src/include $dev/include;

    mkdir -p $static/lib;
    cp -r $src/lib/${v}/libcutensorMg_static.a $static/lib;
    cp -r $src/lib/${v}/libcutensor_static.a $static/lib;

    mkdir -p $out;
    cp -r $src/include $out/include;
    cp -r $src/lib/${v} $out/lib;
    cp $src/LICENSE $out/LICENSE;
  '';
}
