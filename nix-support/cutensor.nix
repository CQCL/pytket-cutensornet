{ pkgs, cudaPackages, fetchzip }:
let
  package_map = {
    x86_64-linux = fetchzip {
      url = "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-2.2.0.0-archive.tar.xz";
      sha256 = sha256:dzim4HMd49pwAP52JinxI7x7I72CWq4CZljZytVJl1c=;
    };
    aarch64-linux = fetchzip {
      url = "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-sbsa/libcutensor-linux-sbsa-2.2.0.0-archive.tar.xz";
      sha256 = sha256:h3xT181Ti/MffrGCBbexwNB1KhXgBRBH2nl0ntuy3B8=;
    };
  };
in pkgs.stdenv.mkDerivation {
  pname = "libcutensor";
  version = "2.2.2.5";
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
