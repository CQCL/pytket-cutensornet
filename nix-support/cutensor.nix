{ pkgs, cudaPackages, fetchzip }:
let
  package_map = {
    x86_64-linux = fetchzip {
      url = "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/libcutensor-linux-x86_64-2.0.2.5-archive.tar.xz";
      sha256 = sha256:06ygtOh3potihT0bm0lhytKpf7zWOA8Ap9rATNvQdTk=;
    };
    aarch64-linux = fetchzip {
      url = "https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-sbsa/libcutensor-linux-sbsa-2.0.2.5-archive.tar.xz";
      sha256 = sha256:CRewkAlXYklDnj6iF6eGv0C8yxHVDOdj+mXqm2dN+po=;
    };
  };
in pkgs.stdenv.mkDerivation {
  pname = "libcutensor";
  version = "2.0.2.5";
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
