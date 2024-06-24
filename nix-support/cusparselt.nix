{ pkgs, cudaPackages, fetchzip }:
let
  package_map = {
    x86_64-linux = fetchzip {
      url = "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-x86_64/libcusparse_lt-linux-x86_64-0.6.1.0-archive.tar.xz";
      sha256 = sha256:+CuYbJ2Sac3kmzdrJQLvdl2oGehx3b6kQqcyHksYxyU=;
    };
    aarch64-linux = fetchzip {
      url = "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.6.1.0-archive.tar.xz";
      sha256 = sha256:dcWXaXT9Q0370QEcYLnnmHRSppU1yn9VR4w9hzV+7jU=;
    };
  };
in pkgs.stdenv.mkDerivation {
  pname = "libcusparse_lt";
  version = "0.6.1.0";
  outputs = [ "out" "lib" "dev" "static" ];
  src = package_map.${pkgs.system};
  installPhase = ''
    mkdir -p $lib/lib;
    cp -r $src/lib/*.so $lib/lib;
    cp -r $src/lib/*.so.* $lib/lib;

    mkdir -p $dev/include;
    cp -r $src/include $dev/include;

    mkdir -p $static/lib;
    cp -r $src/lib/libcusparseLt_static.a $static/lib;

    mkdir -p $out;
    cp -r $src/include $out/include;
    cp -r $src/lib $out/lib;
    cp $src/LICENSE $out/LICENSE;
  '';
}
