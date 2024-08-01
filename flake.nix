{
  description = "Pytket Cutensornet Extension";
  nixConfig.extra-substituters = "https://tket.cachix.org https://cache.nixos.org https://cuda-maintainers.cachix.org https://nix-community.cachix.org";
  nixConfig.trusted-public-keys = ''
    tket.cachix.org-1:ACdm5Zg19qPL0PpvUwTPPiIx8SEUy+D/uqa9vKJFwh0=
    cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY=
    cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E=
    nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs=
  '';
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.tket.url = "github:CQCL/tket";
  inputs.nixpkgs.follows = "tket/nixpkgs";
  outputs = { self, nixpkgs, flake-utils, tket }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
          overlays = [
            (self: super: {
              inherit (tket.packages."${system}") tket pytket;
            })
            (self: super: {
              cuda-bundle = pkgs.callPackage ./nix-support/cuda-bundle.nix {};
            })
            (self: super: {
              cupy' = pkgs.python3Packages.callPackage ./nix-support/cupy.nix {};
            })
            (self: super: {
              pycuquantum = pkgs.python3Packages.callPackage ./nix-support/pycuquantum.nix {};
            })
            (import ./nix-support/pytket-cutensornet.nix)
          ];
        };
      in {
        packages = {
          default = pkgs.pytket-cutensornet;
          cupy = pkgs.cupy';
          pytket-cutensornet = pkgs.pytket-cutensornet;
          tests = pkgs.run-pytket-cutensornet-tests;
        };
        devShells = {
          default = pkgs.mkShell { buildInputs = [ pkgs.pytket-cutensornet ]; };
        };
        checks = {
          # no GPU support in checks at the time of writing
        };
      });
}
