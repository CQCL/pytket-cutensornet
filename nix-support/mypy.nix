{
  python3,
  fetchFromGitHub,
  mypy
}:
mypy.overrideAttrs (oldAttrs: rec {
  version = "1.13.0";
  name = "python${python3.pythonVersion}-mypy-${version}";
  src = fetchFromGitHub {
    owner = "python";
    repo = "mypy";
    rev = "refs/tags/v${version}";
    hash = sha256:P2Ozmj7/7QBmjlveHLsNdYgUAerg0qOoa8pO0iQc5os=;
  };
})
