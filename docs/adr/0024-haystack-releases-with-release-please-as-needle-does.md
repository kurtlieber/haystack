# haystack releases with release-please, as needle does

Until 0.9.0 a haystack release was made by hand. A PR bumped `version` in `pyproject.toml` and `__version__` in
`src/haystack_music/__init__.py` and wrote the version's CHANGELOG section. After the merge someone pushed the
`v<version>` tag, and the tag started `image`, which published the image and started lieber.org's deploy. Every
step could be forgotten. 0.9.0 needed a follow-up commit for a fix its CHANGELOG left out, and
kurtlieber/haystack#41 was merged with no CHANGELOG line at all. needle has released with release-please since its
ADR 0007. haystack never chose otherwise: it left release-please behind with needle's workflows when it moved out
of spotify-discovery (kurtlieber/dotfiles#408).

So haystack uses release-please too:

- `haystack-release` runs release-please on every push to main. It keeps one release PR open with the next version
  (from `feat:` and `fix:`; before 1.0 a breaking change bumps the minor) in `pyproject.toml` and `__init__.py`
  (marked `x-release-please-version`), and the version's section in `CHANGELOG.md`. The PR asks the owner for review.
- Merging that PR is the release. release-please tags `vX.Y.Z` and creates the GitHub Release. Then
  `haystack-release` calls `image` with the tag, which builds, smoke-tests and publishes
  `ghcr.io/kurtlieber/haystack:X.Y.Z`, `:X.Y` and `:latest`, and starts lieber.org's deploy. A tag created with
  `GITHUB_TOKEN` starts no workflow, so `image` can't rely on its own tag trigger. `image` still runs for a tag
  pushed by hand, and by hand (`workflow_dispatch`) with an existing tag to rebuild one.
- The docs check stays: before the release PR is merged, the docs are read against everything it collects, and any
  fix goes in first (AGENTS.md).

**Considered Options**: keep the release by hand (the steps above get forgotten); a release on every merge to main
(no gate before an image reaches `:latest`); release-please with a GitHub App token, so the tag triggers `image`
itself (a long-lived App secret in the repository for one workflow call).

**Consequences**: commits follow conventional commits, or they leave no CHANGELOG line and no version bump. The
CHANGELOG lines are the commit subjects. A release PR can rewrite them, but release-please writes them again on
the next push to main, so do it last. The release PR is opened with `GITHUB_TOKEN`, so `server-ci` and `image`
don't run on it; the release run's image build and smoke test are the check. The repository needs "Allow GitHub
Actions to create and approve pull requests" on. Merging the release PR is the go for the tag, the image and the
deploy: no separate word for the tag push. The first release under release-please starts from v0.9.0's commit
(`bootstrap-sha`).
