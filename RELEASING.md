# Releasing

Smoothify is published to PyPI by tag-push. The version comes from the git tag
itself (via `setuptools-scm`) — there is **no version file to bump**.

## Versioning

Follow [SemVer](https://semver.org/):

- **Major** (`v1.0.0` → `v2.0.0`): incompatible API change — removed args,
  changed signatures, behavioural changes that break existing callers.
- **Minor** (`v1.0.0` → `v1.1.0`): backwards-compatible feature added.
- **Patch** (`v1.0.0` → `v1.0.1`): backwards-compatible bug fix.

Any commit that lands on `main` between releases shows up at install time as
a development version like `0.2.4.dev3+g1234abc`.

## Release checklist

1. **Sync `main`** locally:
   ```bash
   git checkout main
   git pull
   ```

2. **Update [`CHANGELOG.md`](CHANGELOG.md)**: move the `## [Unreleased]` entries to
   `## [<version>] - <YYYY-MM-DD>` and add a fresh empty `[Unreleased]` block at the top.
   Commit on `main`:
   ```bash
   git commit -am "Prepare <version> changelog"
   git push
   ```

3. **Tag and push**:
   ```bash
   git tag v<version>          # e.g. v0.3.0
   git push origin v<version>
   ```
   The tag must match `v<major>.<minor>.<patch>` (pre-release suffixes like
   `v0.3.0rc1` also match the publish workflow's filter).

4. **Approve the deploy**: the [`Publish to PyPI`](.github/workflows/publish.yml)
   workflow runs `uv build` and uploads to PyPI via OIDC trusted publishing.
   The `pypi` GitHub environment is gated by a required reviewer — go to
   **Actions → Publish to PyPI → Review deployments → Approve** to release.

5. **Verify**:
   - PyPI page: <https://pypi.org/project/smoothify/>
   - GitHub Releases: <https://github.com/DPIRD-DMA/Smoothify/releases>
     (auto-generated release notes + wheel/sdist attached).
