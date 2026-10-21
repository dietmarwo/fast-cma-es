# Publishing fcmaes

The repository produces one PyPI project, `fcmaes`: wheels for CPython 3.10
to 3.13 on Linux x86_64 (manylinux_2_28), Windows AMD64 and macOS (Intel and
Apple Silicon), built by cibuildwheel, and the source distribution.

The version is `[project].version` in `pyproject.toml`
(`scripts/package_version.py` prints it). Publication is irreversible: never
reuse a version, and note that the workflow publishes with `skip-existing`,
so an already published version is silently skipped.

## Workflow

`.github/workflows/wheels.yml`, on every pull request and push to `master`:

- `version`: prints the version; on a tag, requires the tag to equal `v` plus
  the version.
- `build_wheels`: cibuildwheel builds the wheels and runs `pytest ./tests` on
  each installed wheel.
- `build_sdist`: builds the source distribution, installs it (the source
  build that `pip` runs when no wheel matches) and runs the tests outside the
  checkout.
- `publish_testpypi`: only on a manual run with `publish_target=testpypi`.
- `publish_pypi`: only on a `v*` tag; generates GitHub artifact attestations
  and publishes through PyPI Trusted Publishing.

## One-time setup

Already done for the production project; kept for recovery or a fork:

1. GitHub environments `pypi` and `testpypi`.
2. Trusted Publishers on PyPI and TestPyPI for owner `dietmarwo`, repository
   `fast-cma-es`, workflow `wheels.yml`, environment `pypi` (respectively
   `testpypi`).

Renaming the workflow file or the environments breaks the Trusted
Publishers.

## Release

1. Update `CHANGELOG.md`: move the `[Unreleased]` entries under the new
   version with today's date, and the "Changes after version ..." list in
   `README.adoc`.
2. Set `[project].version` in `pyproject.toml`.
3. Merge to `master` with green CI (wheels, native tests, documentation).
4. Optionally, run the Wheels workflow manually with
   `publish_target=testpypi` and install from TestPyPI:

   ```bash
   python -m pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ "fcmaes==$(python scripts/package_version.py)"
   ```

5. Tag and push:

   ```bash
   version="$(python scripts/package_version.py)"
   git tag -a "v$version" -m "Release fcmaes $version"
   git push origin "v$version"
   ```

## Post-release verification

```bash
python -m venv /tmp/fcmaes-check
/tmp/fcmaes-check/bin/python -m pip install fcmaes
/tmp/fcmaes-check/bin/python -c "from importlib.metadata import version; import fcmaes.pareto; print(version('fcmaes'))"
```

Check the PyPI page (wheels, sdist, attestations), then create the GitHub
release with the changelog entry.

## Discovery after a release

Follow the [ecosystem publication checklist](https://github.com/dietmarwo/optimization-jobs/blob/main/docs/discovery/README.md).
Package descriptions and README links reach registries only with a new
release. Check the rendered package page after publishing and update the
shared catalog when installation availability changes. Repository topics
and About text are GitHub settings, not fields a commit changes.
