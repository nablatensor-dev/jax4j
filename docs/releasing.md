# Releasing JAX4J

Releases use tags named `vMAJOR.MINOR.PATCH`, such as `v0.1.0`, and a GitHub
release titled with the version, such as `0.1.0`. The tag points to a commit
containing matching Maven versions and the final documentation.

## Prerequisites

- Java 25 and Maven, with `JAVA_HOME` pointing to the JDK.
- GitHub CLI (`gh`) authenticated with permission to push tags and publish
  releases to `nablatensor-dev/jax4j`.
- Git HTTPS credentials configured, for example with
  `gh auth setup-git --hostname github.com` after logging in.
- A clean checkout of `main`, synchronized with `origin/main`.
- For hardware validation, the native libraries and tools described in
  [wgpu support](wgpu.md). Record unavailable devices as unverified.

## Prepare the version

Choose the release version and check that its tag and release do not exist:

```sh
RELEASE_VERSION=0.1.0
REPO=nablatensor-dev/jax4j

git switch main
git pull --ff-only
git status --short
git ls-remote --tags origin "refs/tags/v$RELEASE_VERSION"
gh release list --repo "$REPO"
```

Stop if the checkout has unrelated changes or the version already exists. Never
move a published release tag; use a new version for corrections.

Update the parent and every child POM, including optional modules:

```sh
mvn -Pcuda,rocm,examples,simd versions:set \
  -DnewVersion="$RELEASE_VERSION" \
  -DprocessAllModules=true -DgenerateBackupPoms=false
```

Check every POM's project/parent version, the coordinates in `README.md`, and the
current-version statement in `docs/implementation-status.md`. Preserve historical
upstream versions in source-provenance records. Review the release notes against
[implementation status](implementation-status.md) and the validation results.

## Build and validate

Build every module and run the ordinary test suite:

```sh
mvn -Pcuda,rocm,examples,simd clean verify
```

On a configured Vulkan/wgpu host, run the hardware suites as well:

```sh
mvn -Pcuda,rocm,examples,simd \
  -Djax4j.vulkan.tests=true -Djax4j.wgpu.tests=true verify
```

Explicit hardware checks fail when their backend is unavailable. A successful
compile of the CUDA/ROCm modules does not establish native runtime correctness.
Review failures and skipped tests before proceeding. Check the release diff:

```sh
git diff --check
git diff --stat
git diff
```

## Prepare assets and notes

Publish the seven module JARs plus `SHA256SUMS`. GitHub automatically provides
source archives for the tag. Native GPU libraries are not bundled, and this
workflow does not publish to Maven Central.

Use a fresh directory outside the checkout so it contains only this release's
assets:

```sh
RELEASE_DIR=$(mktemp -d "/tmp/jax4j-release-$RELEASE_VERSION.XXXXXX")
export RELEASE_VERSION RELEASE_DIR
python3 - <<'PY'
from pathlib import Path
import hashlib
import os
import shutil

version = os.environ['RELEASE_VERSION']
destination = Path(os.environ['RELEASE_DIR'])
modules = ['core', 'engine-cuda', 'engine-rocm', 'engine-vulkan',
           'engine-wgpu', 'examples', 'simd']
lines = []
for module in modules:
    name = f'jax4j-{module}-{version}.jar'
    source = Path(f'jax4j-{module}/target') / name
    target = destination / name
    shutil.copy2(source, target)
    lines.append(f'{hashlib.sha256(target.read_bytes()).hexdigest()}  {name}')
(destination / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
PY
```

Write release notes to `$RELEASE_DIR/notes.md`. Include the main features,
Java/native requirements, validation totals and hardware tested, known
limitations, and how to build from the tag. The parent `jax4j` artifact is a POM;
applications normally use `jax4j-core` and the desired engine JARs. SIMD requires
`--add-modules jdk.incubator.vector`, and native access requires
`--enable-native-access=ALL-UNNAMED`.

## Commit, tag and publish

Stage the intended version/documentation changes and inspect the staged diff:

```sh
git add pom.xml jax4j-*/pom.xml README.md docs/
git diff --cached --check
git diff --cached
git commit -m "Release $RELEASE_VERSION"
git push origin main
git tag -a "v$RELEASE_VERSION" -m "Release $RELEASE_VERSION"
git push origin "v$RELEASE_VERSION"

gh release create "v$RELEASE_VERSION" \
  "$RELEASE_DIR"/*.jar "$RELEASE_DIR/SHA256SUMS" \
  --repo "$REPO" --verify-tag \
  --title "$RELEASE_VERSION" --notes-file "$RELEASE_DIR/notes.md"
```

Verify the published tag, assets and their checksums:

```sh
gh release view "v$RELEASE_VERSION" --repo "$REPO"
VERIFY_DIR=$(mktemp -d "/tmp/jax4j-verify-$RELEASE_VERSION.XXXXXX")
gh release download "v$RELEASE_VERSION" --repo "$REPO" --dir "$VERIFY_DIR"
(cd "$VERIFY_DIR" && sha256sum -c SHA256SUMS)
git ls-remote --tags origin "refs/tags/v$RELEASE_VERSION*"
```

If publication fails after pushing the tag, inspect `gh release view` and resume
from the failed step. Keep the existing tag on the validated release commit.

Consumers can build and install matching POMs/JARs locally from the source tag:

```sh
git clone --branch v0.1.0 --depth 1 https://github.com/nablatensor-dev/jax4j.git
cd jax4j
mvn -Pcuda,rocm,examples,simd -DskipTests install
```

Replace `v0.1.0` with the desired release tag. After publication, a separate
commit may advance all POMs to the next development `-SNAPSHOT` version; never
change the published tag to include that commit.

## Maven Central publication

Version 0.1.0 is also published to Maven Central under `com.nablatensor`.
The binary JARs match the existing GitHub release assets; source and Javadoc
JARs were generated from the unchanged v0.1.0 tag. Publishing POMs supplement
that tag's build metadata with required developer/SCM information and explicit
names for the core, Vulkan and wgpu modules. The GitHub tag was not moved.

The Central release was validated on real Vulkan and wgpu hardware (31 and 36
checks respectively); optional CUDA/ROCm/SIMD artifacts were compiled but their
backends were not tested. Examples are not published as library artifacts.

The publication tooling and NablaTensor migration coordinates are maintained in
[quant's release guide](https://github.com/nablatensor-dev/quant/blob/main/docs/releasing.md).
Central credentials remain in the local Maven `central` server entry. Never
commit credentials or overwrite an already-published Central version.
