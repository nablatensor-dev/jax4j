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

Check every POM's project/parent version and the coordinates in `README.md`.
Review the release notes against the documented API limitations and validation
results.

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
On an NVIDIA host with the CUDA driver and NVRTC installed, validate tensor and
all three scalar replay engines explicitly:

```sh
mvn -Pcuda -Djax4j.cuda.tests=true clean verify
```

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


### JAX4J 0.2.0

Version 0.2.0 is published to Central as the parent POM and six library modules
(core, CUDA, ROCm, Vulkan, wgpu and SIMD). Examples remain GitHub-only. Binary
JARs match the v0.2.0 GitHub assets byte for byte. Sources and Javadoc were
generated from the unchanged v0.2.0 tag in an isolated publication tree, with
the same supplementary developer/SCM metadata and module names used for 0.1.0.
The release tag remains at `c388f77e74e0cb900640e51d2641000cf8714fae`.

Central deployment: `53d6d608-6c4e-4a2e-abda-1dac2b8d51c3`. The signed bundle
SHA-256 is `ca1582deac2e74f92399ff25477baea484da7943fda173d79d0c39561dae7847`.
Hardware validation passed 23 CUDA, 31 Vulkan and 36 wgpu checks. ROCm and SIMD
execution remain unverified.

## Reduced packaging for future Central releases

Keep the binary JAR, POM, sources JAR and Javadoc JAR for each library module.
For each of those four payload files, include its GPG `.asc` signature and only
its `.md5` and `.sha1` checksum files. A library module therefore contributes
16 files. A parent, relocation or retirement POM contributes four files: the
POM, its signature and its two checksums. Do not generate checksums for `.asc`
files or signatures for checksum files.

Central requires sources/Javadoc for library JARs and POM metadata for dependency
resolution. Do not remove those files to reduce usage. SHA-256 and SHA-512
checksum sidecars are optional; omit them from the Central upload. Preserve
SHA-256 for the bundle manifest and GitHub `SHA256SUMS` assets: those are separate
integrity records and need not be included in the Central bundle.

Reference: [Central publishing requirements](https://central.sonatype.org/publish/requirements/).

Use a fresh staging directory and inspect the completed ZIP before upload.
Count file entries, excluding directory entries; inspect the artifact/version
list and file suffixes. Confirm there are no `.sha256`, `.sha512`, `.asc.md5` or
`.asc.sha1` sidecars, examples, test reports, native installation archives or
unrequested migration artifacts. Review discrepancies against the counts below
before uploading. Bundling all modules into one ZIP does not reduce the count:
each file inside the upload counts individually. Files inside a library or
Javadoc JAR are not counted separately.

Generate Javadoc with `--no-fonts` as described below. This reduces uploaded
bytes, not file count. Apply packaging changes only to new releases; preserve
published versions, existing migration POMs and release tags.

### JAX4J publication scope and tooling

Publish the `jax4j` parent POM and six library modules: `jax4j-core`,
`jax4j-engine-cuda`, `jax4j-engine-rocm`, `jax4j-engine-vulkan`,
`jax4j-engine-wgpu` and `jax4j-simd`. With reduced checksums, this is
**100 files** per full release: six modules at 16 files plus one parent at four.
`jax4j-examples` remains GitHub-only. Keep the separate GPU engines optional for
consumers; combining them solely to save publishing files is unnecessary.

Generate the signed artifacts with the committed publication profile, using the
new release version already set in the POMs:

```sh
mvn -Pcuda,rocm,simd,central-artifacts verify
```

The profile produces signed artifacts; the bundle assembly step determines
which checksum sidecars are uploaded. Ensure any isolated publication script or
copy of Quant's assembler uses only `md5` and `sha1` for Central payloads. The
shared assembler currently generates all four checksum algorithms and adds
Quant migration artifacts unconditionally; update it according to
[Quant's reduced packaging instructions](https://github.com/nablatensor-dev/quant/blob/main/docs/releasing.md#reduced-packaging-for-future-central-releases)
before using it for future reduced bundles. Its `--jax4j-tree` option includes
Quant as well; it is not a standalone JAX4J publisher. A JAX4J-only release must
contain only the seven coordinates listed above.

Do not attach Quant migration/retirement notices to routine JAX4J releases or
republish unchanged Quant versions. A new combined release containing the four
Quant libraries, both parents and the six JAX4J libraries has **168 files** when
migration notices are omitted. Both projects share the organization's monthly
quota even when uploaded separately.

## Javadoc size: omit bundled fonts

For future releases, generate Javadoc with JDK 25's `--no-fonts` option. Add it
to the `maven-javadoc-plugin` configuration used for publication:

```xml
<configuration>
  <additionalOptions>
    <additionalOption>--no-fonts</additionalOption>
  </additionalOptions>
</configuration>
```

Merge this into the existing configuration and preserve other required options
(such as `--add-modules jdk.incubator.vector` for JAX4J SIMD). Apply it to every
publication profile, including `release` and `central-artifacts` where present,
and to any isolated publication POM generated from a release tag.

The 2026-10-18 usage review matched 90.75 MB across 822 published files. Twenty
Javadoc JARs each bundled the same 25 font files, approximately 4.04 MB per JAR:
80.81 MB of duplicated fonts, about 89% of the total. Omitting those fonts would
reduce that payload to roughly 9.94 MB while keeping API documentation.

Before signing, inspect each generated `*-javadoc.jar` and verify that it has
API HTML pages and no `resource-files/fonts/` entries. Generate and inspect a
fresh Javadoc JAR rather than reusing an existing one. Apply this change to new
release versions only; do not replace published Central artifacts or move tags.
