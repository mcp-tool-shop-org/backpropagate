# Vendored Fonts Notice

This directory vendors the Geist font family for offline self-hosting (Microsoft
Store app; Content-Security-Policy `font-src 'self'`). No CDN or Google Fonts
fetching is used. All files are variable fonts (single woff2 per family covering
the full weight range 100-900).

## Upstream

- Project: Geist Font (by Vercel, in collaboration with basement.studio)
- Upstream repository: https://github.com/vercel/geist-font
- Homepage: https://vercel.com/font
- License: SIL Open Font License, Version 1.1 (OFL-1.1) - see `OFL.txt` in this
  directory (verbatim copy of the upstream LICENSE.txt).

## Vendored release/version

- Version: v1.7.2 (newest stable release at time of vendoring, published
  2026-06-01; not a prerelease)
- npm package `geist@1.7.2`, SIL OPEN FONT LICENSE:
  - Metadata: https://registry.npmjs.org/geist/latest
  - Tarball:  https://registry.npmjs.org/geist/-/geist-1.7.2.tgz
  - Tarball sha1 (npm "shasum"):
    96f6e5d2b3305fd27eacbd5ae4dcfbc5a15e6939 (verified)
  - Tarball integrity (npm, sha512):
    sha512-Gu5lDFa3pLRyoBlBPf0QIFHVdWAnpco7fS1bJm41jyLPFoguBgiubseUN2oLXMgqZ7uxAxDoXcHMhCY/fOTTgg== (verified)
- Equivalent GitHub release (same tag, same contents):
  - Release page: https://github.com/vercel/geist-font/releases/tag/v1.7.2
  - Zip asset:    https://github.com/vercel/geist-font/releases/download/v1.7.2/geist-font-v1.7.2.zip
  - Zip sha256 (GitHub "digest"):
    7fc800d2ac6b92844895196e5041aca55d814c15db70c44f79b3b83ab82b04e2

Files were taken from the npm tarball (package/dist/fonts/... and
package/LICENSE.txt) and copied byte-for-byte (verified with byte comparison)
from these paths inside the tarball:

- package/dist/fonts/geist-sans/Geist-Variable.woff2     -> geist-var.woff2
- package/dist/fonts/geist-mono/GeistMono-Variable.woff2 -> geist-mono.woff2
- package/LICENSE.txt                                    -> OFL.txt

## Vendored files (this directory)

| file             | upstream path                                            | bytes  | sha256 (hex) |
|------------------|----------------------------------------------------------|--------|--------------|
| geist-var.woff2  | package/dist/fonts/geist-sans/Geist-Variable.woff2       | 69,652 | a369fcf5628ea2aa4e1b9e2ec6a5b3624e365bda588e1f0f2f12b564f728fbb8 |
| geist-mono.woff2 | package/dist/fonts/geist-mono/GeistMono-Variable.woff2   | 71,368 | fba8f577f38a2bbcbe818efa6348dd58f36303a10b8737c42fefad275be563ab |
| OFL.txt          | package/LICENSE.txt                                      |  4,368 | 930853ee1daa68554d9e35c8a9175affb74f699fad9a5da6ee5ebe76379d9137 |

Total payload: 145,388 bytes (~142 KB).

Integrity checks performed at vendoring time:

- woff2 magic bytes `wOF2` present at offset 0 of both font files.
- SHA-256 of each file was computed from the downloaded tarball and re-verified
  after copying to this directory.
- npm tarball checksums (sha1 + sha512 integrity) matched the registry.

## Note on variable fonts

Variable fonts are used: each woff2 contains a `wght` axis covering the full
weight range (per upstream, 100-900), so a single file per family serves every
weight via CSS `font-weight`. Declare with `font-weight: 100 900;` in the
@font-face rule (see the companion report at .claude/bp-scratch/geist-report.md
for ready-to-paste CSS).

## What these fonts are used for

UI typography (Geist) and code/console typography (Geist Mono) in the
backpropagate web UI, served from this package as static assets under the
`/fonts/` URL prefix.
