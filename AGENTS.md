# xshi19.github.io

This repository holds committed static output for GitHub Pages. Edit the upstream source repositories. The private `xshi19/site` build produces the homepage HTML (including root pages), `_astro/`, `pagefind/`, `og/`, `images/`, feeds, sitemaps, `llms*.txt`, and `dist-manifest.json`. Keep the legacy mirror pages at their old Pelican URLs.

Homepage publishes may add or modify only files listed by the site build. Never create, modify, or delete anything under `math/`, `incerto-wiki/`, or `normix/` during a homepage update. Never run root-wide deletes or `rsync --delete`.

## Context map

| Area | Where to look |
| --- | --- |
| Homepage source and publish rules | `AGENTS.md` in `xshi19/site` |
| Homepage-owned files | `dist-manifest.json` in this repository |
| `math/` | `xshi19/xshi-math` builds and publishes the math site. |
| `incerto-wiki/` | Keep the legacy mirror and compatibility pages for old links. |
| `normix/` | `xshi19/normix` publishes its project documentation here. |

## Checks

- For a homepage PR, run `git diff --name-only origin/main...HEAD -- math incerto-wiki normix` from this root. It must print nothing. A recursive comparison (`diff -r`) of each of those three directories with its counterpart in a clean checkout of main must also print nothing.
- This repository has no npm scripts. Serve this staging tree locally on port 4321, then run `npm run check:urls -- http://127.0.0.1:4321` from the root of `xshi19/site`. After publish, run `npm run check:urls -- https://xshi19.github.io --live-only` from the same source repository.
