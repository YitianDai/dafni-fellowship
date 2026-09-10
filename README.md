# DAFNI Fellowship website

Single-page site for the Fellowship, served with GitHub Pages. No build step.

## Publish

1. Create a public repository on GitHub (e.g. `dafni-fellowship`) and push these files to `main`.
2. Repository Settings > Pages > Build and deployment: source "Deploy from a branch", branch `main`, folder `/ (root)`. Save.
3. The site appears at `https://<username>.github.io/dafni-fellowship/` within a minute or two. Every push to `main` redeploys.

## Before publishing

- Replace the `[bracketed]` placeholders in `index.html` (dates, venue, participant numbers, ORCID and profile URLs).
- Add logos to `assets/images/`: `dafni-logo.png`, `ukri-logo.png`, `manchester-logo.png`, `supergen-logo.png` (roughly 42 px tall when displayed; PNG with transparent background works best).
- Add the poster as `assets/resources/poster.pdf`, or remove that card.
- Check the email address in the nav, workshop and contact sections.

## Editing

All content, styling and the small menu script live in `index.html`. Update the timeline under `#updates` as the Fellowship progresses and change the "Last updated" date.
