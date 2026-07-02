# Writing Workflow

Use this site as a small public notebook for career logs, research ideas, thinking records, and thought experiments.

## Public posts

Create public posts in `_posts/`.

File names must use this format:

```text
YYYY-MM-DD-short-title.md
```

Example front matter:

```yaml
---
layout: single
title: "A Research Idea"
date: 2026-07-02
categories:
  - research
tags:
  - idea
  - agent
---
```

Posts in `_posts/` are shown at `/posts/`, `/year-archive/`, `/tags/`, and `/categories/`.

If a post has `published: false`, Jekyll will not render it. Remove that line, or set it to `true`, when you want the post to appear publicly.

## Drafts

Put unfinished drafts in `_drafts/`. Drafts do not appear on the deployed site.

To preview drafts locally:

```bash
bundle exec jekyll serve --drafts
```

## Private local notes

Put notes that should stay only on this computer in `_private_posts/`.

That folder is ignored by git, so it will not be uploaded through normal commits and pushes. Avoid putting private writing in `_posts/` or `_drafts/` if this GitHub repository is public, because committed source files can still be read on GitHub even when Jekyll does not publish them.
