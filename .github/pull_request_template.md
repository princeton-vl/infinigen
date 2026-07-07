## Author Requirements

Write a short answer in each description, then change the box to be `[x]` to complete it
(you can also click the boxes after creating the PR)

### What does the PR do?

What GitHub issue is this working on? If the issue is done, say "Fixes #X".
- [ ] <!-- Type # then start searching for the title-->

Description of what was changed in this specific PR
- [ ] <!-- description -->

### Outside sources

List ALL sources consulted to implement the PR that were not written by you, e.g. youtube videos, stackoverflow, any websites at all.

- [ ] List all outside sources sources:
    - <!-- Say "None", Answer and mark with an `x` -->

### Tests

*We will not merge if your branch does not show green checkmark*
It will fail if you have bad linting or formatting, but these should already be checked locally by `pre-commit`

Please run `uv run pytest tests/infinigen2` if you changed any core tools (outside of a specific asset)

### Results

- [ ] Paste your render or unit test command(s):
<!-- commands, or say N/A -->

- [ ] Paste images and commands to demonstrate the results of the PR:
<!-- Paste them and mark with an `x`, or say N/A -->

## Reviewer Requirements

1. Check for unsafe or unmaintainable code
2. Check for accidental new files or unnecessary edits (e.g. formatting, unrelated changes in PR)
3. If there are any "Outside Sources" you *must* request review from `araistrick`
