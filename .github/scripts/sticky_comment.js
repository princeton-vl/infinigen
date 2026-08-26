// Upsert a PR status comment, keyed by the run-scoped marker in the body.
const fs = require('fs');
const path = require('path');

const MARKER_RE = /<!-- integration-render-status[^>]*-->/;
const PENDING_MARKER = '<!-- integration-render-pending -->';

module.exports = async ({github, context}) => {
  const pr = Number(process.env.PR_NUMBER);
  if (!pr) return;

  const file = path.join(process.env.RUNNER_TEMP, process.env.BODY_FILE);
  if (!fs.existsSync(file)) return;
  const body = fs.readFileSync(file, 'utf8');
  const found = body.match(MARKER_RE);
  if (!found) throw new Error(`body ${file} is missing integration-render-status marker`);
  const marker = found[0];

  const comments = await github.paginate(github.rest.issues.listComments, {
    ...context.repo, issue_number: pr, per_page: 100,
  });
  const prev = comments.find(
    c => c.user.type === 'Bot' && (c.body || '').includes(marker));

  if (prev) {
    await github.rest.issues.updateComment({
      ...context.repo, comment_id: prev.id, body,
    });
  } else {
    await github.rest.issues.createComment({
      ...context.repo, issue_number: pr, body,
    });
  }

  // A run cancelled mid-render never posts results, stranding its pending comment.
  const stale = comments.filter(c => {
    const b = c.body || '';
    return c.user.type === 'Bot' && b.includes(PENDING_MARKER) && !b.includes(marker);
  });
  for (const c of stale) {
    await github.rest.issues.deleteComment({...context.repo, comment_id: c.id});
  }
};
