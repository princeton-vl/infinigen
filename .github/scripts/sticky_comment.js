// Upsert this workflow's status comment on a PR, keyed by the same marker as
// STATUS_MARKER in compose_pr_comment.py. Keep the two in sync.
const fs = require('fs');
const path = require('path');

const MARKER = '<!-- integration-render-status -->';

module.exports = async ({github, context}) => {
  const pr = Number(process.env.PR_NUMBER);
  if (!pr) return;

  const file = path.join(process.env.RUNNER_TEMP, process.env.BODY_FILE);
  if (!fs.existsSync(file)) return;
  const body = fs.readFileSync(file, 'utf8');
  if (!body.includes(MARKER)) throw new Error(`body ${file} is missing ${MARKER}`);

  const comments = await github.paginate(github.rest.issues.listComments, {
    ...context.repo, issue_number: pr, per_page: 100,
  });
  const prev = comments.find(
    c => c.user.type === 'Bot' && (c.body || '').includes(MARKER));

  if (prev) {
    await github.rest.issues.updateComment({
      ...context.repo, comment_id: prev.id, body,
    });
  } else {
    await github.rest.issues.createComment({
      ...context.repo, issue_number: pr, body,
    });
  }
};
