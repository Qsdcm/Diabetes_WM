# Git operation policy

- Do not automatically stage, commit, or push changes.
- Only run `git add`, `git commit`, or `git push` when the user explicitly asks for that Git operation in the current request.
- After implementing changes, report modified files and verification results, then leave the changes uncommitted unless instructed otherwise.
- Read-only Git commands such as `git status`, `git diff`, and `git log` are allowed for verification.
