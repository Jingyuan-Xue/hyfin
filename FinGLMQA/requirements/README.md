# requirements

- `phase10.in` — direct dependency declarations for the Phase 10 service.
- `phase10.lock` — the fully pinned lock for the isolated `.venv-phase10`.

Dependency upgrades must be done in a fresh environment, then re-validated by
rerunning the Phase 8/9 regressions, the Phase 10 gates, the HTTP determinism
comparison and the immutable manifest verification.

Do not upgrade packages in place inside `.venv-phase10` and carry on using the
old lock file.

To create the environment from scratch, see
[docs/DEPLOY.md](../../docs/DEPLOY.md#3-create-the-virtualenvs).
