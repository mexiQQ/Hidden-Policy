# Project operating instructions

These instructions apply to the entire repository and to all future work on
the local checkout and the A6000 checkout.

## Local ↔ A6000 file synchronization

GitHub is the only permitted transport for files between the local machine and
the A6000 server.

- Local to A6000: review and commit the intended files locally, push the commit
  to GitHub, then run `git pull --ff-only` in the A6000 checkout.
- A6000 to local: review and commit the intended files on A6000, push the commit
  to GitHub, then run `git pull --ff-only` in the local checkout.
- Never transfer files with `scp`, `rsync`, SFTP, SSH stdin/stdout redirection,
  tar or base64 streams over SSH, mounted remote filesystems, or equivalent
  direct-copy mechanisms.
- SSH remains allowed for interactive access, read-only inspection, command
  execution, process control, monitoring, and running Git commands on A6000.
- Do not add ignored, raw, sensitive, credential-bearing, or oversized files to
  Git merely to move them. In particular, `code/data/`, `code/runtime/`, and
  `code/results/` retain their existing privacy boundary. Publish only reviewed,
  sanitized artifacts such as the tracked aggregate reports.
- If a required file cannot safely or appropriately be committed to GitHub,
  stop and ask the user how to proceed. Do not bypass this policy.
- Before synchronization, verify the sending checkout, staged diff, branch, and
  commit. After synchronization, verify that both checkouts and GitHub resolve
  to the intended commit.
