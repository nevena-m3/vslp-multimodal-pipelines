# Acoustic release data-hygiene audit

The authorized cleanup on 2026-09-24 removed 77 potentially participant-derived current files and one additional historical alias from all advertised branch and tag histories controlled by this Git remote. The path-level manifest, file hashes, and complete pre-rewrite Git bundles are held in a clearly named private backup **outside this repository**. This public report deliberately omits participant-like filenames.

| Category | Files removed from Git history |
|---|---:|
| Participant-like source WAVs | 5 |
| Demographics/participant metadata CSV | 1 |
| Participant-derived generated run artifacts | 47 |
| Generated analysis plots | 21 |
| Notebooks with stored full-dataset or acoustic outputs | 2 |
| Historical document with participant-like identifier | 1 |
| Older path for that same historical document | 1 historical alias |
| **Purge path identities** | **78** |

No automated test depended on these paths. All 77 local files were copied to the private backup and SHA-256 verified before the rewrite. The additional older document path is preserved in the private pre-rewrite Git bundle. The user's local research files in the checkout were **not deleted**; after the rewritten tip was installed, they became ignored, untracked local files. The generated path inventory was moved out of the checkout as well.

`git-filter-repo` 2.47.0 removed the paths from separate bare mirrors of the remote. A second path/content scan found the document's older name, which was removed in a second guarded rewrite. Cleanup commits on the sanitized `september-review` tip strengthened `.gitignore` for generated plots, notebooks with stored outputs, metadata, and participant-named CSVs. Atomic, explicit `--force-with-lease` pushes updated the affected remote branches and tags. A fresh post-push mirror clone had zero purge-path or participant-like identifier matches in its reachable history. The active local branches, remote-tracking refs, tags and index were updated to sanitized commits; old local reflogs were expired and unreachable objects pruned.

The complete pre-rewrite bundles and research-file copies are intentionally retained only in the private backup for recovery. GitHub or other services may retain caches, hidden unreachable objects, clones, or forks outside the refs verified here. This report does **not** claim those external copies were erased. Collaborators with older clones must stop pushing old history and reclone or reset to the rewritten refs.
