# Repository instructions

## Start here

Before working on this repository, read [`docs/README.md`](docs/README.md) completely. It is the canonical project entry point and tells you:

- the current checkpoint and whether work is approved;
- the required reading for the current task;
- the project constraints that apply to all work;
- where detailed architecture, development, semantic, experimental, benchmark, and historical information lives.

Do not read every file under `docs/` by default. Follow the required-reading list and routing table in `docs/README.md`, then follow links only as needed for the task.

## Beginning work

1. Read `docs/README.md` completely.
2. Run `git status --short` and inspect recent `git log` entries. Confirm that the documented checkpoint matches the checkout.
3. Read every document or section listed under **Current work → Required reading**.
4. Use the documentation map to identify any additional task-specific material.
5. Inspect only the source and reference files needed for the selected task; do not reload the entire historical codebase or documentation archive by default.
6. If no next slice is approved, do not infer one. Present the smallest relevant options and ask the project owner one focused question at a time.

## During work

- Work in small, explicitly selected vertical experiments. Do not silently absorb likely future work.
- Treat `cubayes/` and `libshadowfax/` as read-only references unless the project owner explicitly decides otherwise.
- Investigate reversible implementation details independently. Ask before changing product scope, compatibility, benchmark meaning, public interfaces, or expensive architecture.
- Test normal behavior, relevant boundaries, and native error/resource-lifetime paths—not only successful paths.
- Keep commits small and independently understandable.
- Record an approved task's goal, non-goals, acceptance criteria, and required reading in `docs/README.md` before or alongside implementation so interrupted work is recoverable.

## Maintaining the documentation

Documentation is part of the implementation. Update it in the same task whenever code, evidence, decisions, risks, or the handoff changes.

Use these ownership rules:

- `docs/README.md`: current checkpoint, current work, cross-cutting constraints, unresolved risks, and navigation.
- `docs/architecture.md`: how the current system works and the current architectural decisions.
- `docs/development.md`: current build, test, environment, device, and benchmark procedures.
- `docs/flagstat.md`: the current flagstat semantic and work-discovery contract.
- `docs/experiments.md`: completed implementation experiments and their evidence.
- `docs/benchmarks.md`: dated performance campaigns, methodology, measurements, and conclusions.
- `docs/history.md`: superseded directions, decision chronology, and completed checkpoint sequence.

Follow these maintenance principles:

1. Keep current instructions in present-tense documents and historical evidence in historical documents.
2. Give each detailed fact one authoritative home. Short summaries may be repeated only when they link to the details.
3. Put conclusions and current consequences before detailed methods and evidence so readers can stop early.
4. Preserve material caveats, negative results, methodology defects, and disproven assumptions. Do not rewrite history to make a result look cleaner.
5. Mark superseded statements and link to their replacement instead of leaving contradictory current guidance.
6. Use file-and-heading links, not line-number references.
7. Do not add raw logs, ordinary refactor narratives, abandoned code sketches, or speculative roadmaps. Summarize what a later reader needs to reason correctly.
8. Keep the document map and current-task required reading accurate whenever files or headings move.
9. Prefer improving an existing document over creating a new file. Add a file only when it provides a clear progressive-disclosure boundary.

## Ending work

1. Review the complete diff and run the relevant verification commands from `docs/development.md`.
2. Update the current checkpoint, current work, detailed technical document, and history or evidence record as applicable.
3. Record measurements with enough context to interpret them, clearly distinguishing smoke observations from controlled benchmarks.
4. Record defects found during review, remaining risks, and whether another slice was explicitly approved.
5. Commit code and documentation together unless blocked. If blocked or intentionally uncommitted, state exactly why and what remains.
6. Leave the next agent a clear required-reading list and stop at the approved boundary.
