# Anonymous Repository Cleaning Notes

The cleaned repository keeps source code, launch scripts, and documentation needed to reproduce MEDitor-style routing experiments.

Excluded from the anonymous repository:

- submitted or draft paper PDFs
- cached Python bytecode
- Slurm stdout/stderr logs
- generated run directories and rollout archives
- local model, corpus, checkpoint, and dataset paths
- Slurm and shell service-launch scripts
- early legacy scripts with stale package names and absolute cluster paths

Local paths should be supplied through environment variables, config scripts, or Slurm submission environment variables.
