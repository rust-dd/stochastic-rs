# Notebooks

- `colab_cuda_check.ipynb` — the source notebook, no outputs: open it in
  [Colab](https://colab.research.google.com/github/rust-dd/stochastic-rs/blob/main/notebooks/colab_cuda_check.ipynb)
  on a GPU runtime and run all cells to validate the `cuda` back-end.
- `colab_cpu_vs_cuda_paths.ipynb` — the path gallery: seventy-odd processes
  across ten families sampled twice, once on the host and once on CUDA, with
  the two sets of paths side by side, their terminal laws overlaid, a table of
  how far apart the laws sit in standard errors, and wall time per batch. One
  figure per family, and every parameter set is one the crate's own device-law
  suite runs. Open it in
  [Colab](https://colab.research.google.com/github/rust-dd/stochastic-rs/blob/main/notebooks/colab_cpu_vs_cuda_paths.ipynb)
  on a GPU runtime. The two sides draw different streams by construction, so
  what the plots compare is the law, never the path.
- `runs/` — executed copies with their outputs, one per run and machine
  (GitHub's viewer may not render Colab outputs; nbviewer does). Add a new run
  as `runs/colab_cuda_check_<gpu>_<date>.ipynb` rather than overwriting the source.
