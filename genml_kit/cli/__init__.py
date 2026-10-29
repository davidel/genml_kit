"""Console-script entry points for the ``genml_kit`` toolkit.

Each module here exposes a ``main()`` and the argparse wiring that is
specific to it; the heavy lifting (training loop, inference, LLM
generation/eval) stays in the ``genml_kit.*`` library modules.

    * :mod:`genml_kit.cli.train`  -- ``genml-kit-train``
    * :mod:`genml_kit.cli.infer`  -- ``genml-kit-infer``
    * :mod:`genml_kit.cli.llm`    -- ``genml-kit-llm``
"""
