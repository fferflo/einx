Installation
############

einx can be installed as follows:

..  code::

    pip install einx

If you want to install the latest version from GitHub, you can do so using:

..  code::

    pip install git+https://github.com/fferflo/einx.git

einx automatically detects backends like PyTorch when it is run, but does not include hard dependencies for the corresponding packages.
If you plan to use einx with a specific backend, you can also install it as follows:

..  code::

    pip install einx[torch]

This will add a dependency for PyTorch and enforce the version requirements of einx (i.e. PyTorch >= 2.0.0).
This is currently only supported for PyTorch (``einx[torch]``) and Keras (``einx[keras]``).