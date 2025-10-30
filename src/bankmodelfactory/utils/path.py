import sys
from pathlib import Path

def get_project_root(add_to_sys_path: bool = False):
    """
    Add the project root (where pyproject.toml is located) to sys.path if requested.

    Parameters
    ----------
    add_to_sys_path : bool, optional
        If True, the project root will be added to sys.path. Default is False.

    Returns
    -------
    Path
        The resolved project root path.
    """
    root = Path().resolve().parents[0]

    if add_to_sys_path:
        sys.path.append(str(root))

    return root