from pathlib import Path

import infinigen.assets.sim_objects.blueprints as blueprints


def blueprint_path_completion(blueprint_path, root=None):
    """
    Takes in a local xml path and returns a full path.
        if @xml_path is absolute, do nothing
        if @xml_path is not absolute, load xml that is shipped by the package

    Args:
        xml_path (str): local xml path
        root (str): root folder for xml path. If not specified defaults to
            infinigen.assets.sim_blueprints
    Returns:
        str: Full (absolute) xml path
    """
    if blueprint_path.startswith("/"):
        full_path = blueprint_path
    else:
        if root is None:
            root = blueprints
        full_path = Path(root.__path__[0]) / blueprint_path
    return Path(full_path) if isinstance(full_path, str) else full_path
