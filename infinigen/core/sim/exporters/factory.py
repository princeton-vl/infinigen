def sim_exporter_factory(exporter="mjcf"):
    """
    Creates an instance of a sim exporter
    """

    if exporter in ["usd", "usda", "usdc"]:
        from infinigen.core.sim.exporters.usd_exporter import export

        return export
    elif exporter == "mjcf":
        from infinigen.core.sim.exporters.mjcf_exporter import export

        return export
    elif exporter == "urdf":
        from infinigen.core.sim.exporters.urdf_exporter import export

        return export
    else:
        print(
            f"Exporter type {exporter} is not supported. Supported types \
                include [usd, mjcf, urdf]."
        )
