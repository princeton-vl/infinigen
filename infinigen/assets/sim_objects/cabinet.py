from numpy.random import uniform

from infinigen.assets.objects.shelves.cabinet import CabinetBaseFactory


class CabinetFactory(CabinetBaseFactory):
    extra_exclude = {("link_1", "link_2")}

    def sample_params(self):
        params = dict()
        params["Dimensions"] = (
            uniform(0.25, 0.35),
            uniform(0.3, 0.7),
            uniform(0.9, 1.8),
        )

        params["bottom_board_height"] = 0.083
        params["shelf_depth"] = params["Dimensions"][0] - 0.01
        num_h = int((params["Dimensions"][2] - 0.083) / 0.3)
        params["shelf_cell_height"] = [
            (params["Dimensions"][2] - 0.083) / num_h for _ in range(num_h)
        ]
        params["shelf_cell_width"] = [params["Dimensions"][1]]
        self.shelf_params = self.shelf_fac.sample_params()
