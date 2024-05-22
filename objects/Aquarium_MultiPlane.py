#########################################################################################################
# This Class is not yet supported in this framework. It is a placeholder for future development. ########
#########################################################################################################

import numpy as np
import sys

sys.path.append("helper_functions/")
sys.path.append("../helper_functions/")

from Fish import Fish  # noqa E402
from Aquarium_SinglePlane import SinglePlaneAquarium  # noqa E402


class MultiPlaneAquarium(SinglePlaneAquarium):
    """Represents an aquarium which has multiple planar boundaries (>=2). Parent class "SinglePlaneAquarium":"""

    __doc__ += SinglePlaneAquarium.__doc__  # type: ignore

    def __init__(self, num_image_reflections=3, _init_tests=True, **kwds):
        self.degree = num_image_reflections

        super().__init__(**kwds, _init_tests=False)
        assert len(self.boundaries) > 1, (
            "This aquarium must have more than one boundary, "
            "typically representing walls of tank, water surface or bottom of tank."
        )
        for b in self.boundaries:
            assert b.get_type().lower() == "plane", "All boundaries for this aquarium should be planar."

        # Run assertion tests
        if _init_tests:
            print(self.run_tests())

    def compute_image_point_currents(self, fish):
        assert isinstance(fish, Fish), "Need a fish object to compute the image point currents."
        currents_mag = fish.get_point_currents_magnitude()
        currents_loc = fish.get_point_currents_location()
        assert (
            currents_mag.shape[0] == currents_loc.shape[0]
        ), "Magnitudes and locations of point currents numbers should match."
        assert currents_loc.shape[1] == 3, "Locations of point currents should be shape (N,3)."

        # image_mag = np.array([])
        # image_loc = np.array([]).reshape([0,3])

        # prepare the dict which keeps track of the degree of the reflection and the reflection planes themselves
        images_dict = {"0": {"-1": currents_loc}}  # 0th degree reflection is the real currents
        for deg in np.arange(self.degree) + 1:
            images_dict[str(deg)] = {}  # prepare the dict which keeps the deg'th degree reflections
            for k in images_dict[str(deg - 1)]:
                last_reflection_id = int(
                    k.split("<>")[-1]
                )  # keep track of the last reflection - cannot reflect current image with respect to this boundary
                for bi, bound in enumerate(self.boundaries):
                    if bi != last_reflection_id:
                        k_current = k + "<>" + str(bi)  # current image's reflection history
                        # reflect point currents with respect to planar boundary
                        images_dict[str(deg)][k_current] = (
                            images_dict[str(deg - 1)][k]
                            - 2
                            * ((images_dict[str(deg - 1)][k] - bound.get_point()) @ bound.get_normal()).reshape(-1, 1)
                            * bound.get_normal()
                        )

        # parse the dict to recover all image locations and magnitudes
        image_mag = np.array([])
        image_loc = np.array([]).reshape([0, 3])
        for k1 in images_dict:
            if int(k1) > 0:
                for k2 in images_dict[k1]:
                    image_mag = np.hstack([image_mag, currents_mag])  # image magnitudes remain the same upon reflection
                    image_loc = np.vstack([image_loc, images_dict[k1][k2]])

        return image_mag, image_loc

    def details(self):
        """Provides the details of the object. Can be adjusted in child classes to include more information."""
        details_string = super().details()
        details_string += (
            "This aquarium type has multiple planar boundaries. It expands on the SinglePlaneAquarium class."
        )
        return details_string

    def run_tests(self):
        """Sanity assertion checks to ensure code robustness."""
        super().run_tests()
        return "Success!"
