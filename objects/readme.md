Objects relevant for the physics model, implementing methods for interactions between them.

## ElectricObject
Base object to start building more complex electric objects that share some common electric properties.

## Fish
Base "fish" implementing basic functionality to work with Fish objects in the physics model. 
- `__init__(...)`: Initializes a Fish object with various parameters.
- `compute_transdermal_signal(...)`: Computes the transdermal signal processed by receptors on the fish's skin.
- `compute_receptors_responses(...)`: Applies the receptor filters to the transdermal signal to compute individual responses of the fish receptors.
- `update_receptors(...)`: Updates the receptors of the fish, i.e., moves the fish in space.
- `update_point_currents(...)`: Updates the point currents of the fish.
- `update_eod_wave_form_and_delay(...)`: Updates the EOD wave form and delay of the fish.
- `update_receptors_filters(...)`: Updates the receptor filters of the fish.
- `is_equal(...)`: Checks if the current fish object is equal to another fish object.
- `visualize_scatter(...)`: Visualizes the fish as a scatter plot.
- `create_scatter_graph_obj(...)`: Creates a scatter graph object for visualization.
- `create_normals_graph_obj(...)`: Creates a cone graph object for visualizing the normals of the fish.
- `create_point_currents_graph_obj(...)`: Creates a scatter graph object for visualizing the point currents of the fish.
- `visualize_triangulation(...)`: Visualizes the triangulation of the fish.
- `details(...)`: Returns a string with details about the fish object.
- `get_ID(...)`: Returns the ID of the fish object.
- `get_receptors_locations(...)`: Returns the locations of the fish's receptors.
- `get_receptors_normals(...)`: Returns the normals to the skin surface at the receptor locations.
- `get_sampling_rate(...)`: Returns the sampling rate of the EOD wave form.
- `get_eod_wave_form(...)`: Returns the wave form of the EOD of the fish.
- `get_time_stamps(...)`: Returns the time stamps of the fish object.
- `get_skin_resistivity(...)`: Returns the resistivity of the fish's skin.
- `get_point_currents_magnitude(...)`: Returns the magnitudes of the point currents of the fish.
- `get_point_currents_location(...)`: Returns the locations of the point currents of the fish.
- `get_N_receptors(...)`: Returns the number of receptors of the fish.
- `get_receptors_filters(...)`: Returns the temporal convolutional filters for the receptors.
- `get_N_filters(...)`: Returns the number of filters for the receptors.
- `run_tests(...)`: Runs tests on the fish object.

### Generation
### Capsule
### IceCreamCone


## Aquarium
### SinglePlane
### MultiPlane


## Worm
Base "worm" referring to nearby object w.r.t. the fish. Defines the center position that is shared across worm types.
### SmallSpherical
A small spherical "worm" represents nearby objects w.r.t. the fish that can take different sizes, resistances, capacitances, positions.


## Boundaries
### Planar
### Spherical



