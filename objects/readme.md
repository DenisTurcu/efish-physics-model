Objects relevant for the physics model, implementing methods for interactions between them.

## ElectricObject
Base object to start building more complex electric objects that share some common electric properties. The `ElectricObject` class has the following methods:

1. `__init__(self, conductivity, relative_permittivity, assert_err, _init_tests)`: This is the constructor method that initializes an instance of the `ElectricObject` class. It takes several optional parameters such as `conductivity`, `relative_permittivity`, `assert_err`, and `_init_tests`. It also converts the `conductivity` value to the main SI unit using the `convert2mainSI` function and initializes other instance variables. It runs assertion tests if `_init_tests` is `True`.

2. `__str__(self) -> str`: This method provides a string representation of the object when the `print` function is used. It calls the `details()` method to get the details of the object.

3. `details(self) -> str`: This method returns a string containing the details of the object, including the conductivity, relative permittivity, and assertion allowed error.

4. `is_equal(self, other, simple_return) -> bool | tuple`: This method compares the current object with another `ElectricObject` object. It checks if the conductivity and relative permittivity of both objects are within a certain tolerance. It returns a simple boolean comparison or a detailed comparison as a tuple.

5. `get_conductivity(self) -> float`: This method returns the conductivity of the object.

6. `get_resistivity(self) -> float`: This method calculates and returns the resistivity of the object using the `rho24sig` function.

7. `get_relative_permittivity(self) -> float`: This method returns the relative permittivity of the object.

8. `get_input_argument_names(self, print_arguments)`: This method returns the names of the input arguments required to create a new object. It also has an optional parameter `print_arguments` which, if set to `True`, prints the arguments.

9. `run_tests(self) -> str`: This method runs tests to ensure the functionality of the object. It checks if the conductivity and relative permittivity values match the values obtained from their respective getter methods.

10. `_initialize_input_argument_names(cls) -> list[str]`: This is a class method that returns a list of input arguments required to create a new object of the `ElectricObject` class.


## Fish
Base "fish" implementing basic functionality to work with Fish objects in the physics model. The `Fish` class has the following methods:

1. `__init__`: Initializes a `Fish` object with various parameters such as receptor locations, receptor normals, point currents magnitudes, etc.

2. `compute_transdermal_signal`: Computes the transdermal signal processed by receptors on the fish's skin based on the electric field, water conductivity, and temporal waveform.

3. `compute_receptors_responses`: Applies the receptor filters to the transdermal signal to compute individual responses of the fish receptors.

4. `update_receptors`: Updates the receptors of the fish by moving the fish in space.

5. `update_point_currents`: Updates the point currents of the fish by moving the fish in space.

6. `update_eod_wave_form_and_delay`: Updates the EOD wave form and delay of the fish.

7. `update_receptors_filters`: Updates the receptor filters of the fish.

8. `is_equal`: Compares the current fish to another fish for equality.

9. `visualize_scatter`: Visualizes the fish as scatter points, with optional intensity coloring and other customization options.

10. `create_scatter_graph_obj`: Creates a plotly graphical object to represent the fish as scatter points.

11. `create_normals_graph_obj`: Creates a plotly graphical object to represent the normals on the fish's skin.

12. `create_point_currents_graph_obj`: Creates a plotly graphical object to represent the point currents within the fish.

13. `visualize_triangulation`: Visualizes the surface of the fish using the triangulation method.

14. `details`: Provides the details of the fish.

15. `get_ID`: Extracts the ID property of the fish object.

16. `get_receptors_locations`: Extracts the receptors locations property of the fish object.

17. `get_receptors_normals`: Extracts the receptors normals property of the fish object.

18. `get_sampling_rate`: Extracts the sampling rate property of the fish object.

19. `get_eod_wave_form`: Extracts the EOD wave form property of the fish object.

20. `get_time_stamps`: Extracts the time stamps property of the fish object.

21. `get_skin_resistivity`: Extracts the skin resistivity property of the fish object.

22. `get_point_currents_magnitude`: Extracts the point currents magnitudes property of the fish object.

23. `get_point_currents_location`: Extracts the point currents locations property of the fish object.

24. `get_N_receptors`: Extracts the number of receptors property of the fish object.

25. `get_receptors_filters`: Extracts the receptor filters property of the fish object.

26. `get_N_filters`: Extracts the number of filters property of the fish object.

27. `_initialize_input_argument_names`: Initializes the input argument names for the fish object.

28. `run_tests`: Runs tests to ensure the correctness of the fish object.

These methods provide functionality for initializing and manipulating a `Fish` object, computing signals and responses, visualizing the fish, and extracting various properties of the fish.

### Generation
This class builds on its parent `Fish` class and sets the basis for fish objects used with the physics model. If receptor locations and associated normals can be provided directly for an un-bent fish, this class can be used on it's own for further simulations. It also sets a basis for constructing artificially looking fish, such as `IceCreamConeFish` and `CapsuleFish`. The `FishGeneration` class has the following methods:

1. `__init__(...)`: Initializes a `FishGeneration` object with various parameters such as `nose_position`, `fish_length`, `angle_yaw`, etc. It also calls the parent class `Fish`'s `__init__` method.

2. `update_receptors(...)`: Updates the receptors of the fish, either the base receptors or all receptors, based on the `base_receptors` parameter.

3. `update_parameters(...)`: Updates the spatial parameters of the fish such as `nose_position`, `angle_yaw`, `angle_pitch`, etc.

4. `initialize_point_currents(...)`: Defines the point currents of the fish along its length. The point currents are distributed uniformly and their sum is set to 0.

5. `initialize_receptors_and_normals(...)`: Initializes the locations and surface normals of the fish's receptors based on the provided initialization method.

6. `init_receptors_and_normals_random(...)`: Initializes the receptors and normals randomly on the fish's surface.

7. `init_receptors_and_normals_random_uniDense(...)`: Initializes the receptors and normals uniformly randomly with uniform density on the fish's surface.

8. `init_receptors_and_normals_grid(...)`: Initializes the receptors and normals on a grid on the fish's surface.

9. `init_receptors_and_normals_grid_uniDense(...)`: Initializes the receptors and normals on a grid with uniform density on the fish's surface.

10. `init_receptors_and_normals_manual(...)`: Initializes the receptors and normals based on manually provided points and normals.

11. `bend_and_rigid_transform(...)`: Transforms the locations and normals of the fish based on its bends and overall rotation.

12. `initialize_fish_bend_details(...)`: Initializes the bend details of the fish, including the bend locations and angles.

13. `initialize_bend_rotations(...)`: Initializes the rotation objects corresponding to the fish's bends.

14. `initialize_main_rotation(...)`: Initializes the rotation object corresponding to the overall rotation of the fish.

15. `is_equal(...)`: Compares the `FishGeneration` object with another object of the same class or its subclasses to check for equality.

16. `details(...)`: Provides details about the `FishGeneration` object, including its properties such as `nose_position`, `length`, `yaw`, etc.

17. Getter methods: `get_nose_position()`, `get_length()`, `get_yaw()`, `get_pitch()`, `get_roll()`, `get_bend_locations()`, `get_bend_angle_lateral()`, `get_bend_angle_dorso_ventral()`. These methods extract the corresponding properties of the fish object.

18. `_initialize_input_argument_names(...)`: Initializes the input argument names for the `FishGeneration` class.

19. `run_tests(...)`: Runs tests to check the correctness of the fish object's properties.

Note: Some methods inherit from the parent class `Fish` and have their documentation strings updated accordingly.

### IceCreamCone


### Capsule
The `CapsuleFish` class has the following methods:

1. `__init__(...)`: Initializes a capsule-shaped fish object with a head, body, and tail. It takes optional arguments for the rostro-caudal semi-axis of the tail cap.

2. `is_equal(...)`: Compares the current fish object with another fish object to check if they are equal. Returns a boolean value or a tuple of boolean values.

3. `init_receptors_and_normals_random(...)`: Initializes the receptors and normals randomly on the fish surface. It takes a dictionary of receptor details as input and returns the receptor locations and normals.

4. `init_receptors_and_normals_grid_uniDense(...)`: Initializes the receptors and normals on the fish surface in a uniform grid pattern. It takes a dictionary of receptor details as input and returns the receptor locations and normals.

5. `details()`: Provides the details of the CapsuleFish class.

6. `get_rostro_caudal_semi_axis_tail()`: Returns the rostro-caudal semi-axis of the tail cap.

7. `run_tests()`: Runs tests for the CapsuleFish class.


## Aquarium
The `Aquarium` class has the following methods:

1. `__init__(self, fish_objs=[], worm_objs=[], boundaries=[], _init_tests=True, **kwds)`: Initializes the aquarium for the fish. It takes optional parameters for fish objects, worm objects, and boundaries. It initializes the parent class properties, initializes the aquarium boundaries, empties the aquarium, inserts provided objects, and runs assertion tests.

2. `initialize_boundaries(self, boundaries: list[tuple[str, dict]])`: Initializes the boundaries of the aquarium based on the provided list of boundary types and properties.

3. `_insert_object(self, obj_list, obj, obj_instance, image_currents=False) -> list[tuple | Worm | Fish]`: Inserts the given object into the aquarium. If the object is a fish, it also inserts or updates its associated image currents.

4. `_remove_object(self, obj_list, obj, obj_instance, image_currents=False) -> list[tuple | Worm | Fish]`: Removes the given object from the aquarium. If the object is a fish, it also removes or updates its associated image currents.

5. `empty_aquarium(self)`: Empties the aquarium by removing all fish and worms.

6. `remove_all_fish(self)`: Removes all fish from the aquarium, including their image currents.

7. `remove_all_worms(self)`: Removes all worms from the aquarium.

8. `update_image_currents(self, obj: Fish, mode="insert")`: Updates the image currents generated by the boundaries for a specific fish object.

9. `insert_fish(self, obj: Fish, _check_points=True)`: Inserts a fish object into the aquarium.

10. `insert_worm(self, obj: Worm, _check_points=True)`: Inserts a worm object into the aquarium.

11. `remove_fish(self, obj: Fish)`: Removes a fish object from the aquarium.

12. `remove_worm(self, obj: Worm)`: Removes a worm object from the aquarium.

13. `electric_potential_and_field_single_fish(self, points: np.ndarray, fish_id: int, return_potential=True, return_field=True, include_image_point_currents=True) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]`: Computes the electric potential and field at given points due to a single fish in the aquarium.

14. `electric_potential_and_field(self, points: np.ndarray, return_potential=True, return_field=True) -> tuple[np.ndarray | None, np.ndarray | None]`: Computes the electric potential and field at given points due to all fish in the aquarium.

15. `transdermal_signal_for_all_fish(self, include_perturbations=True) -> tuple[np.ndarray, np.ndarray]`: Computes the electric signals for all fish in the aquarium.

16. `select_bounded_points(self, points: np.ndarray, simple_return: bool) -> np.ndarray | tuple[np.ndarray, np.ndarray]`: Selects only the points within the aquarium's boundaries from the given points.

17. `verify_points(self, points: list | np.ndarray, obj: tuple | Worm | Fish | None = None, simple_return=True) -> bool | tuple[bool, list | np.ndarray]`: Verifies if the provided points are located within the boundary limits of the aquarium.

18. `update_time_stamps(self)`: Updates the time stamps of the aquarium.

19. `visualize_aquarium(self, potential: np.ndarray | None = None, E_field: np.ndarray | None = None, electric_signal: np.ndarray | None = None, time_points: int | list | tuple = 10, shown_time_prefix: str = "", range_points: int = 30, lengths_prefix: str = "", show_fish: int = 0, show_electric_signal: bool = False, show_normals: int = 0, show_point_currents: int = 0, show_image_currents: int = 0, show_worms: bool | int = False, show_boundaries: bool = False, show_potential: bool = False, show_field: bool = False, potential_double_sided_percentile: float = 10, E_field_double_sided_percentile: float = 10, include_active_objects: bool = False, include_passive_objects: bool = False, electric_signal_perturbation_only: bool = False, xaxis_title: str = "X AXIS TITLE", yaxis_title: str = "Y AXIS TITLE", zaxis_title: str = "Z AXIS TITLE", fig_width: int = 800, fig_height: int = 800) -> go.Figure`: Visualizes the aquarium with fish, worms, boundaries, electric potential, and electric field.

20. `create_image_currents_graph_obj(self, id_fish: int, marker_size: int = 15, units_prefix: str = "") -> go.Scatter3d`: Creates a plotly graphical object to show the image point currents of a specific fish in the aquarium.

21. `details(self)`: Provides the details of the aquarium, including its boundaries.

22. `get_boundaries(self) -> list[Boundary]`: Returns the boundaries of the aquarium.

23. `get_fish(self) -> list[Fish]`: Returns the fish objects in the aquarium.

24. `get_worms(self) -> list[Worm]`: Returns the worm objects in the aquarium.

25. `get_image_point_currents(self) -> tuple[list, list]`: Returns the image point currents of the fish objects in the aquarium.

26. `get_time_stamps(self) -> np.ndarray`: Returns the time stamps of the aquarium.

27. `_initialize_input_argument_names(cls)`: Initializes the input argument names for the aquarium class.

28. `run_tests(self) -> str`: Runs tests for the Aquarium class, including fish and worm insertion and removal.

### SinglePlane
The `SinglePlaneAquarium` class implements the following methods:

1. `__init__(self, _init_tests=True, **kwds)`: This method is the initialization method for the `SinglePlaneAquarium` class. It calls the initialization method of the parent class `Aquarium` and performs some additional checks and assertions specific to the `SinglePlaneAquarium` class.

2. `update_image_currents(self, obj: Fish, mode: str = "insert")`: This method updates the image currents generated by the single planar boundary. It calls the `update_image_currents` method of the parent class `Aquarium` and performs additional computations specific to the `SinglePlaneAquarium` class.

3. `compute_image_point_currents(self, fish: Fish) -> tuple[np.ndarray, np.ndarray]`: This method computes the image points for a single planar boundary by reflecting the fish point currents with respect to the boundary. It returns the magnitude and location of the image point currents.

4. `details(self)`: This method provides the details of the `SinglePlaneAquarium` object. It calls the `details` method of the parent class `Aquarium` and adds additional information specific to the `SinglePlaneAquarium` class.

In addition to these methods, the `SinglePlaneAquarium` class also inherits methods from its parent class `Aquarium`.

### MultiPlane
Not yet implemented for this framework.


## Worm
Base "worm" referring to nearby object w.r.t. the fish. Defines the center position that is shared across worm types. The `Worm` class has the following methods:

1. `__init__(self, center_position: list | np.ndarray = [0, 0, 0], _init_tests: bool = True, **kwds)`: This method is the constructor of the `Worm` class. It initializes a `Worm` object with a center position in world coordinates. It also has an optional argument `_init_tests` to specify whether to run initialization tests. It calls the constructor of the parent class `ElectricObject` and initializes the `r_vec` attribute with the converted center position. It also runs assertion tests if `_init_tests` is `True`.

2. `is_equal(self, other: Self, simple_return: bool = True) -> bool | tuple`: This method compares the current `Worm` object with another `Worm` object for equality. It calls the `is_equal` method of the parent class `ElectricObject` and adds an additional comparison for the center position. It returns a boolean value indicating whether the objects are equal, or a tuple containing the boolean value, a list of truth values for each comparison, and a list of comparison names.

3. `create_graph_obj(self)`: This method raises a `NotImplementedError` and provides a message indicating that the function needs to be implemented according to the geometry of the specified object. This suggests that subclasses of `Worm` should override this method.

4. `details(self) -> str`: This method provides the details of the `Worm` object. It calls the `details` method of the parent class `ElectricObject` and appends the center position of the worm to the details string. It returns the details string.

5. `get_position(self) -> np.ndarray`: This method extracts the position of the `Worm` object. It returns the `r_vec` attribute, which represents the center position of the worm.

6. `get_points(self) -> np.ndarray`: This method extracts the surface points of the `Worm` object. For a point worm, it returns the center position as a single point. For larger worms, this method should be overridden to return multiple points from the surface of the worm.

7. `_initialize_input_argument_names(cls) -> list[str]`: This is a class method that initializes the input argument names for the `Worm` class. It calls the `_initialize_input_argument_names` method of the parent class `ElectricObject` and appends the argument name for the center position.

8. `run_tests(self)`: This method runs tests for the `Worm` object. It calls the `run_tests` method of the parent class `ElectricObject` and asserts that the `r_vec` attribute matches the position returned by `get_position()`. It returns a success message if all tests pass.

Note: There are additional methods referenced from the `Worm` class in the `Aquarium.py` file.


### SmallSpherical
A small spherical "worm" represents nearby objects w.r.t. the fish that can take different sizes, resistances, capacitances, positions. The `SmallSphericalWorm` class has the following methods:

1. `__init__(self, radius, resistance, capacitance, derivative_filter, _init_tests, **kwds)`: This method initializes the `SmallSphericalWorm` object. It takes several optional parameters such as `radius`, `resistance`, `capacitance`, and `derivative_filter`. It also calls the `__init__` method of the parent class `Worm` and sets the `radius`, `der_filt`, and electrical properties of the worm. It can also run assertion tests if `_init_tests` is set to `True`.

2. `differential_equation_coefficients(self, sig_water, eps_r_water) -> tuple`: This method computes the coefficients of the differential equation that solves the distortion waveform based on the physics model results. It takes the conductivity of water (`sig_water`) and the relative permittivity of water (`eps_r_water`) as input and returns the coefficients as a tuple.

3. `dipole_waveform_single_fish(self, wave_form, sig_water, eps_r_water, sampling_frequency) -> np.ndarray`: This method computes the waveform of the induced dipole based on relevant computations from the physics model. It takes the original waveform (`wave_form`), conductivity of water (`sig_water`), relative permittivity of water (`eps_r_water`), and the sampling frequency (`sampling_frequency`) as input. It returns the waveform of the induced dipole as a NumPy array.

4. `dipole_waveform_quality_check(self, dipole_wave_form, wave_form, wave_form_der, sampling_frequency, f0, f1, p0, p1, resolution_multiplication, runtime_err_factor, check_err) -> np.ndarray`: This method checks the quality of the solution of the induced dipole waveform. It takes the dipole waveform (`dipole_wave_form`), original waveform (`wave_form`), derivative of the original waveform (`wave_form_der`), sampling frequency (`sampling_frequency`), and the coefficients of the differential equation (`f0`, `f1`, `p0`, `p1`) as input. It returns the adjusted dipole waveform as a NumPy array.

5. `perturbation_magnitude_single_fish(self, receptor_locs, E_field, return_potential, return_field, include_radius_factor) -> tuple`: This method returns the electric potential/field perturbation magnitude at given locations. It takes the receptor locations (`receptor_locs`), electric field at the center of the worm (`E_field`), and optional parameters (`return_potential`, `return_field`, `include_radius_factor`) as input. It returns a tuple containing the potential perturbation and the electric field perturbation.

6. `perturbation_trace(self, receptor_locs, aquarium_obj, return_potential, return_field) -> tuple`: This method returns the electric potential/field perturbations computed at given locations. It takes the receptor locations (`receptor_locs`), an `Aquarium` object (`aquarium_obj`), and optional parameters (`return_potential`, `return_field`) as input. It returns a tuple containing the potential perturbation and the electric field perturbation.

7. `initiate_electrical_properties(self, resistance, capacitance)`: This method parses the electric properties of the worm. It takes the resistance and capacitance as input and sets the `R` and `C` attributes of the worm accordingly.

8. `create_graph_obj(self, size_scale, units_prefix) -> go.Scatter3d`: This method creates a plotly graphical object representing the worm. It takes optional parameters `size_scale` and `units_prefix` and returns a `go.Scatter3d` object.

9. `is_equal(self, other, simple_return) -> bool | tuple`: This method checks if the current worm object is equal to another worm object. It takes another `Worm` object (`other`) and an optional parameter (`simple_return`) as input. It returns a boolean value or a tuple containing the boolean value, truth values, and comparison results.

10. `details(self) -> str`: This method provides the details of the small spherical worm. It returns a string containing the details.

11. `get_radius(self) -> float | np.ndarray`: This method extracts the worm's radius. It returns the radius as a float or a NumPy array.

12. `_initialize_input_argument_names(cls) -> list[str]`: This is a class method that initializes the input argument names for the `SmallSphericalWorm` class. It returns a list of input argument names.


## Boundaries
The `Boundary` class represents a parent class for different types of boundaries in an aquarium. It serves as a blueprint for creating specific boundary types such as planar boundaries or spherical boundaries.

The class has several methods and attributes:

1. `__init__(self, central_point: list | np.ndarray, assert_err: float = 1e-12)`: This is the constructor method that initializes the boundary object. It takes two parameters:
   - `central_point`: The reference point of the boundary. The type can be either a list or a NumPy array. The central point helps compute properties specific to the boundary type, such as normals for planar boundaries or the center for spherical boundaries.
   - `assert_err` (optional): This parameter specifies the assertion error value for what counts as "0" error. It has a default value of `1e-12`.

2. `get_reference_point(self)`: This method returns the central point of the boundary.

3. `get_type(self)`: This method returns the type of the boundary.

4. `verify_points(self, points: list | np.ndarray, simple_return: bool = True) -> bool | tuple`: This method verifies whether provided points are located within the boundary limits. It takes two parameters:
   - `points`: The given points' locations to test for positioning with respect to the boundary. The type can be either a list or a NumPy array.
   - `simple_return` (optional): This parameter determines whether to return only the truth value of the test or return the processed points as well with individual tests for each point. It has a default value of `True`.
   
   The method performs the following steps:
   - Converts the `points` parameter into a NumPy array.
   - Reshapes the array if it has a shape of `(N,)` to `(N, 3)`, assuming each point has three coordinates.
   - Asserts that the shape of the points array is `(N, 3)`, where `N` is the number of points. If the assertion fails, it raises an `AssertionError`.
   - If `simple_return` is `True`, it returns `False` to indicate that the points are not within the boundary.
   - If `simple_return` is `False`, it returns a tuple containing `False` and the processed points array.

Overall, the `Boundary` class provides a foundation for defining and working with different types of boundaries in an aquarium.

### Planar
The provided code implements a class called `Plane` that represents a planar boundary. This class is a subclass of another class called `Boundary`. 

The `Plane` class has the following functionalities:

1. Initialization: The `__init__` method initializes a planar boundary for an aquarium. It takes a `normal` parameter, which is a list or numpy array representing the normal vector that defines the direction of the half-space within the boundary limits. The `normal` vector is normalized to have a unit length. The method also calls the `__init__` method of the parent class `Boundary` using the `super()` function.

2. Verification of Points: The `verify_points` method verifies whether a given set of points lies within the specific "Plane" boundary. It takes a `points` parameter, which is a set of points to be verified. The method calls the `verify_points` method of the parent class `Boundary` using `super().verify_points(points, simple_return=False)`. It then checks if the points satisfy the boundary condition by calculating the dot product between the difference of each point and the central point with the normal vector. If `simple_return` is `True`, it returns a boolean value indicating whether all points satisfy the boundary condition. Otherwise, it returns a tuple containing a boolean value for each point and the points themselves.

3. Extraction of Normal Vector: The `get_normal` method returns the normal vector of the planar boundary. It simply returns the `self.normal` attribute.

The `Plane` class inherits functionality from the `Boundary` class, which is not shown in the provided code excerpt.

### Ellipsoid
Not yet implemented

### Spherical
Not yet implemented

Disclaimer: Writing this README file has been aided by GitHub Copilot.