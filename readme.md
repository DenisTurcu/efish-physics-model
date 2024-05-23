## Example Simulation

The `example-simulation.ipynb` notebook demonstrates what this framework does. It is a simple tutorial on how to use it. When running this notebook, you should see the figures below being generated as you run the code. Running the notebook should take less than 10s, and most of this time is due to constructing the interactive 3D visualizations. The computation time should be much less than 1s.

#### Dummy eod for simulation
![image](viz-readme/viz-dummy-eod.png "Dummy EOD waveform visualization")

#### Simple fish visualization in 3D
The receptors on the skin surface of the fish are color-coded by the x-axis position. Point currents that generate the EOD are along the midline of the fish (+red/-blue). Not in this example the fish is not straigth, rather it has tail bend at multiple locations to mimic a swimming fish.
![image](viz-readme/viz-fish.png "Simple fish visualization")

#### simple aquarium visualization in 3D 
This includes a fish, a boundary (horizontal at z=0) that creates the image currents, and a worm. Receptor modulation due to worm presence should be noticeable on the skin, close to the worm (especially in the interactive plot). Equipotential surfaces are also shown during the fish discharge.
![image](viz-readme/viz-aquarium.png "Aquarium visualization")
