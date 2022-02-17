# jocl_cell_automata_alife_fix
High performance alife simulation utilizing the GPU via OpenCL
![Alt Text](https://github.com/jonahshader/jocl_cell_automata_alife_fix/blob/master/images/simulation_scale.gif)

This is an artificial life (alife) simulation where creatures can eat (food is green), place walls (walls are red), and reproduce, yielding natural pressures for evolution to occur. Creatures can see a radius around them in three colors and their brain is a recurrent neural network where each neuron takes in the weighted sum of the preceding layer from time t and the previous timestep t-1.

This alife simulation is written with performance in mind. I placed some constraints on the development for the sake of performance. For example, the quantity of creatures in the simulation is finite to avoid memory reallocation. This particular constraint makes it quite different from other alife simulators. Instead of creature reproducing and creating a number of offsprings, creatures in this simulation instead "reproduce" by reviving dead creatures and copying over genetics into the revived creature. Intelligent life still arises in this simulation despite this unconventional reproduction system.

Another constraint is the quantization of the world. Instead of having floating point numbers represent the location of creatures, I used integers and constrained the world to a grid. With this, collision detection runs in constant time, along with collision resolution, creature vision, and world modifications. This means the speed of the simulation is O(n) where n is the number of creatures. Otherwise, I would have had to use something like quadtree collision detection which runs in O(n lg n). 
