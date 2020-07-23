# halite3_pytorch

This is a pytorch rewrite of my halite3_ml repo (which was originally written using TensorFlow).

The neural network architecture definition in this repo is slightly cleaner and more accessible 
compared to the original TF version.

I haven't performed any rigorous tests to ensure the final trained bot performs as well as the 
original bot submitted for competition, but a visual inspection suggests the bot plays just as 
well, if not better. Other improvements such as balancing class weights per game step and 
map size may be part of the reason for improved performance.

Note: This repo is mostly intended for reference. File paths will need to be restored in the code 
for the project to work properly.
