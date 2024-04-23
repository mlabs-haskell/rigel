This is a visualizer to study howc KDTree works.

Run `./pybuild.py init venv` to setup the virtual env and install requirements.txt

Activate the virtual environment and run `python src/main.py`.

The purple point is the target point, whose nearest neighbour will be found.

You can move it around with WASD.

The green point is the deepest node closest to the target. It's an internal detail, feel free to ignore.

The orange point is the cursor point.

Press Left, Right to move to the left/right subtree of the cursor.
Press Up to go to the parent node of the cursor.

Press R to randomize the set of points.
Press T to randomize the target.
