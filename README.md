# autonomous-driving-game
A simple python simulation for autonomous driving path planning and decision making. This is original coming from one of the programming assignment from stanford CS221: Artificial Intelligence. I modified the files to add path planning part and simple decision making strategy in making lane change

```python
python search.py
```

This command will search the drivable path from the current location and goal locaiton of the map and produce drivable path and drive on it. The map information is stored in the layout files.

```python
python game.py
```
This is for the simple decision making in lane changes, for instance the car is surrounded by other cars and then subject car would like to make lane change to align himsel with the goal lane.