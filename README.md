# Artificial Inteligence for Dino Game

Artificial Inteligence implementation for the Dino game

## Steps

This implementation was made in Python3.8.2. Before running the application, it's necessary to install some dependencies, for that run the following commands in your terminal:

```
pip3 install -r requirements.txt
```


### Graphic
To fast train the model, theres also a version without the graphics, for that run:

```
python3 dinoAIrenderless.py
```

## KNN Model
The model consists of a genetic algorithm to search the coordinates of the points used in the KNN classifier.

To run it, you may run the application as:

```
python3 mainGA.py
```

### Best Result
Solution: `[212 345 142 101 300 131 394 345  43 651 300 383 300 300 130  15 325 732 171 345 127]`
Results: `[1089.5, 210.0, 219.0, 1113.75, 1200.5, 1262.25, 287.0, 331.25, 181.0, 226.0, 1609.75, 111.25, 168.75, 183.0, 1068.0, 337.25, 941.5, 163.5, 136.75, 465.5, 1123.25, 1264.25, 276.75, 1196.5, 296.75, 1145.5, 1133.5, 117.0, 1135.75, 1139.75]`
Mean: 671.15  Std: 484.46


## Rule Based Model
The model consists of a series of rules and facts, from which it is possible to create new facts and decide the action to be made by the dinossaur.

To run it, you may run the application as:

```
python3 mainRuleBasedClassifier.py
```

### Best Result 
Results: `[1206.0, 1291.25, 899.75, 1619.5, 1240.75, 979.0, 1305.25, 1255.5, 1341.5, 525.5, 1100.5, 1252.0, 1090.0, 1260.5, 1024.0, 1327.25, 1088.5, 1055.5, 1419.25, 1193.0, 1277.25, 1250.75, 1255.0, 1093.75, 1222.5, 1328.0, 1187.0, 1018.0, 1384.5, 977.0]`
Mean: 1182.275  Std: 193.09

---

Implemented by Sophie Dilhon
