import numpy as np
import random, operator, pandas as pd
import matplotlib.pyplot as plt
import time 
import skfuzzy as fuzz 
from skfuzzy import control as ctrl 

dataset = pd.read_csv("Admission_Predict.csv", index_col=0)

#inputs
gre = ctrl.Antecedent(np.arange(0, 341, 1), 'GRE score')
uRating = ctrl.Antecedent(np.arange(1, 6, 1), 'University rating')
gpa = ctrl.Antecedent(np.arange(0, 11, 0.01), 'GPA')

#output
chance = ctrl.Consequent(np.arange(0, 1, 0.01), 'Chance of admit')

#membership functions
gre.automf(3)
uRating.automf(3)
gpa.automf(3)
chance.automf(3) 

gre.view()
raw_input("Press Enter to continue...")
uRating.view()
raw_input("Press Enter to continue...")
gpa.view()
raw_input("Press Enter to continue...")
chance.view()
raw_input("Press Enter to continue...")

#rules
rule1 = ctrl.Rule(gre['good'] and gpa['good'], chance['good'])
rule2 = ctrl.Rule(gre['average'] or gpa['average'], chance['average'])
rule3 = ctrl.Rule(gre['poor'] or gpa['poor'], chance['poor'])
rule4 = ctrl.Rule(uRating['poor'], chance['good'])
#rule5 = ctrl.Rule(uRating['average'], chance['average'])
#rule6 = ctrl.Rule(uRating['good'], chance['poor'])

#rule5.view()
#raw_input("Press Enter to continue...")

chance_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

myChance = ctrl.ControlSystemSimulation(chance_ctrl)

myChance.input['GRE score'] = 337
myChance.input['University rating'] = 4
myChance.input['GPA'] = 9.65

myChance.compute()

print myChance.output['Chance of admit']
chance.view(sim = myChance)
raw_input("Press Enter to continue...")