import numpy as np
import random, operator, pandas as pd
import matplotlib.pyplot as plt
import time 
import skfuzzy as fuzz 
from skfuzzy import control as ctrl 

#links uteis:
#https://scikit-fuzzy.readthedocs.io/en/latest/auto_examples/plot_tipping_problem_newapi.html#example-plot-tipping-problem-newapi-py
#https://scikit-fuzzy.readthedocs.io/en/latest/userguide/fuzzy_control_primer.html
#https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.control.html
#https://pythonhosted.org/scikit-fuzzy/api/api.html

dataset = pd.read_csv("Admission_Predict.csv", index_col=0)

#inputs
gre = ctrl.Antecedent(np.arange(0, 341, 1), 'GRE score')
uRating = ctrl.Antecedent(np.arange(1, 6, 1), 'University rating')
gpa = ctrl.Antecedent(np.arange(0, 10.1, 0.01), 'GPA')

toefl = ctrl.Antecedent(np.arange(0, 121, 1), 'TOEFL score')
sop = ctrl.Antecedent(np.arange(1, 5.5, 0.5), 'Statement of Purpose strength')
lor = ctrl.Antecedent(np.arange(1, 5.5, 0.5), 'Letter of Recommendation strength')
rExp = ctrl.Antecedent(np.arange(0, 2, 1), 'Research experience')

#output
chance = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'Chance of admit')

#membership functions
gre.automf(3)
uRating.automf(3)
gpa.automf(3)
chance.automf(3) 

toefl.automf(3)
sop.automf(3)
lor.automf(3)

gre.view()
raw_input("Press Enter to continue...")
uRating.view()
raw_input("Press Enter to continue...")
gpa.view()
raw_input("Press Enter to continue...")
chance.view()
raw_input("Press Enter to continue...")

#rules
rule1 = ctrl.Rule(gre['good'] & gpa['good'], chance['good'])
rule2 = ctrl.Rule(gre['poor'] | gpa['poor'], chance['poor'])
rule3 = ctrl.Rule(uRating['poor'], chance['good'])

rule4 = ctrl.Rule(toefl['poor'], chance['poor'])
rule5 = ctrl.Rule(sop['poor'], chance['poor'])
rule6 = ctrl.Rule(lor['poor'], chance['poor'])
rule7 = ctrl.Rule(rExp=0, chance['poor'])

#rule5.view()
#raw_input("Press Enter to continue...")

chance_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])

myChance = ctrl.ControlSystemSimulation(chance_ctrl)

myChance.input['GRE score'] = 314
myChance.input['TOEFL score'] = 103
myChance.input['University rating'] = 2
myChance.input['Statement of Purpose strength'] = 2
myChance.input['Letter of Recommendation strength'] = 3
myChance.input['GPA'] = 8.21
myChance.input['Research experience'] = 0


myChance.compute()

print myChance.output['Chance of admit']
chance.view(sim = myChance)
raw_input("Press Enter to continue...")