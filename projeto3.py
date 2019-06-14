#MC906 - Introducao a Inteligencia Artificial
#1s2019 - Unicamp
#Prof. Esther Luna Colombini

#Raysa Masson Benatti RA 176483

#undergraduate project whose goal is to build a fuzzy system to solve a chosen problem
#here, we want to know what would be the probability, from 0 to 1, for an individual to be admitted into the US graduate system given some parameters
#original dataset provided by Mohan S Acharya, Asfia Armaan, Aneeta S Antony (https://bit.ly/2KDNa9O)
#this is a dataset canonically used to practice linear regression models
#the goal is to compare results provided by the original system (regression) vs. by different configurations of our system (fuzzy) 

import numpy as np
import pandas as pd
import skfuzzy as fuzz 
from skfuzzy import control as ctrl 


#--------------------------------------------------#
#aux functions

#computes MAE (mean absolute error) between reference results and fuzzy results
def mae(dataset):
    s = np.sum(dataset['Difference'])
    return s / dataset.shape[0]

#--------------------------------------------------#

#converts csv file into dataframe
dataset = pd.read_csv("Admission_Predict.csv", index_col=0)
#adds two more columns
dataset['Fuzzy chance of admit'] = 'Default'
dataset['Difference'] = 'Default'

#--------------------------------------------------#

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

#--------------------------------------------------#
#membership functions

#first choice: automatically provided triangular functions 
def tri():   
    gre.automf(3)
    gpa.automf(3)
    toefl.automf(3)
    chance.automf(3)
    
#second choice: custom gaussian functions 
def gauss(): 
    #last parameter = sigma value
    gre['poor'] = fuzz.gaussmf(gre.universe, 0, 60)
    gre['average'] = fuzz.gaussmf(gre.universe, 170, 60)
    gre['good'] = fuzz.gaussmf(gre.universe, 340, 60)
    gpa['poor'] = fuzz.gaussmf(gpa.universe, 0, 2)
    gpa['average'] = fuzz.gaussmf(gpa.universe, 5, 2)
    gpa['good'] = fuzz.gaussmf(gpa.universe, 10, 2)
    toefl['poor'] = fuzz.gaussmf(toefl.universe, 0, 20)
    toefl['average'] = fuzz.gaussmf(toefl.universe, 60, 20)
    toefl['good'] = fuzz.gaussmf(toefl.universe, 120, 20)
    chance['poor'] = fuzz.gaussmf(chance.universe, 0, 0.2)
    chance['average'] = fuzz.gaussmf(chance.universe, 0.5, 0.2)
    chance['good'] = fuzz.gaussmf(chance.universe, 1, 0.2)

#for rExp it would make no sense to use the standard triangular or gaussian functions
#let us customize it and use it like this in both cases 
rExp['no'] = fuzz.trimf(rExp.universe, [0, 0, 1])
rExp['yes'] = fuzz.trimf(rExp.universe, [0, 1, 1])

#for these parameters, a triangular function suits better
uRating.automf(3)
sop.automf(3)
lor.automf(3)

#comment tri() to run the system on gaussian membership functions, and vice-versa 
#tri()
gauss()

#--------------------------------------------------#
#membership functions graphs 
#this block can be left commented since it is not needed for the system to work
#the raw_input lines are necessary to avoid a bug (at least in my work environment) 
"""
gre.view()
raw_input("Press Enter to continue...")
uRating.view()
raw_input("Press Enter to continue...")
gpa.view()
raw_input("Press Enter to continue...")
chance.view()
raw_input("Press Enter to continue...")
toefl.view()
raw_input("Press Enter to continue...")
sop.view()
raw_input("Press Enter to continue...")
lor.view()
raw_input("Press Enter to continue...")
rExp.view()
raw_input("Press Enter to continue...")
"""
#--------------------------------------------------#
#rules

rule1 = ctrl.Rule(gre['good'] & gpa['good'], chance['good'])
rule2 = ctrl.Rule(gre['poor'] | gpa['poor'] | rExp['no'], chance['poor'])
rule3 = ctrl.Rule(uRating['poor'], chance['good'])
rule4 = ctrl.Rule(toefl['poor'] | sop['poor'] | lor['poor'], chance['poor'])

#--------------------------------------------------#
#running the system 

#selects rules to build the system on, among the ones described above 
#to shut off some rule, just delete it from the list of parameters below 
chance_ctrl = ctrl.ControlSystem([rule1, rule3, rule4])

myChance = ctrl.ControlSystemSimulation(chance_ctrl)

#computes fuzzy result for each one of the rows 
#to run with a subset of the rules described above, comment lines of inputs that are not used 
for i in range(dataset.shape[0]):
    myChance.input['GRE score'] = dataset.iat[i, 0]
    myChance.input['TOEFL score'] = dataset.iat[i, 1]
    myChance.input['University rating'] = dataset.iat[i, 2]
    myChance.input['Statement of Purpose strength'] = dataset.iat[i, 3]
    myChance.input['Letter of Recommendation strength'] = dataset.iat[i, 4]
    myChance.input['GPA'] = dataset.iat[i, 5]
    #myChance.input['Research experience'] = dataset.iat[i, 6]
    myChance.compute()
    #the result given by the fuzzy system is our system's output, rounded to 2 decimal places  
    dataset.iat[i, 8] = round(myChance.output['Chance of admit'], 2) 
    #let us compare the results 
    dataset.iat[i, 9] = abs(dataset.iat[i,7] - dataset.iat[i,8])

print(dataset.head(10))

#--------------------------------------------------#
#results analysis

#creates two more dataframes for results storaging 
bestResults = pd.read_csv("Best_Results.csv", index_col=0)
worstResults = pd.read_csv("Worst_Results.csv", index_col=0)

min_difference = dataset['Difference'].min()
print(min_difference)
max_difference = dataset['Difference'].max()
print(max_difference)

#feeds recently created dataframes with information 
#the best and worst matches from the original dataset will be put into them 
#good match: result given by the fuzzy system has little or no difference from the original result (min_difference)
#bad match: result given by the fuzzy system has considerable difference from the original result (max_difference)
#we select those matches and copy all their info into datasets bestResults and worstResults 
k=0
r=0
for i in range(dataset.shape[0]):
    if(dataset.iat[i, 9] == min_difference):
        for j in range(0, bestResults.shape[1]):
            if(j == 0):
                bestResults.iat[k, j] = dataset.index[i]
            else:
                bestResults.iat[k, j] = dataset.iat[i, j-1]
        k+=1
    if(dataset.iat[i, 9] == max_difference):
        for j in range(0, worstResults.shape[1]):
            if(j == 0):
                worstResults.iat[r, j] = dataset.index[i]
            else:
                worstResults.iat[r, j] = dataset.iat[i, j-1]
        r+=1

print(bestResults.head(20))
print(worstResults.head(20))

#config sets which config is being used, so filename will be recorded accordingly 
#list of configs is described in this project's report and as follows:
#1 = rule1+rule2+rule3+rule4; tri()
#2 = rule1+rule3+rule4; tri()
#3 = rule1+rule2+rule4; tri()
#4 = rule1+rule2+rule3; tri()
#5 = S; gauss() | S = set of rules used in whatever config showed minimum value of mean absolute error
config = 5
bestResults.to_csv(r'' + str(config) + 'bestResults.csv')
worstResults.to_csv(r'' + str(config) + 'worstResults.csv')
dataset.to_csv(r'' + str(config) + 'finalDataset.csv')

print mae(dataset)

#chance.view(sim = myChance)
#raw_input("Press Enter to continue...")

#some links:
#https://scikit-fuzzy.readthedocs.io/en/latest/auto_examples/plot_tipping_problem_newapi.html#example-plot-tipping-problem-newapi-py
#https://scikit-fuzzy.readthedocs.io/en/latest/userguide/fuzzy_control_primer.html
#https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.control.html
#https://pythonhosted.org/scikit-fuzzy/api/api.html