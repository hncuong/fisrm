import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt

def positive_normal_random_gen(mu = 15,sigma=30, size=1000):
    count = 0
    ran_list = []
    while (count < size):
        a = np.random.normal(mu, sigma)
        if (a >= 0):
            ran_list.append(int(a))
            count = count + 1
            if (count >= size):
                break
    # count = np.zeros(300)
    # for a in ran_list:
    #     count[a] = count[a]+1
    # plt.figure(1)
    # plt.plot(count)
    return np.array(ran_list)

# generate data
data_size = 10000

data = pd.DataFrame(np.random.randint(low=0, high=5, size=(data_size, 4)), columns=['TQ','DPQ', 'C', 'OU'])


data ['DI']= positive_normal_random_gen(mu=15,sigma=30,size=data_size)
data ['DFT']=positive_normal_random_gen(mu=-60,sigma=35,size=data_size)
data ['DFO']=positive_normal_random_gen(mu=-60,sigma=35,size=data_size)
data ['RD']=positive_normal_random_gen(mu=-60,sigma=25,size=data_size)

data.to_csv("fisrm.csv", index=False);
# xac dinh phan phoi cho cac node o input layer
# cpd_tq = TabularCPD(variable='TQ', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])
# cpd_c = TabularCPD(variable='C', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])
# cpd_dpq = TabularCPD(variable='DPQ', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])
# cpd_ou = TabularCPD(variable='OU', variable_card=5, values=[[0.2, 0.2,0.2,0.2,0.2]])

# dinh nghia cau truc mang bayes 
model = BayesianModel([('TQ', 'DFT'), ('DPQ', 'DI'), ('C','DI'),('DI','DFT'),('DI','RD'),('DFT','RD'),('RD','DFO'),('OU','DFO')])

model.fit(data, estimator_type=BayesianEstimator, prior_type="BDeu",equivalent_sample_size=10) # default equivalent_sample_size=5
# for cpd in model.get_cpds():
#     print(cpd)
# print model.get_cpds()[2]


infer = VariableElimination(model)
# DI_distribution = infer.query(['RD']) ['RD'].values
DI_distribution = infer.query(['DFT'], evidence={'DPQ': 2, 'C': 3, 'TQ': 4,'OU':1})['DFT'].values
max_DI = np.argmax(DI_distribution)
print max_DI
print infer.query(['DPQ']) ['DPQ']
print model.get_cpds()[1]
plt.figure(2)
plt.plot(DI_distribution)
plt.show()