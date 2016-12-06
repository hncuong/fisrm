import numpy as np
import pandas as pd
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
