from numpy import random
import matplotlib.pyplot as plt
bandit = 10
pulls = 1000
mean=[]
std=[]

for i in range (bandit):
    m=random.randint(0,9)
    s=random.randint(0,9)
    mean.append(m)
    std.append(s)
print('Mean', mean)
print('Standard Deviation', std)
avm=sum(mean)/len(mean)
print(avm)
rewards=[]
print('Rewards:')
for i in range(bandit):
    x = random.normal(loc= mean[i], scale = std[i], size = pulls)
    print(x)
    rewards.append(x)

#  Scenario 1
# randomly choose any of the levers

average=[]
iteration=[]
lever=[]
final=[]
total=0
for i in range(pulls):
    iteration.append(i+1)
    l = random.randint(1,bandit)
    lever.append(l)
    final.append(rewards[l-1][i])
    total+=rewards[l-1][i]
    avg=total/(i+1)
    average.append(avg)
print(final)
print(lever)
print(total)
print(average)
plt.plot(iteration, average)
plt.xlabel('Tth Iteration')
plt.ylabel('Average')
plt.title('Average Reward vs Tth iteration')
plt.show()




print('/////////')


#  Scenario 2
# lever which has given us the maximum average reward till now
lever2=[]
final2=[]
total2=0
iteration2=[]
average2=[]
for i in range(bandit):
    iteration2.append(i + 1)
    l = i+1
    lever2.append(l)
    final2.append(rewards[l - 1][i])
    total2 += rewards[l - 1][i]
    avg = total2 / (i + 1)
    average2.append(avg)
print("Final",final2)
print("Lever",lever2)
print("Total",total2)
print("Average",average2)
maxrew=max(final2)
maxl=0
for i in range(len(final2)):
    if final2[i]==maxrew:
        maxl=lever2[i]
print(maxrew, maxl)
for i in range(bandit, pulls):
    iteration2.append(i + 1)
    total2 += maxrew
    avg = total2 / (i + 1)
    average2.append(avg)

plt.plot(iteration2, average2)
plt.xlabel('Tth Iteration')
plt.ylabel('Average')
plt.title('Average Reward vs Tth iteration')
plt.show()

print('/////////')
#  Scenario 3
#  epsilon
epsilon=random.uniform(0,1)
print(epsilon)
lever3=[]
average3=[]
iteration3=[]
final3=[]
total3=0
currentbestl=0
currentbestr=0
for i in range(pulls):
    iteration3.append(i + 1)
    lol=random.uniform(0,1)
    if lol<=epsilon:
        print('Value smaller than epsilon')
        iteration.append(i + 1)
        l=random.randint(1,bandit)
        print(f'Lever pulled: {l}')
        lever3.append(l)
        final3.append(rewards[l-1][i])
        total3+=rewards[l-1][i]
        avg=total3/(i+1)
        average3.append(avg)
        currentbestr=max(final3)

        for i in range(len(final3)):
            if final3[i] == currentbestr:
                currentbestl = lever3[i]
        print('Current Best Lever:', currentbestl)
        print('Current Best Reward:', currentbestr)

    if lol > epsilon:
        print('Value greater than epsilon')
        lever3.append(currentbestl)
        final3.append(currentbestr)
        total3 += currentbestr
        avg = total3 / (i + 1)
        average3.append(avg)

plt.plot(iteration3, average3)
plt.xlabel('Tth Iteration')
plt.ylabel('Average')
plt.title('Average Reward vs Tth iteration')
plt.show()


#  Scenario 4
#  epsilon decrease
epsilon=0.4
print(epsilon)
lever4=[]
average4=[]
iteration4=[]
final4=[]
total4=0
currentbestl=0
currentbestr=0
for i in range(pulls):
    epsilon=epsilon/(1+i+1)
    print('epsilon', epsilon)
    iteration4.append(i + 1)
    lol=random.uniform(0,1)
    if lol<=epsilon:
        print('Value smaller than epsilon')
        iteration.append(i + 1)
        l=random.randint(1,bandit)
        print(f'Lever pulled: {l}')
        lever4.append(l)
        final4.append(rewards[l-1][i])
        total4+=rewards[l-1][i]
        avg=total4/(i+1)
        average4.append(avg)
        currentbestr=max(final4)

        for i in range(len(final4)):
            if final4[i] == currentbestr:
                currentbestl = lever4[i]
        print('Current Best Lever:', currentbestl)
        print('Current Best Reward:', currentbestr)

    if lol > epsilon:
        print('Value greater than epsilon')
        lever4.append(currentbestl)
        final4.append(currentbestr)
        total4 += currentbestr
        avg = total4 / (i + 1)
        average4.append(avg)
print(currentbestl, currentbestr)
plt.plot(iteration4, average4)
plt.xlabel('Tth Iteration')
plt.ylabel('Average')
plt.title('Average Reward vs Tth iteration')
plt.show()


