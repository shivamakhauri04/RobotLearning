import math
import numpy as np
from collections import defaultdict
import sys
from matplotlib import pyplot as plt

# find the minimum value for the hash map
def nearest_val(map_value,map_associtaionWeights):
	t=[abs(map_value-x) for x in map_associtaionWeights.keys()]
	return t.index(min(t))


# Generate dataset
def dataset():
	data=np.linspace(0,0.5*np.pi,100)

	#generate training data and test data
	input_train=[data[x] for x in range(0,len(data)) if x%3!=0]
	input_test=[data[y] for y in range(0,len(data)) if y%3 ==0]
	
	# Labels for the dataset
	target_train=np.cos(input_train)
	target_test=np.cos(input_test)
	print ()

	return input_train,input_test,target_train,target_test,data


# Create lookup table for the association space and weight space
def generate_hashMap(input_train,target_train,map_associtaionWeights,weight_size,association_weight):
	#map from input values --> 
	input_levels=set(input_train)
	for i in input_levels:
		# convert float data to integer
		i=round(float(i*association_weight),4)
		# one added for darta continuity
		for j in range(int(i),int(i)+weight_size+1):
			map_associtaionWeights[i][j]=0

	lookup_table=dict()
	count=0
	for i in input_levels:
		#convert numpy.float to int..
		i=round(float(i*association_weight),4)
		# map between the target data and lookup data
		lookup_table[i]=target_train[count] 
		count+=1


def train(input_train,association_weight,map_associtaionWeights,target_train,weight_size):
	epoch=1
	total_epochs= 100
	error_list=list()
	num=0
	while epoch <=total_epochs:
		error_sum=0
		count=0
		for i in input_train:
			map_value=round(float(i*association_weight),4)
			# adjust weights
			if map_value not in map_associtaionWeights.keys():
				print("input not in the map")
				num+=1
				continue
			if map_value  in map_associtaionWeights.keys():		
				weight_sum=0
				for v in map_associtaionWeights[map_value].values():
					weight_sum=weight_sum+v
				#calculate error
				error=target_train[count]-weight_sum
				# calculate error delta
				error_delta=error/weight_size
				error_sum+=error

				if error >= 0:
					# for the hash map
					for index,k in enumerate(map_associtaionWeights[map_value].keys()):
						if index==0:  
							# corner index of the weight matrix
							map_associtaionWeights[map_value][k]+=0.5*error_delta
						elif index==weight_size:
    						# the other corner of the weight matrix
							map_associtaionWeights[map_value][k]+=0.5*error_delta
						else:
							map_associtaionWeights[map_value][k]+=error_delta
			count+=1
		error_sum =error_sum/len(target_train)
		error_list.append(error_sum)		
		print("error delta %f, error sum for  %d epoch is %f "%(error_delta,epoch,error_sum))
		epoch+=1


# create test pipeline
def test(input_test,association_weight,map_associtaionWeights,target_test,weight_size):
	test_error=0
	predicted_list=list()
	# iterate over the test set
	for index,t in enumerate(input_test):
		map_value=round(float(t*association_weight),4)
		#initialize the weight sum to zero .
		if map_value not in map_associtaionWeights.keys():
			k=nearest_val(map_value,map_associtaionWeights)
			new_map_value=list(map_associtaionWeights.keys())[k]
			map_value=new_map_value
		
		weight_sum=0
		# sum of the weights
		for l in map_associtaionWeights[map_value].values():
			weight_sum=weight_sum+l
		test_error+=target_test[index]-weight_sum
		predicted_list.append(weight_sum)
	print("Average test error for weight_size value %d is---> %.6E"%(weight_size,test_error/len(input_test)))
	return predicted_list


#plot results
def plot(predicted_list,input_test,target_test,data):
    # plot the images
	acc = []
	for i,_ in enumerate(predicted_list):
    		acc.append(input_test[i]-predicted_list[i])
	plt.plot(input_test,predicted_list,color="blue",linewidth="2.5",linestyle="-")
	plt.plot(input_test,target_test,'r*',markersize=3.5)
	#plt.plot(input_test,acc,color="green",linewidth="2.5",linestyle="-")
	plt.title('Plot of Original and Apprximated Function using Continuous CMAC')
	plt.legend(['Approximated Function','Original Function','Accuracy'])
	plt.grid()
	plt.show()


def plotAccuracy(predicted_list,input_test,target_test,data):
    # plot the images
	time = []
	for i in range(34,0,-1):
    		time.append(i)
	acc = []
	for i,_ in enumerate(predicted_list):
    		acc.append(predicted_list[i]-input_test[i])
	plt.plot(time,acc,color="green",linewidth="2.5",linestyle="-")
	plt.title('Accuracy vs Convergence time')
	plt.legend(['Accuracy','Convergence Time'])
	plt.grid()
	plt.show()

def plotIn(predicted_list,input_test,target_test,data):
    # plot the images
	acc = []
	for i,_ in enumerate(predicted_list):
    		acc.append(input_test[i]-predicted_list[i])
	plt.plot(input_test,acc,color="green",linewidth="2.5",linestyle="-")
	plt.title('Accuracy vs Original Function using Continuous CMAC')
	plt.legend(['Original Function','Accuracy'])
	plt.grid()
	plt.show()


def main():
		map_associtaionWeights=defaultdict(dict)
		input_train,input_test,target_train,target_test,data = dataset()
		association_weight=35
		weight_size=35
		generate_hashMap(input_train,target_train,map_associtaionWeights,weight_size,association_weight)
		train(input_train,association_weight,map_associtaionWeights,target_train,weight_size)
		predicted_list = test(input_test,association_weight,map_associtaionWeights,target_test,weight_size)
		plot(predicted_list,input_test,target_test,data)
		plotAccuracy(predicted_list,input_test,target_test,data)
		plotIn(predicted_list,input_test,target_test,data)

if __name__ == "__main__":
    main()
