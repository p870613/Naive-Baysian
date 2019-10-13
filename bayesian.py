#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def gen_train_test_data():
    test_ratio = .3 
    df = pandas.read_csv('yelp_labelled.txt', sep='\t', header=None, encoding='utf-8')
    count_vect = CountVectorizer()
    X = count_vect.fit_transform(df[0])
    y = df[1].tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)
    return X_train, X_test, y_train, y_test


def multinomial_nb(X_train, X_test, y_train, y_test):
    # TODO: fill this function
    # train by X_train and y_train
    # report the predicting accuracy for both the training and the test data
    x_train =  X_train.toarray()
    x_test =  X_test.toarray() 
    x_train_1 = []
    x_train_0 = []
    for i in range (len(y_train)):
        if(y_train[i] == 0):
            x_train_0.append(x_train[i])
        else:
            x_train_1.append(x_train[i])

    prior_0 = float(len(x_train_0) / len(y_train))
    prior_1 = float(len(x_train_1) / len(y_train))
    
    
    total_feature_0 = []
    total_feature_sum_0 = 0
    likelihood_0 = []
    for i in range (len(x_train_0[0])):
        sum = 0
        for j in range(len(x_train_0)):
            sum = sum + x_train_0[j][i]
        total_feature_0.append(sum)
        total_feature_sum_0 = total_feature_sum_0 + sum
        
    for item in total_feature_0:
        likelihood_0.append(float((item +1) / (total_feature_sum_0 + len(total_feature_0))))
    
       
    total_feature_1 = []
    total_feature_sum_1 = 0
    likelihood_1 = []
    for i in range (len(x_train_1[0])):
        sum = 0
        for j in range(len(x_train_1)):
            sum = sum + x_train_1[j][i]
        total_feature_1.append(sum)
        total_feature_sum_1 = total_feature_sum_1 + sum
    
    for item in total_feature_1:
        likelihood_1.append(float((item +1) / (total_feature_sum_1 + len(total_feature_1))))

    #train_data_test
    count = 0
    for (item, ans) in zip(x_train, y_train):
        p0 = 1.0
        p1 = 1.0
        
        #0
        p0 = p0 * prior_0
        for i in range(len(item)):
            p0 = p0 * math.pow(likelihood_0[i], item[i])  
        
        #1
        p1 = p1 * prior_1
        for i in range(len(item)):
            p1 = p1 * math.pow(likelihood_1[i], item[i])
            
        sol = 0
        if(p1 > p0):
            sol = 1
        else:
            sol = 0

        if(sol == ans):
            count = count + 1
    print("multinomial_nb_train : ", count / len(x_train))

    #test_data_test
    count = 0
    for (item, ans) in zip(x_test, y_test):
        p0 = 1.0
        p1 = 1.0
        
        #0
        p0 = p0 * prior_0
        for i in range(len(item)):
            p0 = p0 * math.pow(likelihood_0[i], item[i])  
        
        #1
        p1 = p1 * prior_1
        for i in range(len(item)):
            p1 = p1 * math.pow(likelihood_1[i], item[i])
            
        sol = 0
        if(p1 > p0):
            sol = 1
        else:
            sol = 0

        if(sol == ans):
            count = count + 1
    print("multinomial_nb_test  : ",count / len(x_test))

def bernoulli_nb(X_train, X_test, y_train, y_test):
    # TODO: fill this function
    # train by X_train and y_train
    # report the predicting accuracy for both the training and the test data 
    x_train =  X_train.toarray()
    x_test =  X_test.toarray() 
    x_train_1 = []
    x_train_0 = []
    for i in range (len(y_train)):
        if(y_train[i] == 0):
            x_train_0.append(x_train[i])
        else:
            x_train_1.append(x_train[i])

    prior_0 = float(len(x_train_0) / len(y_train))
    prior_1 = float(len(x_train_1) / len(y_train))
    
    
    total_feature_0 = []
    total_feature_sum_0 = 0
    likelihood_0 = []
    for i in range (len(x_train_0[0])):
        sum = 0
        for j in range(len(x_train_0)):
            if(x_train_0[j][i] != 0):
                sum = sum + 1
        total_feature_0.append(sum)
        
    for item in total_feature_0:
        likelihood_0.append(float((item +1) / (len(x_train_0) + len(total_feature_0))))
    
       
    total_feature_1 = []
    total_feature_sum_1 = 0
    likelihood_1 = []
    for i in range (len(x_train_1[0])):
        sum = 0
        for j in range(len(x_train_1)):
            if(x_train_1[j][i] != 0):
                sum = sum + 1
        total_feature_1.append(sum)
    
    for item in total_feature_1:
        likelihood_1.append(float((item +1) / (len(x_train_1) + len(total_feature_1))))

    #train_data_test
    count = 0
    for (item, ans) in zip(x_train, y_train):
        p0 = 1.0
        p1 = 1.0
        
        #0
        p0 = p0 * prior_0
        for i in range(len(item)):
           
            if(item[i]>=1):
                p0 = p0 * math.pow(likelihood_0[i], 1)  
        
        #1
        p1 = p1 * prior_1
        for i in range(len(item)):
            if(item[i]>=1):
                p1 = p1 * math.pow(likelihood_1[i], 1)
            
        sol = 0
        if(p1 > p0):
            sol = 1
        else:
            sol = 0

        if(sol == ans):
            count = count + 1
    print("bernoulli_nb_train : ", count / len(x_train))

    #test_data_test
    count = 0
    for (item, ans) in zip(x_test, y_test):
        p0 = 1.0
        p1 = 1.0
        
        #0
        p0 = p0 * prior_0
        for i in range(len(item)):
            if(item[i]>=1):
                p0 = p0 * math.pow(likelihood_0[i], 1)  
        
        #1
        p1 = p1 * prior_1
        for i in range(len(item)):
            if(item[i]>=1):
                p1 = p1 * math.pow(likelihood_1[i], 1)
            
        sol = 0
        if(p1 > p0):
            sol = 1
        else:
            sol = 0

        if(sol == ans):
            count = count + 1
    print("bernoulli_nb_test  : ",count / len(x_test))


def main(argv):
    X_train, X_test, y_train, y_test = gen_train_test_data()
    multinomial_nb(X_train, X_test, y_train, y_test)
    bernoulli_nb(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main(sys.argv)


