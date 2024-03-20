# from logistic_regression import logistic
# from decision_tree_classification import decision_tree
# from k_nearest_neighbors import k_nearest
# from kernel_svm import kernel_svm
# from naive_bayes import naive
# from random_forest_classification import random_forest
# from support_vector_machine import support


# logistic()
# decision_tree()
# k_nearest()
# kernel_svm()
# naive()
# random_forest()
# support()

x = int(input("input something: "))
first_number , second_number = 0, 1
result = 0
new_result = 0
for i in range(0, x-1):
    result = first_number + second_number
    new_result += result
    #Odd
    if(i % 2 != 0):
        first_number = result
    else:
        second_number = result
print(new_result)