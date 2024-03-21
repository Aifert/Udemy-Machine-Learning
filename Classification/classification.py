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

def is_leap(year):
    if(year % 4 == 0):
        if(year % 100 == 0):
            if(year % 400 == 0):
                return True
            return False
        return True
    return False

def print_leap(start_year, end_year):
    for year in range(start_year, end_year + 1):
        if(is_leap(year)):
            print(year)
        

print_leap(2000, 2020)