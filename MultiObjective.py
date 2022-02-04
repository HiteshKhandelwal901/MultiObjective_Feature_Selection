from importlib.resources import path
import numpy as np
from multiprocessing import Pool, cpu_count
import pandas as pd
from scipy import rand
from sklearn.neighbors import KNeighborsClassifier
import sklearn
from collections import defaultdict
from sklearn.metrics import hamming_loss
from skmultilearn.adapt import MLkNN
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from operator import itemgetter
from collections import defaultdict
import pandas as pd
import os
import random
import math
from sklearn.model_selection import train_test_split

NUM_EXPERIMENTS = 1
NUM_SOLUTIONS = 50
NUM_ITERATIONS = 5
DIMENSION = 9
K = 10
data_path = 'Data'
cache = defaultdict()




REPORT_PATH = "./Reports"
NAME_OF_FILE = 'take2_batch4_report_medical.xlsx'

def univariate_feature_elimination(X,y, k):
    """
    Function for univariate feature elimination based on CH2 critrian. 
    """
    sk = SelectKBest(score_func=chi2, k=4)
    X_new = sk.fit_transform(X,y)
    scores = sk.scores_
    columns_to_drop = get_least_columns(X, scores,k)
    X = X.drop(columns = columns_to_drop)
    return X

def get_least_columns(X, scores,k):
    """
    Helper function for Univariate feature elimination. Returns the names of the columns of dataframe X
    which ahave the least ch2 scores 
    """
    columns_to_remove = []
    names = defaultdict()
    for index,col in enumerate(X.columns):
        names[col] = scores[index]
    remove_dic = dict(sorted(names.items(), key = itemgetter(1))[:k])
    for keys in remove_dic:
        columns_to_remove.append(keys)
    return columns_to_remove

def create_report(metric):
    report_df = pd.DataFrame(metric)
    print("Report:", report_df)
    if not os.path.exists(REPORT_PATH):
        print("Creating Report directory", REPORT_PATH)
        os.mkdir(REPORT_PATH)
    report_df.to_excel(os.path.join(REPORT_PATH, NAME_OF_FILE))



class Solution:
    def __init__(self, name):
        self.name = name
        self.loss = 1
        self.pos = [self.random_generator_binary() for i in range(DIMENSION)]
        self.active_features = []
        self.num_attr = 0
        self.rank = -1
        self.pareto_solution = False
        self.clf = KNeighborsClassifier(n_neighbors=10)
        self.crowding_distance = 0
    
    def reset(self):
        self.rank = -1
        self.pareto_solution = False

    def random_generator_binary(self):
        num = random.uniform(0, 1)
        if num > 0.5:
            return 1
        else:
            return 0

    def feature_index(self):
        index_ofactive_features = []
        for index, dim in enumerate(self.pos):
            if dim ==1:
                index_ofactive_features.append(index)
        return index_ofactive_features

    
    def get_hamming_loss(self, X,y, metric = False):
        print("inside hamming loss")
        self.classifier_fit(X,y)
        y_pred = self.classifier_predict(X)
        loss = hamming_loss(y_pred, y)
        return loss

    def classifier_fit(self, X,y):
        print("inside classifier fit")
        x = np.array(X)
        y = np.array(y)
        self.clf.fit(x,y )


    def classifier_predict(self,X):
        return self.clf.predict(np.array(X))
        #return self.clf.predict(np.array(X)).toarray()

    def evaluate(self, X, Y) -> None:
        #print("inside evaluate")
        size = X.shape[1]
        #print("pos = ", self.pos)
        self.active_features = self.feature_index()
        #print("length of active features = ", len(self.active_features))
        #print("active features = ", self.active_features)
        #print("feature index {}".format(self.feature_index))
        self.num_attr = len(self.active_features)
        X = X.iloc[:, self.active_features]
        #print(X)
        #print("first 2 : ", X.iloc [0:3, :])
        #print("X.shape = ", X.shape)
        #print("X = ", X)
        index_sum = sum(self.active_features)
        #if subset is not empty
        #print("X shape = ", X.shape)
        if X.shape[1] > 0:
            #print("True, shape > 1")
            #if the subset is alreasy seen before, get the score and loss from cache
            if index_sum in cache:
                #print("found in cache")
                ham_loss = cache[index_sum]
                #print("ham_loss returned = ", ham_loss)
                self.loss = ham_loss
            else:
                #print("not in cache")
                ham_loss = self.get_hamming_loss(X,Y)
                #print("ham_loss returned = ", ham_loss)
                self.loss = ham_loss
                #print("printing ham loss after setting ", self.loss)
                cache[index_sum] = ham_loss
        else:
            #print("False shape < 1")
            self.loss = 1
        return None
    


    def move(self, source, target) ->None:
        #step1 : loop through the dimensions to set position
        for i in range(len(self.pos)):
            #if ith position of both target and source is same then assign any
            if source.pos[i] == target.pos[i]:
                self.pos[i] = target.pos[i]
            #if prob > 0.5 set pos as target's pos else set pos as source's pos
            else:
                rand_num = random.random()
                #print("random = ", random)
                if rand_num < 0.5:
                    self.pos[i] = target.pos[i]
                else:
                    self.pos[i] = source.pos[i]
        return None

    def __str__(self) -> str:
        return "NAME : " + self.name +  " HAM LOSS :" + str(self.loss) + " NUM ATTR :" +  str(self.num_attr) +  " RANK :" + str(self.rank) + " IS PARETO : " + str(self.pareto_solution) + " CROWDING DISTANCE : " + str(self.crowding_distance) + " ACTIVE ERROR : " + str(self.active_features)


def set_crowding_distance(pop, sol):
    #print("SOL = ", sol)
    #print("Inside crowding distance")
    #Step1 : Sort all the solutions with key as attr
    #print("----sorting---")
    sorted_pop = sorted(pop, key = lambda x: x.num_attr)
    #print("After sorting")
    #for i in range(len(pop)):
        #print(pop[i])
    #print("looping to find the index")
    #step3 : get the index of neighbours 
    for i in range(len(sorted_pop)):
        if sorted_pop[i].name == sol.name:
            index = i
            break
    #print("index = ", index)

    if index == len(pop)-1:
        b = math.sqrt(math.pow((sorted_pop[index].num_attr - sorted_pop[index-1].num_attr), 2))
        #print("doing sorted_pop[i+1].loss {} - sorted_pop[i-1].loss {}".format(sorted_pop[index+1].loss,sorted_pop[index-1].loss))
        l = math.sqrt(math.pow((sorted_pop[index].loss - sorted_pop[index-1].loss), 2))
        sol.crowding_distance = round(2*(b+l),3)
        if round(2*(b+l), 3) == 0:
            print("1 crowding distance zero")
        return None
    
    #case 1 : first solution
    if index == 0:
        b = math.sqrt(math.pow((sorted_pop[index].num_attr - sorted_pop[index+1].num_attr), 2))
        l = math.sqrt(math.pow((sorted_pop[index].loss - sorted_pop[index+1].loss), 2))
        sol.crowding_distance = round(2*(b+l), 3)

        if round(2*(b+l), 3) == 0:
            print("2 crowding distance zero")
        return None
        


    #print("index = ", index)
    #print(sorted_pop[0])
    #print(sorted_pop[1])
    #print(sorted_pop[2])
    #print("left neighbour = ", index -1 ,  sorted_pop[index-1])
    
    #print("check if correct left neg = {}".format(sorted_pop[0]))
    #print("Solution = ", pop[i])
    #print("right neighbour  = ", index + 1 , sorted_pop[index+1])
    #print("check if correct right neg = {}".format(sorted_pop[2]))
    #step4 : get the lengths of the sides based on euclidian distance
    #print("doing sorted_pop[i+1].num_attr {} - sorted_pop[i-1].num_attr {}".format(sorted_pop[index+1].num_attr,sorted_pop[index-1].num_attr))
    b = math.sqrt(math.pow((sorted_pop[index+1].num_attr - sorted_pop[index-1].num_attr), 2))
    #print("doing sorted_pop[i+1].loss {} - sorted_pop[i-1].loss {}".format(sorted_pop[index+1].loss,sorted_pop[index-1].loss))
    l = math.sqrt(math.pow((sorted_pop[index+1].loss - sorted_pop[index-1].loss), 2))
    #print("B = ", b)
    #print("L = ", l)

    perimeter = round(2*(b + l), 3)
    #print("perimeter = ", perimeter)
    if round(2*(b+l), 3) == 0:
        print("crowding distance zero")

    sol.crowding_distance = perimeter
    
            
    return None



def read_data():
    """
    Function to read data from the path, do preprocessing and return X and Y
    """



    """
    #print("inside read data")
    final_path = os.path.join(data_path, 'Iris.csv')
    print(final_path, type(final_path))
    data  = pd.read_csv(final_path)
    #print("read data")
    Y = data[['Species']]
    X = data.drop(columns = ['Species', 'Id'])
    

    scaled_features = sklearn.preprocessing.MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(scaled_features, index= X.index, columns= X.columns)
    #uncomment to run with chi^2
    #X = univariate_feature_elimination(X,Y,15)
    """

    
    final_path = os.path.join(data_path, 'breast-cancer.csv')
    data  = pd.read_csv(final_path)
    Y = data.iloc[:, -1:]
    X = data.iloc[:, 1: -1]
    #scaled_features = sklearn.preprocessing.MinMaxScaler().fit_transform(X.values)
    #X = pd.DataFrame(scaled_features, index= X.index, columns= X.columns)


    return X,Y

def Rank(pop : list) -> None:
    """
    Perfoms ranking of all solutions. Loops through each solution and compares it with every other soltuion
    to check if the solution is dominated. if a solution is dominated then its pareto_solution attr will be 
    set to False. At the end, all the solutions with pareto domination = True will be rank 1 Solution. Repeat 
    process and asign the subsequent rank to all other solutions
    """
    #loop until remaining is zero solutions
    remaining = 10000000
    rank = 0
    iter = 0
    #print("Inside Rank function")
    while(remaining >1):
        remaining = 0
        rank = rank +1
        #print("Rank or iter:", rank)
        #print("looping through each sol and checking for pareto condition")
        #Step1 : Loop through each solution ( outter loop)
        for i in range(len(pop)):
            #print("i = ", i)
            #step2 : Current solution  = soltuions[loop_index]
            curr_sol = pop[i]
            #print("Curr_sol = {}".format(curr_sol))
            #check if the curr solution has not yet been assigned a rank, default rank  = -1
            #print("checking if curr sol already has a rank set")
            if curr_sol.rank != -1 :
                pass
                #print("Yes the curr sol has the rank set and rank  = {}", curr_sol.rank)
                #print("Breaking")
                #break
            else:
                #print("not asssigned")
                #step3 : Loop through every other solution (Inner loop)
                for j in range(len(pop)):
                    #print("J = ", j)
                    #print("checking if J  = I basically condition for same solution")
                    if j!= i:
                        #print("False")
                        sol = pop[j]
                        #print("SOL : ", sol)
                        #step3.1 : check if the sol has not yet been assigned a rank 
                        #print("checking if rank has not been assigned") 
                        if sol.rank == -1:
                            #print("True not assigned")
                            #print("Checking for dominace")
                            #print("sol.ham_loss = {} sol.num_attr = {} currsol.ham_loss ={} cursol.attr = {}".format(sol.loss, sol.num_attr, curr_sol.loss, curr_sol.num_attr))
                            if sol.loss < curr_sol.loss and sol.num_attr < curr_sol.num_attr:
                                #step3.1.1: Then Pareto_solution = False
                                #print("True, {} dominated by {}".format(curr_sol, sol))
                                curr_sol.pareto_solution = False
                                #print("breaking")
                                break
                            else:
                                #step3.1.2.1: Pareto_solution = True
                                #print("Not yet dominated")
                                curr_sol.pareto_solution = True
                        else :
                            #print("Rank assigned ", sol.rank)
                            pass
                    else:
                        #print("True Same")
                        pass
            #print("At the end of inner loop, curr_sol.paretosolution = ", curr_sol)
        if rank > 20:
            #print("rank = ", rank)
            #print("returning because of infinite loop")
            return -10
        
        
        
        #print("DONE WITH CHECKING FOR DOMINATION")
        #step4 : Loop through each solution and assign rank to them
        #print("Assing the Rank")
        for i in range(len(pop)):
            #print("checking if {} is pareto".format(pop[i]))
            if pop[i].pareto_solution == True and pop[i].rank == -1:
                #print("True")
                pop[i].rank = rank
            else:
                #print("False")
                remaining = remaining + 1
                #print("Reamining  = ", remaining)
        #Edge case when only one solution is remaining, assign it the last rank
        #print("At the end of loop, remaining = ", remaining)
        #print("checking edge case if remaining is just 1")
        if remaining == 1:
            #print("True")
            #print("looping and getting that one solution")
            for i in range(len(pop)):
                if pop[i].pareto_solution == False:
                    #print("{} has pareto false".format(pop[i]))
                    pop[i].rank = rank+1
                    break
        else:
            #print("False")
            pass
        iter = iter+1

        #print("AT the end of iteration  ", iter)
        #for i in range(len(pop)):
        #    print(pop[i])
    return 1


def print_solution(pop):
    #print("inside print soltuion")
    #print("length of pop = ", len(pop))
    for i in range(len(pop)):
        #print("i = ", i)
        print(pop[i])
        #print(pop[i].loss)
        #print(pop[i].rank)




def MultiObjective_Optimization(X:pd.DataFrame, Y:pd.DataFrame) -> None :
    #print("X = ")
    #print(X)
    #print(Y)
    #print("INSIDE MULTIOBJECTIVE FUNCTION")
    #Step1 : Initalize soltuions
    #print("INITIALIZING SOLUTIONS")
    pop = []
    for i in range(0, NUM_SOLUTIONS):
        pop.append(Solution(str(i)))
        #print(pop[i])
    #print("-----After Initalization------")
    #print_solution(pop)
    #print("------------STEP 1 DONE ---------")
    it = 0
    while it< NUM_ITERATIONS:
        print("ITER  || ", it)
        #Step2 : Evaluate 
        #print("evaluating solutions")
        """
        pop.append(Solution(str(0)))
        pop.append(Solution(str(1)))
        pop.append(Solution(str(2)))
        pop.append(Solution(str(3)))
        pop.append(Solution(str(4)))

        pop[0].loss = 0.1739
        pop[0].num_attr = 4

        pop[1].loss = 0.1749
        pop[1].num_attr = 3

        pop[2].loss = 0.1743
        pop[2].num_attr = 4

        pop[3].loss = 0.1739
        pop[3].num_attr = 3

        pop[4].loss = 0.1743
        pop[4].num_attr = 4
        """
        
        #print("EVALUATING\n\n")
        #print("length of pop = ", len(pop))
        for i in range(len(pop)):
            #print("evaluations  {}".format(pop[i].name))
            pop[i].evaluate(X,Y)
            #print(pop[i])
        
        #print("-----After Evaluting---\n\n")
        #print_solution(pop)
        #print("----STEP2 DONE--------\n\n")
        
        
        #print("updating crwoding distance\n\n")
        for i in range(len(pop)):
            #print("I  = ", i)
            set_crowding_distance(pop, pop[i])
            #print("DONE")
        
        #print("------After Updating Crowding Distance -----\n\n")
        #print_solution(pop)
        #print("------STEP 3 DONE ----------\n\n")


        #step3:  Rank all the solutions



        #print("Ranking solutions")
        flag = Rank(pop)
        print_solution(pop)
        break
        #print("-----RANKING DONE-------\n\n")
        #print("-----STEP 4 DONE----------\n\n")
        
        new_pop = []
        wh_it = 0

        #step4 : Loop till you get NUM_SOLUTIONS OF new soltuions
        while(len(new_pop)!= len(pop)):
                #Step4a : Randomly select two solutions
                #print("Inside while loop with iter = ", wh_it) 
                wh_it = wh_it + 1
                r1 = random.randint(0, len(pop)-1)
                r2 = random.randint(0, len(pop)-1)
                while r1!=r2:
                    r2 = random.randint(0, len(pop)-1)

                s1 = pop[r1]
                s2 = pop[r2]
                #print("RANDOM S1 : {} ".format(s1))
                #print("RANDOM S2 : {} ".format(s2))
                
                #print("CHECKING IF RANKS ARE SAME")
                #step4b : If both the solutions are of same rank 
                if s1.rank == s2.rank:
                        #print("TRUE, SAME RANK")
                        #print("CHECKING IF s1 crowding dist = {} > s2 crowding dist = {}".format(s1.crowding_distance,s2.crowding_distance ))
                        #step4b.1 : let s2 be the source(lower cd distance), s1 be the target(higher cd distance), s be the new solution
                        if s1.crowding_distance > s2.crowding_distance:
                            #print("TRUE")
                            s = Solution("S" + str(it+1))
                            s.move(s2, s1)
                        else:
                            #print("FALSE")
                        #step4b.2 : else let S1 be source, s2 be target and s be the new solution
                            s = Solution("S" + str(wh_it+1))
                            s.move(s1, s2)
                #step4c : else
                else:  
                        #print("FALSE, DIFFERENT RANKS")
                        #step4c.1 : Intialize new solution = S
                        s = Solution("S" + str(wh_it+1))

                        #step4c.2 : Let the one with higher rank be s1 and the other be s2'
                        #print("CHECKING IF s1 rank = {} < s2 rank = {}".format(s1.rank,s2.rank))
                        if s1.rank < s2.rank:
                            #print("TRUE s1 < s2")
                            s.move(s2, s1)
                        else:
                            #print("False, s1 > s2")
                            s.move(s1,s2)

                #step4c.4 :  Add S to the new_soltuions list
                #print("NEW SOLUTION  = {}".format(s))
                #print("Appedning it to new pop")
                new_pop.append(s)


        #step5 : new list = Concatinate new and old solution
        #print("DONE CREATING NEW SOLTUIONS\n\n")
        #print_solution(new_pop)
        #print("Concating new and old pop\n\n")
        concat_pop = pop + new_pop
        #print("Printing concat pop\n\n")
        #print_solution(concat_pop)

        #step6 : Reset LOSS, RANK, PARETO of all solutions in the new list
        #print("Restting loss and pareto of concat pop \n\n")
        for i in range(len(concat_pop)):
            concat_pop[i].reset()
            #step7 : Evaluate loss for each of the new list
            concat_pop[i].evaluate(X,Y)
        
        #print("DONE, Printing concat pop\n\n")
        #print_solution(concat_pop)

                
        #print("RERANKING\n\n")
        #step8 : Rank all the soltuions in the new list
        Rank(concat_pop)
        #print("concat pop after re ranking = \n\n")
        #print_solution(concat_pop)
        #print("----DONE RERANKING----\n\n")
        
        #print("REMOVING THE BOTTOM K = ", K)
        #step9 : Remove the bottom K solutions
        #print("First sorting \n\n")
        sorted_pop =  sorted(concat_pop, key = lambda x: x.rank)
        #print("Done sorting\n\n")
        #print("printing sorted pop \n\n")
        #print_solution(sorted_pop) 
        #print("DONE WITH REMOVING BOTTOM K \n\n")
        #print("AT THE END, FINAL POP \n\n") 
        pop = sorted_pop[:K]
        #print_solution(pop)
     
        #step10: Repeat until convergence | NUM_OF_ITERATIONS
        #print("----END BREKAING----")
        print_solution(pop)
        it = it+1
    return pop

def single_run(experiment_id: int) -> dict[str, int]:
    """
    Perfoms the single run of the Multiobjective algorithm. Used in Pool's args to parallelize the run
    """
    print("Running Experiment", experiment_id)
    X, Y = read_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
    print(X)
    print("INFO : \n")
    print("X_train", X_train.shape)
    print("X_test", X_test.shape)
    print("Y_train", Y_train.shape)
    print("Y_test", Y_test.shape)
    print("\n\n ------Running MultiObjective Code -----")
    
    pop = MultiObjective_Optimization(X_train, Y_train)

    print("printing returned pop")
    print_solution(pop)

    return None


def run_experiments(num_experiments: int) -> None:
    """
    Perform Black Hole Algorithms multiple item with different random seed each time
    Processess the runs parallely depending upon the num of processes/experiment
    """

    experiment_list = list(range(num_experiments))
    with Pool(processes=min(num_experiments, 8, cpu_count())) as pool:
        res = pool.map(single_run, experiment_list)
        #create_report(res)   

def main():
    run_experiments(NUM_EXPERIMENTS)

if __name__ == "__main__":
    main()