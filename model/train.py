import pickle
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, load_linnerud, load_diabetes
from sklearn import tree
import time,random

dataset = []
# MAX_DEPTH=10
MAX_DEPTH=[None,None,None,10,12,15,18,20]

names = ["wine", "linnerud", "cancer",
            "digits-10", "digits-12", "digits-15","diabets-18"]

dataset.append(load_wine())
dataset.append(load_linnerud())
dataset.append(load_breast_cancer())
dataset.append(load_digits())
dataset.append(load_digits())
dataset.append(load_digits())
dataset.append(load_diabetes())
# dataset.append(load_diabetes())


for i,d in enumerate(dataset):
    X, y = d.data,d.target
    # clf = tree.DecisionTreeClassifier()
    if MAX_DEPTH[i] is None:
        clf = tree.DecisionTreeClassifier(random_state=2)
    else:
        clf = tree.DecisionTreeClassifier(random_state=2, max_depth=MAX_DEPTH[i])
    clf = clf.fit(X, y)
    # tree.export_graphviz(clf,names[i]+".dot")
    # Get the number of unique classes
    num_classes = clf.n_classes_
    print("#Classifications: ", num_classes)
    print(names[i], " #Instances ", len(X))
    print(names[i], " #Feature ", len(X[0]))
    print(names[i], " Depth ", clf.get_depth())
    # print(clf.tree_.feature)
    usedFeatures={}
    cnt=0
    decisionNodes=0
    for f in clf.tree_.feature:
        if f != -2:
            if str(f) not in usedFeatures.keys():
                usedFeatures[str(f)] = cnt
                cnt+=1
            decisionNodes+=1
    print("# Used features is: ", len(usedFeatures.keys()))
    print(names[i], " #Nodes ", decisionNodes)
    # print(usedFeatures)
    modelName = "./"+names[i]+".model"
    with open(modelName, 'wb') as fid:
        pickle.dump(clf, fid)

    #Plaintext evaluation
    # repeatTimes = 1000
    repeatTimes = 1
    # start_T = time.time()
    for j in range(repeatTimes):
        # tVec = random.choice(X)
        tVec = X[0]
        # ret = clf.apply([tVec])
        start = time.time()
        ret = clf.predict([tVec])
        # print("required time is: ",time.time() - start )
        # print("Final evaluation results is: ", ret)
    # end_T = time.time()
    # print("total time used is: ",(end_T-start_T),"\n\n")

# print(len(X))
# print(X[149])
# tree.plot_tree(clf)


# save the classifier
# with open('my_dumped_classifier.pkl', 'wb') as fid:
#     pickle.dumps(gnb, fid)

# # load it again
# with open('my_dumped_classifier.pkl', 'rb') as fid:
#     gnb_loaded = cPickle.load(fid)
