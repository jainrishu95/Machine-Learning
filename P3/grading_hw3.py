# code check
import time
import os


def assign_score_to_accuracy(proposal, ref, base_score=1., cutoff=-1e-1):
    """
    Assign a score comparing a student's output to our reference output.
    Input:
        `proposal`: student's output accuracy. (float)
        `ref`: the accuracy from our reference code. (float)
        `base_score`: the score we assign to a subproblem. (float; default=1.0)
        `cutoff`: the lowest acceptable accuracy. (float; MUST BE NEGATIVE)
    """
    score = 0
    if proposal < 1. + 1e-6:  # to avoid numerical error; accuracy should < 1.0
        diff = max(-1, min(0., proposal - ref))
        if diff > cutoff:
            score = base_score * (1 - diff / cutoff)
    return score


# boosting.py
def check_boosting():
    score = 0
    try:
        os.system("python3 boosting_check.py > boosting_check_output.txt")  # run students' code
        with open("boosting_check_output.txt", "r") as f:
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            if f.readline().strip() == "Betas are correct":
                score += 2
            f.readline()
            ada = f.readline()
            ada = ada.split(":")[1].strip().split(" ")
            score += assign_score_to_accuracy(float(ada[0]), 0.75, 2)
            score += assign_score_to_accuracy(float(ada[1]), 1.0, 2)
    except:
        pass
    os.system("rm boosting_check_output.txt")
    try:
        os.system("bash boosting.sh > boosting_check_output.txt")
        with open("boosting_check_output.txt", "r") as f:
            ada = f.readline()
            ada = ada.split(":")[1].strip().replace("[", "").replace("]", "").split(",")
            ada = [float(x.strip()) for x in ada]
            score += assign_score_to_accuracy(ada[0], 1.0, 3.0)
            score += assign_score_to_accuracy(ada[1], 0.94999999999999996, 3.0)
            score += assign_score_to_accuracy(ada[2], 0.94999999999999996, 3.0)
    except:
        pass
    os.system("rm boosting_check_output.txt")
    return score


def check_pegasos():
    score = 0.0
    try:
        import pegasos
        start_time = time.time()
        test_acc, train_obj = pegasos.pegasos_mnist()
        time_taken = time.time() - start_time
        if time_taken <= 15:
            score += 2

        model_solution = {
            'k=1000_lambda=0.1': 0.8100,
            'k=1_lambda=0.1': 0.7790,
            'k=10_lambda=0.1': 0.7980,
            'k=100_lambda=0.1': 0.8040,
            'k=100_lambda=1': 0.7880,
            'k=100_lambda=0.01': 0.7980
        }
        for key, value in model_solution.items():
            if key in test_acc:
                score += assign_score_to_accuracy(test_acc[key], value, 3.0)
    except:
        pass
    return score


def check_decision_tree():
    """ There is a problem in the original code and we decide to
    give everyone full points for decision tree.
    """
    return 10.0


if __name__ == "__main__":
    score_pegasos = check_pegasos()

    score_boosting = check_boosting()

    score_decision_tree = check_decision_tree()

    with open('output_hw3.txt', 'w') as f:
        f.write("[Pegasos]" + str(score_pegasos) + '\n')
        f.write("[Boosting]" + str(score_boosting) + '\n')
        f.write("[Decision Tree]" + str(score_decision_tree) + '\n')
        f.write("[Total]" + str(score_boosting + score_pegasos + score_decision_tree) + '\n')
