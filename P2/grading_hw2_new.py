import os
import json
import numpy as np


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


def check_grade_logistic_multiclass():
    binary_file = 'logistic_binary.out'
    multiclass_file = 'logistic_multiclass.out'

    results = {
        'bin_synthetic': 0,
        'bin_mnist': 0,
        'multinomial_3': 0,
        'multinomial_5': 0,
        'multinomial_mnist': 0,
        'ovr_3': 0,
        'ovr_5': 0,
        'ovr_mnist': 0
    }

    model_result = {
        'bin_syn_train': 0.994,
        'bin_syn_val': 1.,
        'bin_mnist_train': 0.871,
        'bin_mnist_val': 0.834,
        'multinomial_3_train': 0.902857,
        'multinomial_3_val': 0.840,
        'multinomial_5_train': 0.894286,
        'multinomial_5_val': 0.906667,
        'multinomial_mnist_train': 0.9454,
        'multinomial_mnist_val': 0.896,
        'ovr_3_train': 0.908571,
        'ovr_3_val': 0.840,
        'ovr_5_train': 0.765714,
        'ovr_5_val': 0.800,
        'ovr_mnist_train': 0.926800,
        'ovr_mnist_val': 0.897
    }

    if os.path.exists(binary_file):
        with open(binary_file, 'r') as f:
            line = f.readline()
            while line:
                description = line
                accuracies = f.readline()
                parts = accuracies.split(' ')

                try:
                    train = float(parts[2][:-1])
                    val = float(parts[5][:-1])
                except:
                    train = 0.
                    val = 0.

                if 'synthetic' in description:
                    if abs(train - model_result['bin_syn_train']) < 0.001:
                        results['bin_synthetic'] += 1
                    if abs(val - model_result['bin_syn_val']) < 0.001:
                        results['bin_synthetic'] += 1
                if 'MNIST' in description:
                    if abs(train - model_result['bin_mnist_train']) < 0.001:
                        results['bin_mnist'] += 1
                    if abs(val - model_result['bin_mnist_val']) < 0.001:
                        results['bin_mnist'] += 1
                line = f.readline()

    if os.path.exists(multiclass_file):
        with open(multiclass_file, 'r') as f:
            line = f.readline()
            while line:
                description = line
                ovr_model_descrition = f.readline()
                ovr_accuracies = f.readline()
                multinomial_model_descrition = f.readline()
                multinomial_accuracies = f.readline()
                ovr_parts = ovr_accuracies.split(' ')
                multinomial_parts = multinomial_accuracies.split(' ')

                try:
                    ovr_train = float(ovr_parts[2][:-1])
                    ovr_val = float(ovr_parts[5][:-1])
                except:
                    ovr_train = 0.
                    ovr_val = 0.

                try:
                    multinomial_train = float(multinomial_parts[2][:-1])
                    multinomial_val = float(multinomial_parts[5][:-1])
                except:
                    multinomial_train = 0.
                    multinomial_val = 0.

                if '3' in description:
                    if abs(ovr_train - model_result['ovr_3_train']) < 0.001:
                        results['ovr_3'] += 1
                    if abs(ovr_val - model_result['ovr_3_val']) < 0.001:
                        results['ovr_3'] += 1
                    if abs(multinomial_train - model_result['multinomial_3_train']) < 0.001:
                        results['multinomial_3'] += 1
                    if abs(multinomial_val - model_result['multinomial_3_val']) < 0.001:
                        results['multinomial_3'] += 1

                if '5' in description:
                    if abs(ovr_train - model_result['ovr_5_train']) < 0.001:
                        results['ovr_5'] += 1
                    if abs(ovr_val - model_result['ovr_5_val']) < 0.001:
                        results['ovr_5'] += 1
                    if abs(multinomial_train - model_result['multinomial_5_train']) < 0.001:
                        results['multinomial_5'] += 1
                    if abs(multinomial_val - model_result['multinomial_5_val']) < 0.001:
                        results['multinomial_5'] += 1

                if 'MNIST' in description:
                    if abs(ovr_train - model_result['ovr_mnist_train']) < 0.001:
                        results['ovr_mnist'] += 1
                    if abs(ovr_val - model_result['ovr_mnist_val']) < 0.001:
                        results['ovr_mnist'] += 1
                    if abs(multinomial_train - model_result['multinomial_mnist_train']) < 0.001:
                        results['multinomial_mnist'] += 1
                    if abs(multinomial_val - model_result['multinomial_mnist_val']) < 0.001:
                        results['multinomial_mnist'] += 1
                line = f.readline()

    return results


def calculate_dnn_misc():
    import dnn_misc
    np.random.seed(123)

    scores_dnn_misc = {
        "linear_forward": 0,
        "linear_backward": 0,
        "relu_forward": 0,
        "relu_backward": 0,
        "dropout_forward": 0,
        "dropout_backward": 0,
        "all_correct_bonus": 0
    }
    # example data
    X = np.random.normal(0, 1, (5, 3))

    # example modules
    check_linear = dnn_misc.linear_layer(input_D=3, output_D=2)
    check_relu = dnn_misc.relu()
    check_dropout = dnn_misc.dropout(r=0.5)

    # check_linear.forward
    hat_X = check_linear.forward(X)
    ground_hat_X = np.array([[0.42525407, -0.2120611],
                             [0.15174804, -0.36218431],
                             [0.20957104, -0.57861084],
                             [0.03460477, -0.35992763],
                             [-0.07256568, 0.1385197]])

    if (hat_X.shape[0] != 5) or (hat_X.shape[1] != 2):
        pass
    else:
        max_relative_diff = np.amax(np.abs(ground_hat_X - hat_X) / (ground_hat_X + 1e-8))
        # print('max_diff_output: ' + str(max_relative_diff))
        if max_relative_diff >= 1e-7:
            pass
            # print('linear.forward might be wrong')
        else:
            # print('linear.forward should be correct')
            scores_dnn_misc['linear_forward'] = 2
    # print('##########################')

    # check_linear.backward
    grad_hat_X = np.random.normal(0, 1, (5, 2))
    grad_X = check_linear.backward(X, grad_hat_X)

    ground_grad_X = np.array([[-0.32766959, 0.13123228, -0.0470483],
                              [0.22780188, -0.04838436, 0.04225799],
                              [0.03115675, -0.32648556, -0.06550193],
                              [-0.01895741, -0.21411292, -0.05212837],
                              [-0.26923074, -0.78986304, -0.23870499]])

    ground_grad_W = np.array([[-0.27579345, -2.08570514],
                              [4.52754775, -0.40995374],
                              [-1.2049515, 1.77662551]])

    ground_grad_b = np.array([[-4.55094716, -2.51399667]])

    if (grad_X.shape[0] != 5) or (grad_X.shape[1] != 3):
        uuuu = 0
        # print('Wrong output dimension of linear.backward')
    else:
        max_relative_diff_X = np.amax(np.abs(ground_grad_X - grad_X) / (ground_grad_X + 1e-8))
        # print('max_diff_grad_X: ' + str(max_relative_diff_X))
        max_relative_diff_W = np.amax(np.abs(ground_grad_W - check_linear.gradient['W']) / (ground_grad_W + 1e-8))
        # print('max_diff_grad_W: ' + str(max_relative_diff_W))
        max_relative_diff_b = np.amax(np.abs(ground_grad_b - check_linear.gradient['b']) / (ground_grad_b + 1e-8))
        # print('max_diff_grad_b: ' + str(max_relative_diff_b))

        if (max_relative_diff_X >= 1e-7) or (max_relative_diff_W >= 1e-7) or (max_relative_diff_b >= 1e-7):
            pass
            # print('linear.backward might be wrong')
        else:
            # print('linear.backward should be correct')
            scores_dnn_misc['linear_backward'] = 2
    # print('##########################')

    # check_relu.forward
    hat_X = check_relu.forward(X)
    ground_hat_X = np.array([[0., 0.99734545, 0.2829785],
                             [0., 0., 1.65143654],
                             [0., 0., 1.26593626],
                             [0., 0., 0.],
                             [1.49138963, 0., 0.]])

    if (hat_X.shape[0] != 5) or (hat_X.shape[1] != 3):
        uuuu = 0
        # print('Wrong output dimension of relu.forward')
    else:
        max_relative_diff = np.amax(np.abs(ground_hat_X - hat_X) / (ground_hat_X + 1e-8))
        # print('max_diff_output: ' + str(max_relative_diff))
        if max_relative_diff >= 1e-7:
            pass
            # print('relu.forward might be wrong')
        else:
            # print('relu.forward should be correct')
            scores_dnn_misc['relu_forward'] = 2
    # print('##########################')

    # check_relu.backward
    grad_hat_X = np.random.normal(0, 1, (5, 3))
    grad_X = check_relu.backward(X, grad_hat_X)
    ground_grad_X = np.array([[-0., 0.92746243, -0.17363568],
                              [0., 0., -0.87953634],
                              [0., -0., -1.72766949],
                              [-0., 0., 0.],
                              [-0.01183049, 0., 0.]])

    if (grad_X.shape[0] != 5) or (grad_X.shape[1] != 3):
        pass
        # print('Wrong output dimension of relu.backward')
    else:
        max_relative_diff_X = np.amax(np.abs(ground_grad_X - grad_X) / (ground_grad_X + 1e-8))
        # print('max_diff_grad_X: ' + str(max_relative_diff_X))

        if (max_relative_diff_X >= 1e-7):
            pass
            # print('relu.backward might be wrong')
        else:
            # print('relu.backward should be correct')
            scores_dnn_misc['relu_backward'] = 2
    # print('##########################')

    # check_dropout.forward
    hat_X = check_dropout.forward(X, is_train=True)
    scores_dnn_misc['dropout_forward'] = 2

    # check_dropout.backward
    grad_hat_X = np.random.normal(0, 1, (5, 3))
    grad_X = check_dropout.backward(X, grad_hat_X)
    ground_grad_X = np.array([[0., -0.39530184, -1.45606984],
                              [-1.22062684, -0., 0.],
                              [0., 1.7354356, 2.53503582],
                              [4.21567995, -0.4721789, -0.46416366],
                              [-2.15627882, 2.32636907, 1.04498015]])

    if (grad_X.shape[0] != 5) or (grad_X.shape[1] != 3):
        pass
        # print('Wrong output dimension of dropout.backward')
    else:
        max_relative_diff_X = np.amax(np.abs(ground_grad_X - grad_X) / (grad_X + 1e-8))
        # print('max_diff_grad_X: ' + str(max_relative_diff_X))

        if (max_relative_diff_X >= 1e-7):
            pass
            # print('dropout.backward might be wrong')
        else:
            # print('dropout.backward should be correct')
            scores_dnn_misc['dropout_backward'] = 2
    # print('##########################')
    if sum(scores_dnn_misc.values()) == 12:
        scores_dnn_misc['all_correct_bonus'] = 2
    return scores_dnn_misc


def check_grade_dnn(file_name):
    score_dnn = {x: 0 for x in file_name}
    lower_ratio = 0.999
    upper_ratio = 1.001
    for jfile in file_name:
        if not os.path.exists(jfile):
            score_dnn[jfile] = 0
        else:
            sol = json.load(open(jfile, 'r'))
            c_sol = json.load(open('sol_' + jfile, 'r'))

            if not ('train' in sol and
                    'val' in sol and
                    len(sol['train']) == len(c_sol['train']) and
                    len(sol['val']) == len(c_sol['val'])):
                score_dnn[jfile] = 0
            else:
                total = len(sol['val']) + len(sol['train'])
                file_score = 0
                for i in range(len(c_sol['train'])):
                    try:
                        file_score += assign_score_to_accuracy(float(sol['train'][i]), float(c_sol['train'][i]))
                    except:
                        pass
                for i in range(len(c_sol['val'])):
                    try:
                        file_score += assign_score_to_accuracy(float(sol['val'][i]), float(c_sol['val'][i]))
                    except:
                        pass
                score_dnn[jfile] = file_score * float(2) / total
    score_dnn['CNN2_lr0.001_m0.9_w0.0_d0.5.json'] *= 2

    return score_dnn


if __name__ == "__main__":

    logistic_multiclass = {
        'bin_synthetic': 0,
        'bin_mnist': 0,
        'multinomial_3': 0,
        'multinomial_5': 0,
        'multinomial_mnist': 0,
        'ovr_3': 0,
        'ovr_5': 0,
        'ovr_mnist': 0
    }

    scores_dnn_misc = {
        "linear_forward": 0,
        "linear_backward": 0,
        "relu_forward": 0,
        "relu_backward": 0,
        "dropout_forward": 0,
        "dropout_backward": 0,
        "all_correct_bonus": 0
    }

    file_name = ['MLP_lr0.01_m0.0_w0.0_d0.0.json', 'MLP_lr0.01_m0.0_w0.0_d0.5.json',
                 'MLP_lr0.01_m0.0_w0.0_d0.95.json', 'LR_lr0.01_m0.0_w0.0_d0.0.json',
                 'CNN_lr0.01_m0.0_w0.0_d0.5.json', 'CNN_lr0.01_m0.9_w0.0_d0.5.json',
                 'CNN2_lr0.001_m0.9_w0.0_d0.5.json']

    score_dnn = {x: 0 for x in file_name}

    try:
        logistic_multiclass = check_grade_logistic_multiclass()
    except:
        pass
    try:
        scores_dnn_misc = calculate_dnn_misc()
    except:
        pass
    try:
        score_dnn = check_grade_dnn(file_name)
    except:
        pass

    total_score = 0.0
    with open('output_hw2.txt', 'w') as f:
        for key, value in logistic_multiclass.items():
            f.write("[" + key + "]:" + str(value) + '\n')
            total_score += value
        for key, value in scores_dnn_misc.items():
            f.write("[" + key + "]:" + str(value) + '\n')
            total_score += value
        for key, value in score_dnn.items():
            f.write("[" + key + "]:" + str(value) + '\n')
            total_score += value
        f.write("[Total]:" + str(total_score) + '\n')
