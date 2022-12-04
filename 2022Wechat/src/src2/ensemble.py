import optuna 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from category_id_map import lv2id_to_lv1id, CATEGORY_ID_LIST, lv2id_to_category_id, category_id_to_lv1id, category_id_to_lv2id
from pprint import pprint

def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'mean_f1': mean_f1,
                    'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    }

    return eval_results

def objective_fold0(trial):
    w1 = trial.suggest_float('w1', 0, 1)
    w2 = trial.suggest_float('w2', 0, 1-w1)
    w3 = trial.suggest_float('w3', 0, 1-w1-w2)

    logits_pl = np.load("./src/src2/pl_version/checkpoints_finetune_fold0/test_b_logits_fold0.npy")
    logits_tez = np.load("./src/src2/tez_version/checkpoints_finetune_fold0/test_b_logits_fold0.npy")
    logits_lxmert = np.load("./src/src2/lxmert/checkpoints_finetune_fold0/test_b_logits_fold0.npy")

    targets = np.load("./src/src2/pl_version/checkpoints_finetune_fold0/test_b_target_fold0.npy")

    logits = w1 * logits_pl + w2 * logits_tez + w3 * logits_lxmert
    preds = logits.argmax(1)

    results = evaluate(preds, targets)

    return results['mean_f1']

study_fold0 = optuna.create_study(direction='maximize')
study_fold0.optimize(objective_fold0, n_trials=100)

fold0_params = study_fold0.best_params
pprint(fold0_params)

def objective_fold1(trial):
    w1 = trial.suggest_float('w1', 0, 1)
    w2 = trial.suggest_float('w2', 0, 1-w1)
    w3 = trial.suggest_float('w3', 0, 1-w1-w2)

    logits_pl = np.load("./src/src2/pl_version/checkpoints_finetune_fold1/test_b_logits_fold1.npy")
    logits_tez = np.load("./src/src2/tez_version/checkpoints_finetune_fold1/test_b_logits_fold1.npy")
    logits_lxmert = np.load("./src/src2/lxmert/checkpoints_finetune_fold1/test_b_logits_fold1.npy")

    targets = np.load("./src/src2/pl_version/checkpoints_finetune_fold1/test_b_target_fold1.npy")

    logits = w1 * logits_pl + w2 * logits_tez + w3 * logits_lxmert
    preds = logits.argmax(1)

    results = evaluate(preds, targets)

    return results['mean_f1']


study_fold1 = optuna.create_study(direction='maximize')
study_fold1.optimize(objective_fold1, n_trials=100)

fold1_params = study_fold1.best_params  
pprint(fold1_params)

def objective_fold2(trial):
    w1 = trial.suggest_float('w1', 0, 1)
    w2 = trial.suggest_float('w2', 0, 1-w1)
    w3 = trial.suggest_float('w3', 0, 1-w1-w2)

    logits_pl = np.load("./src/src2/pl_version/checkpoints_finetune_fold2/test_b_logits_fold2.npy")
    logits_tez = np.load("./src/src2/tez_version/checkpoints_finetune_fold2/test_b_logits_fold2.npy")
    logits_lxmert = np.load("./src/src2/lxmert/checkpoints_finetune_fold2/test_b_logits_fold2.npy")

    targets = np.load("./src/src2/pl_version/checkpoints_finetune_fold2/test_b_target_fold2.npy")

    logits = w1 * logits_pl + w2 * logits_tez + w3 * logits_lxmert
    preds = logits.argmax(1)

    results = evaluate(preds, targets)

    return results['mean_f1']


study_fold2 = optuna.create_study(direction='maximize')
study_fold2.optimize(objective_fold2, n_trials=100)

fold2_params = study_fold2.best_params 
pprint(fold2_params)


def objective_fold3(trial):
    w1 = trial.suggest_float('w1', 0, 1)
    w2 = trial.suggest_float('w2', 0, 1-w1)
    w3 = trial.suggest_float('w3', 0, 1-w1-w2)

    logits_pl = np.load("./src/src2/pl_version/checkpoints_finetune_fold3/test_b_logits_fold3.npy")
    logits_tez = np.load("./src/src2/tez_version/checkpoints_finetune_fold3/test_b_logits_fold3.npy")
    logits_lxmert = np.load("./src/src2/lxmert/checkpoints_finetune_fold3/test_b_logits_fold3.npy")

    targets = np.load("./src/src2/pl_version/checkpoints_finetune_fold3/test_b_target_fold3.npy")

    logits = w1 * logits_pl + w2 * logits_tez + w3 * logits_lxmert
    preds = logits.argmax(1)

    results = evaluate(preds, targets)

    return results['mean_f1']


study_fold3 = optuna.create_study(direction='maximize')
study_fold3.optimize(objective_fold3, n_trials=100)

fold3_params = study_fold3.best_params 
pprint(fold3_params)


fold0_pl = np.load("./src/src2/pl_version/checkpoints_finetune_fold0/test_b_fold0.npy")
fold1_pl = np.load("./src/src2/pl_version/checkpoints_finetune_fold1/test_b_fold1.npy")
fold2_pl = np.load("./src/src2/pl_version/checkpoints_finetune_fold2/test_b_fold2.npy")
fold3_pl = np.load("./src/src2/pl_version/checkpoints_finetune_fold3/test_b_fold3.npy")

fold0_tez = np.load("./src/src2/tez_version/checkpoints_finetune_fold0/test_b_fold0.npy")
fold1_tez = np.load("./src/src2/tez_version/checkpoints_finetune_fold1/test_b_fold1.npy")
fold2_tez = np.load("./src/src2/tez_version/checkpoints_finetune_fold2/test_b_fold2.npy")
fold3_tez = np.load("./src/src2/tez_version/checkpoints_finetune_fold3/test_b_fold3.npy")

fold0_lxmert = np.load("./src/src2/lxmert/checkpoints_finetune_fold0/test_b_fold0.npy")
fold1_lxmert = np.load("./src/src2/lxmert/checkpoints_finetune_fold1/test_b_fold1.npy")
fold2_lxmert = np.load("./src/src2/lxmert/checkpoints_finetune_fold2/test_b_fold2.npy")
fold3_lxmert = np.load("./src/src2/lxmert/checkpoints_finetune_fold3/test_b_fold3.npy")

result = (fold0_params['w1'] * fold0_pl + fold0_params['w2'] * fold0_tez + fold0_params['w3'] * fold0_lxmert) \
    + (fold1_params['w1'] * fold1_pl + fold1_params['w2'] * fold1_tez + fold1_params['w3'] * fold1_lxmert) \
    + (fold2_params['w1'] * fold2_pl + fold2_params['w2'] * fold2_tez + fold2_params['w3'] * fold2_lxmert) \
    + (fold3_params['w1'] * fold3_pl + fold3_params['w2'] * fold3_tez + fold3_params['w3'] * fold3_lxmert)

np.save("./data/result_2.npy", result / 4)