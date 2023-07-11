import numpy as np
import pandas as pd
import torchtuples as tt
from sksurv.ensemble import RandomSurvivalForest
from pycox.models import CoxTime
from pycox.models import CoxPH
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

np.random.seed(1234)

c_RSF = []
b_RSF = []

c_Cox = []
b_Cox = []

c_DeepSurv = []
b_DeepSurv = []

data = np.genfromtxt("data_external.csv", delimiter=',')
kf = KFold(n_splits=5, random_state=1, shuffle=False)
test_index_list = [np.arange(0, 120, 1),
                    np.arange(120, 244, 1),
                    np.arange(244, 352, 1),
                    np.arange(352, 470, 1),
                    np.arange(470, 592, 1)]

for i in range(5):
    test_index = test_index_list[i]
    train_index = np.arange(0, 592, 1)
    for j in test_index:
        train_index = train_index[train_index != j]

    data_train = data[train_index]
    data_test = data[test_index]
    x_train, x_test = data[train_index, :-2].astype('float32'), data[test_index, :-2].astype('float32')
    t_train, t_test = data[train_index, -2].astype('float32'), data[test_index, -2].astype('float32')
    e_train, e_test = data[train_index, -1].astype('bool'), data[test_index, -1].astype('bool')
    risk = x_test[:, 14]

    # Cox
    columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
               '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
               '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
               '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
               '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
               '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
               '60', 'duration', 'event']
    df_train = pd.DataFrame(data_train, columns=columns)
    df_val = pd.DataFrame(data_test, columns=columns)

    labtrans = CoxTime.label_transform()
    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))
    val = tt.tuplefy(x_train, y_val)
    net = MLPVanillaCoxTime(x_train.shape[1], [128, 128], True, 0.1)
    model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
    model.optimizer.set_lr(0.01)
    model.fit(x_train, y_train, 64, 100, [tt.callbacks.EarlyStopping()], False, val_data=val)
    model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test)
    ev = EvalSurv(surv, t_test, e_test, censor_surv='km')
    c_index = ev.concordance_td()
    time_grid = np.linspace(t_test.min(), t_test.max(), 100)
    IBS = ev.integrated_brier_score(time_grid)
    c_Cox.append(c_index)
    b_Cox.append(IBS)
    if i == 0:
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 0.4)[0]].mean(1), c='#2ca02c', linewidth=2.5,
                 label='risk=1')
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 0.6)[0]].mean(1), c='#1f77b4', linewidth=2.5,
                 label='risk=1.5')
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 0.8)[0]].mean(1), c='#ff7f0e', linewidth=2.5,
                 label='risk=2')
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 1)[0]].mean(1), c='#d62728', linewidth=2.5,
                 label='risk=2.5')
        plt.xlabel("Time(hours)", fontsize=16)
        plt.ylabel("Non-bleeding probability", fontsize=16)
        plt.title("Predicted non-bleeding curves-Cox", fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/risk groups-Cox.jpg')
        plt.show()

        plt.plot(surv.index.values, surv.to_numpy().mean(1), linewidth=2, label='All')
        plt.plot(surv.index.values, surv.values[:, np.where(e_test == 1)[0]].mean(1), linewidth=2.5, label='Bleeding group')
        plt.plot(surv.index.values, surv.values[:, np.where(e_test == 0)[0]].mean(1), linewidth=2.5, label='Non-bleeding group')
        plt.xlabel("Time(hours)", fontsize=16)
        plt.ylabel("Non-bleeding probability", fontsize=16)
        plt.title("Predicted non-bleeding curves-Cox", fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/bleeding groups-Cox.jpg')
        plt.show()


    # DeepSurv
    y_train = (t_train, e_train)
    y_test = (t_test, e_test)
    test = x_test, y_test
    net = tt.practical.MLPVanilla(x_train.shape[1], [128, 128], 1, True, 0.2, output_bias=False)
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(0.001)
    model.fit(x_train, y_train, 64, 512, [tt.callbacks.EarlyStopping()], False, val_data=test, val_batch_size=128)
    model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test)
    ev = EvalSurv(surv, t_test, e_test, censor_surv='km')
    c_index = ev.concordance_td()
    time_grid = np.linspace(t_test.min(), t_test.max(), 100)
    IBS = ev.integrated_brier_score(time_grid)
    c_DeepSurv.append(c_index)
    b_DeepSurv.append(IBS)
    if i == 0:
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 0.4)[0]].mean(1), c='#2ca02c', linewidth=2.5,
                 label='risk=1')
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 0.6)[0]].mean(1), c='#1f77b4', linewidth=2.5,
                 label='risk=1.5')
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 0.8)[0]].mean(1), c='#ff7f0e', linewidth=2.5,
                 label='risk=2')
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 1)[0]].mean(1), c='#d62728', linewidth=2.5,
                 label='risk=2.5')
        plt.xlabel("Time(hours)", fontsize=16)
        plt.ylabel("Non-bleeding probability", fontsize=16)
        plt.title("Predicted non-bleeding curves-DeepSurv", fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/risk groups-DeepSurv.jpg')
        plt.show()


        plt.plot(surv.index.values, surv.to_numpy().mean(1), linewidth=2, label='All')
        plt.plot(surv.index.values, surv.values[:, np.where(e_test == 1)[0]].mean(1), linewidth=2.5, label='Bleeding group')
        plt.plot(surv.index.values, surv.values[:, np.where(e_test == 0)[0]].mean(1), linewidth=2.5, label='Non-bleeding group')
        plt.xlabel("Time(hours)", fontsize=16)
        plt.ylabel("Non-bleeding probability", fontsize=16)
        plt.title("Predicted non-bleeding curves-DeepSurv", fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/bleeding groups-DeepSurv.jpg')
        plt.show()



    # RSF
    x_train_df, x_test_df = pd.DataFrame(x_train), pd.DataFrame(x_test)
    y_train_df = np.array(pd.DataFrame({'cens': e_train, 'time': t_train}).to_records(index=False))
    y_test_df = np.array(pd.DataFrame({'cens': e_test, 'time': t_test}).to_records(index=False))
    rsf = RandomSurvivalForest(n_estimators=700,  random_state=111)
    rsf.fit(x_train_df, y_train_df)
    surv_data = rsf.predict_survival_function(x_test_df)
    surv = pd.DataFrame(surv_data.T, rsf.event_times_)
    ev = EvalSurv(surv, t_test, e_test.astype('int32'), censor_surv='km')
    c_index = ev.concordance_td()
    time_grid = np.linspace(t_test.min(), t_test.max(), 100)
    IBS = ev.integrated_brier_score(time_grid)
    c_RSF.append(c_index)
    b_RSF.append(IBS)
    if i == 0:
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 0.4)[0]].mean(1), c='#2ca02c',
                 linewidth=2.5,
                 label='risk=1')
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 0.6)[0]].mean(1), c='#1f77b4',
                 linewidth=2.5,
                 label='risk=1.5')
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 0.8)[0]].mean(1), c='#ff7f0e',
                 linewidth=2.5,
                 label='risk=2')
        plt.plot(surv.index.values, surv.to_numpy()[:, np.where(risk == 1)[0]].mean(1), c='#d62728', linewidth=2.5,
                 label='risk=2.5')
        plt.xlabel("Time(hours)", fontsize=16)
        plt.ylabel("Non-bleeding probability", fontsize=16)
        plt.title("Predicted non-bleeding curves-RSF", fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/risk groups-RSF.jpg')
        plt.show()

        plt.plot(surv.index.values, surv.to_numpy().mean(1), linewidth=2, label='All')
        plt.plot(surv.index.values, surv.values[:, np.where(e_test == 1)[0]].mean(1), linewidth=2.5,
                 label='Bleeding group')
        plt.plot(surv.index.values, surv.values[:, np.where(e_test == 0)[0]].mean(1), linewidth=2.5,
                 label='Non-bleeding group')
        plt.xlabel("Time(hours)", fontsize=16)
        plt.ylabel("Non-bleeding probability", fontsize=16)
        plt.title("Predicted non-bleeding curves-RSF", fontsize=16)
        plt.legend(fontsize=16)
        plt.grid()
        plt.tight_layout()
        plt.savefig('results/bleeding groups-RSF.jpg')
        plt.show()


result = np.array([[np.mean(c_Cox), np.mean(c_DeepSurv), np.mean(c_RSF)],
                   [np.mean(b_Cox), np.mean(b_DeepSurv), np.mean(b_RSF)]])
columns = ['CoxPH', 'DeepSurv', 'RSF']
index = ['C_index', 'Brier Score']
df = pd.DataFrame(data=result, index=index, columns=columns)
df.to_csv('results/result.txt', sep='\t')
# print(df)

