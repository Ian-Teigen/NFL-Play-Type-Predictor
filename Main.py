#Imports
import nflreadpy as nfl
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
import seaborn as sns
import matplotlib.pyplot as plt

#Data exploration and preparation
pbp_train = nfl.load_pbp([2022, 2023, 2024])
pbp_test_initial = nfl.load_pbp(2025)

pbp = pbp_train.to_pandas() 
pbp = pbp[pbp['season_type'] == 'REG']

pbp_test_initial = pbp_test_initial.to_pandas()
pbp_test_initial = pbp_test_initial[pbp_test_initial['season_type'] == 'REG']

print(pbp.shape) 
print(pbp.columns.tolist())
print(pbp.head())

#Separating it into the pre snap features that can be used to predict the play type (run vs pass)
X = ['yardline_100','half_seconds_remaining','game_seconds_remaining','drive','down','goal_to_go','ydstogo',
     'posteam_timeouts_remaining','defteam_timeouts_remaining','score_differential']
y = ['play_type']

pbp_train = pbp[X + y]
pbp_test = pbp_test_initial[X + y]
print(pbp_train.columns.tolist())
print(pbp_test.shape)

pbp_train = pbp_train[pbp_train['play_type'].isin(['run', 'pass'])] #Splitting the dataset so that it only includes runs and passes
pbp_train = pbp_train[pbp_train['down'].isin([1,2,3])] #Splitting the dataset so that it only 1st 2nd and 3rd downs
pbp_test = pbp_test[pbp_test['play_type'].isin(['run', 'pass'])]
pbp_test = pbp_test[pbp_test['down'].isin([1,2,3])]


print(pbp_train.describe())
print(pbp_train.isnull().sum())

target = (pbp_train['play_type'] == 'pass').astype(int) #Changing it into 1 for pass and 0 for run
test_target = (pbp_test['play_type'] == 'pass').astype(int) 

#Model fitting
X_train = pbp_train[X]
y_train = target
X_test = pbp_test[X]
y_test = test_target

xgb_model = XGBClassifier(
    n_estimators = 200,
    max_depth = 5,
    learning_rate = 0.1,
    random_state = 42   
)

xgb_model.fit(X_train, y_train)

pred_class = xgb_model.predict(X_test) #Predicts explicitly run (0) or pass (1)
pred_prob = xgb_model.predict_proba(X_test)[:, 1] #Predicts the probability of a pass

accuracy = accuracy_score(y_test, pred_class)
log_l = log_loss(y_test, pred_prob)

print(accuracy) #0.70
print(log_l) #0.57

importance = pd.Series(xgb_model.feature_importances_, index=X).sort_values(ascending=False) #feature importance
print(importance)

plt.figure(figsize = (8,6))
sns.barplot(x = importance.values, y = importance.index)
plt.title("Feature Importance")
plt.ylabel("Feature")
plt.savefig('images/feature_importance.png', dpi=150, bbox_inches='tight')

actual = pbp_test.groupby(['down', 'ydstogo'])['play_type'].apply(lambda x: (x == 'pass').mean()).reset_index()
actual.columns = ['down', 'ydstogo', 'rate']
actual['type'] = 'Actual'

plot_df = X_test.copy()
plot_df['pred_prob'] = pred_prob
predicted = plot_df.groupby(['down', 'ydstogo'])['pred_prob'].mean().reset_index()
predicted.columns = ['down', 'ydstogo', 'rate']
predicted['type'] = 'Predicted'

combined = pd.concat([actual, predicted], ignore_index=True)
combined = combined[combined['ydstogo'] <= 25]  #Clips it so that it only includes 25 yards or less
combined['down'] = combined['down'].astype(int)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

downs = [1, 2, 3]

for ax, d in zip(axes, downs):
    subset = combined[combined['down'] == d]
    sns.lineplot(
        data=subset,
        x='ydstogo', y='rate',
        style='type',
        marker='o',
        ax=ax,
        legend=(d == 1)  #only show the legend once on the first panel
    )
    ax.set_title(f"Down {d}")
    ax.set_xlabel("Yards to Go")
    ax.set_ylabel("Pass Rate" if d == 1 else "")

plt.tight_layout()
plt.savefig('images/pass_rate.png', dpi=150, bbox_inches='tight')
