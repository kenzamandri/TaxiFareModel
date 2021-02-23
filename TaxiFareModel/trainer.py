# imports
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pipe_distance = make_pipeline(DistanceTransformer(), StandardScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder(handle_unknown='ignore'))
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']
        feat_eng_bloc = ColumnTransformer([('distance', pipe_distance, dist_cols),
                                       ('time', pipe_time, time_cols)], remainder='drop')
        pipe_cols = Pipeline(steps=[('feat_eng_bloc', feat_eng_bloc),
                                ('regressor', LinearRegression())])
        return pipe_cols

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        trained = self.pipeline.fit(self.X, self.y)
        return trained

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        Rmse = compute_rmse(y_pred, y_test)
        return Rmse


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df_clean = clean_data(df)
    # set X and y
    X = df_clean.drop(columns='fare_amount')
    y = df_clean['fare_amount']
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # train
    trainer = Trainer(X_train, y_train)

    train = trainer.run()
    # evaluate
    evaluation = trainer.evaluate(X_test, y_test)

    print('TODO')
