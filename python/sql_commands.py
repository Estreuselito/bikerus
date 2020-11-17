#### this script contains all sql commands used within this project ####

create_table_hours = ''' 
        CREATE TABLE hours( 
            instant INTEGER, 
            dteday DATE, 
            season INTEGER, 
            yr INTEGER, 
            mnth INTEGER,
            hr INTEGER,
            holiday INTEGER,
            weekday INTEGER,
            workingday INTEGER,
            weathersit INTEGER,
            temp REAL,
            atemp REAL,
            hum REAL,
            windspeed INTEGER,
            casual INTEGER,
            registered INTEGER,
            cnt INTEGER 
            ) 
        '''

create_table_hours_complete = ''' 
        CREATE TABLE hours_complete( 
            datetime DATE, 
            dteday DATE, 
            season INTEGER, 
            yr INTEGER, 
            mnth INTEGER,
            hr INTEGER,
            holiday INTEGER,
            weekday INTEGER,
            workingday INTEGER,
            weathersit INTEGER,
            temp REAL,
            atemp REAL,
            hum REAL,
            windspeed INTEGER,
            casual INTEGER,
            registered INTEGER,
            cnt INTEGER 
            ) 
        '''

create_table_hours_preprocessed = ''' 
        CREATE TABLE hours_preprocessed( 
            datetime DATE,  
            season INTEGER, 
            yr INTEGER, 
            mnth INTEGER,
            hr INTEGER,
            holiday INTEGER,
            weekday INTEGER,
            workingday INTEGER,
            weathersit INTEGER,
            temp REAL,
            hum REAL,
            windspeed INTEGER,
            cnt INTEGER 
            ) 
        '''

create_table_raw = """
    CREATE TABLE raw(
        Duration INTEGER,
        [Start date] DATE,
        [End date] DATE,
        [Start station number] INTEGER,
        [Start station] TEXT,
        [End station number] INTEGER,
        [End station] TEXT,
        [Bike number] INTEGER,
        [Member type] TEXT
        )
    """

create_table_station_locations = """
    CREATE TABLE station_locations(
        OBJECTID INTEGER,
        ID INTEGER,
        ADDRESS TEXT,
        TERMINAL_NUMBER	INTEGER,
        LATITUDE FLOAT(10, 8),
        LONGITUDE FLOAT(11, 8),
        INSTALLED TEXT,
        LOCKED TEXT,
        INSTALL_DATE DATE,
        REMOVAL_DATE DATE,
        TEMPORARY_INSTALL TEXT,
        NUMBER_OF_BIKES	INTEGER,
        NUMBER_OF_EMPTY_DOCKS INTEGER,
        X FLOAT,
        Y FLOAT,
        SE_ANNO_CAD_DATA FLOAT,
        OWNER TEXT
        )
    """

create_table_max_min_count = """
    CREATE TABLE max_min_count(
        max INTEGER,
        min INTEGER
        )
    """

create_table_X_train_test = """
    CREATE TABLE {}(
            datetime DATE,  
            season INTEGER, 
            yr INTEGER, 
            mnth INTEGER,
            hr INTEGER,
            holiday INTEGER,
            weekday INTEGER,
            workingday INTEGER,
            weathersit INTEGER,
            temp REAL,
            hum REAL,
            windspeed INTEGER
            ) 
        """

create_table_Y_train_test = """
    CREATE TABLE {}(
            cnt INTEGER
            ) 
        """

create_table_predicted_df = """
    CREATE TABLE predicted_df(
        datetime DATE,  
        season INTEGER, 
        yr INTEGER, 
        mnth INTEGER,
        hr INTEGER,
        holiday INTEGER,
        weekday INTEGER,
        workingday INTEGER,
        weathersit INTEGER,
        temp REAL,
        hum REAL,
        windspeed INTEGER,
        cnt INTEGER,
        cnt_norm INTEGER,
        [cnt_pred_sklearn.neural_network._multilayer_perceptron] INTEGER,
        [cnt_pred_norm_sklearn.neural_network._multilayer_perceptron] INTEGER,
        [cnt_pred_sklearn.ensemble._forest] INTEGER,
        [cnt_pred_norm_sklearn.ensemble._forest] INTEGER, 
        [cnt_pred_catboost.core] INTEGER,
        [cnt_pred_norm_catboost.core] INTEGER
        )
    """
