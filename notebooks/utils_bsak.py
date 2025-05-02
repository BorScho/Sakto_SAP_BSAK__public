import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import re


def plot_confusion_matrix(y_test, y_pred, labels, cmap="Blues"):
    xticklabels = labels
    yticklabels = labels
    # define confusion matrix:
    conf_matrix = np.zeros((len(np.unique(y_test)), len(np.unique(y_test))), dtype=int)
    # count how often each true class /pred class combination occurs:
    for i in range(len(y_test)):
        true_class = y_test[i]
        pred_class = y_pred[i]
        conf_matrix[true_class, pred_class] += 1
    # vizualize the matrix:
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels)
    #sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"Confusion Matrix - Support: {len(y_test)}")
    plt.show()


def plot_top_k_confusion_matrix(y_test, y_pred_prob, labels, top_k=3, cmap="Blues", show_off_top_k_info=False):
    """ 
    Plot a confusion map based on top-k-accuracy.
    Hit is, when y_test is in the top k of y_pred_prob. 
    If y_test is no hit, it does not appear in the matrix, i.e. the sum of the matrix-entries is less than
    the support. "Set show_off_top_k_info" to "true" to get an extra column, showing the count for the off-top-k.

    Input:
    y_test : ground truth data-labels
    y_pred_prob : the probabilites for each class; we sort the probabilites and take the top k
    labels : the labels for the axises of the matrix (not the data-labels)
    show_off_top_k_info : it true, displays an extra column in the matrix to show the number of datapoints off top-k
    """
    k = top_k
    if show_off_top_k_info: # if true append one off-top column to the confusion matrix
        xticklabels = np.append(labels.astype(str), [f"off_top_{k}"]) # if y_true is not in the top_k it is "off_top_k"
    else:
        xticklabels = labels.astype(str)
    yticklabels = labels.astype(str)
    top_k_preds = np.argsort(y_pred_prob, axis=1 )[:, ::-1][:, :k] # sort probabilites asscending, take top k
    # define the top-k confusion matrix:
    conf_matrix = np.zeros((len(np.unique(y_test)), len(xticklabels)), dtype=int)
    # count how often the true class is contained in the top-k predictions:
    for i in range(len(y_test)):
        true_class = y_test[i]
        # check if the true class is contained in the top-k predictions for this instance:
        if true_class in top_k_preds[i]:
            pred_class = top_k_preds[i][0]  # the first (i.e. most probable) prediction is chosen as the dominant class
            conf_matrix[true_class, pred_class] += 1
        else:
            conf_matrix[true_class, len(xticklabels) - 1] += 1
            if show_off_top_k_info:
                print(f"Off-top-{k}:")
                print(labels[true_class], labels[top_k_preds[i]])
    # vizualize the matrix:
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels)
    #sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title(f"Top-{k} Confusion Matrix - Support: {len(y_test)}")
    plt.show()


def top_k_accuracy_factory(top_k=3):
    """ 
    factory to create a top-k-accuracy metric function for arbitrary k:
    create a user defined metric function for top-k-accuracy like this
    top_k_acc = top_k_accuracy_factory(top_k=k) 
    """
    k = top_k
    def top_k_accuracy(y_true, y_pred_prob):
        """
        Input:
        y_true : ground truth labels
        y_pred_prob: the probabilites for each class; we sort the probabilites and get the top k
        Returns:
        accuracy : hit is, when y_true is in the top k.
        """
        nonlocal k
        top_k_preds = np.argsort(y_pred_prob, axis=1)[:, ::-1][:, :k]
        # count how often the true class is contained in the top k:
        correct = sum([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        
        # calculate the top k accuracy
        return correct / len(y_true)
    
    return top_k_accuracy


def target_min_value_records(dataframe, target_column, min_value_records=2):
    """ 
    for a stratified split of a dataframe in a train and test frame, we have to make sure, that there are enough
    records in the dataset to split them with the same relative frequency between the target and test sets.
    + target_column - the column that is going to be predicted, y
    + min_value_records - the number of records for a target value that minimally has to be available to be able to split - must be greater or equal than 2
        
    Returns:
    + if min_value_records >= 2: returns array of target values, each of which has at least "min_value_counts" records
    + returns the dataframe if not min_value_records >= 2 .
    """
    from sklearn.preprocessing import LabelEncoder
    df = dataframe
    target = target_column
    # for stratification, all target classes have to have more than 1 record: 
    if min_value_records < 2:
        return df
    a = df[target].value_counts() > min_value_records
    min_record_target_values = a[a].index

    return min_record_target_values


def is_date_column(series):
    """
    Checks if a column contains mostly date values in supported formats.
    Returns True if at least 90% of the input series match a date format
    """
    # Define regex patterns for different date formats
    date_patterns = [
        re.compile(r"^(0[1-9]|[12][0-9]|3[01])\.(0[1-9]|1[0-2])\.(\d{4})$"),  # dd.mm.yyyy
        re.compile(r"^(0[1-9]|[12][0-9]|3[01])-(0[1-9]|1[0-2])-(\d{4})$"),  # dd-mm-yyyy
        re.compile(r"^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/(\d{4})$"),  # dd/mm/yyyy
        re.compile(r"^(\d{4})/(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])$"),  # yyyy/mm/dd
        re.compile(r"^(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$")  # yyyy-mm-dd
    ]
    # Check if a column contains mostly date values in supported formats.
    def matches_date_format(value):
        return any(pattern.match(value) for pattern in date_patterns)
    matches = series.dropna().apply(lambda x: matches_date_format(str(x)))  # Check against patterns

    return matches.mean() > 0.9  # At least 90% should match a date format


def is_decimal_column(series):
    """
    Check if a column contains mostly decimal-format - like 1.234,56 or 0,34 - values.
    Converts decimal / money-value columns to floats
    Function to detect decimal values in a column
    Returns True if at least 90% of the input series match a decimal format
    """
    # Regex pattern for detecting decimal format (thousand dots & decimal comma)
    decimal_pattern = re.compile(r"^\d{1,3}(\.\d{3})*(,\d*)?$|^\d+(,\d*)?$")
    matches = series.dropna().apply(lambda x: bool(decimal_pattern.match(str(x))))  # Apply regex

    return matches.mean() > 0.9  # At least 90% of values should match


def convert_column_decimal2float(series):
    """ 
    Converts German-formatted decimal strings to float, handles '123,', '9.00-', etc. 
    """
    # Tausenderpunkte entfernen und Komma zu Punkt
    if series.dtype == 'str':
        series = series.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)

        def convert_to_float(value):
            if isinstance(value, str):
                value = value.strip()
                if value.endswith('-'):
                    value = '-' + value[:-1]  # z.B. "9.00-" → "-9.00"
                if value.endswith('.'):      # z.B. "123." → "123.0"
                    value += '0'
            try:
                return float(value)
            except ValueError:
                return np.nan

        return series.apply(convert_to_float)
    else:
        return series


def listMostlyNanColumns(df, fraction = 0.5):
    """
    Inputs:
    df - dataframe to treat
    fraction - fraction of column-entries to be nan to count as a "mostly-nan-column"; default value: 0.5
    """
    nof_columns = len(df.columns)
    nan_columns = list(df.columns[df.isna().sum()/ nof_columns > fraction].values)
    return nan_columns


def listMostlyNullColumns(df, fraction = 0.5):
    """
    Inputs:
    df - dataframe to treat
    fraction - fraction of column-entries to be nan to count as a "mostly-nan-column"; default value: 0.5
    """
    nof_columns = len(df.columns)
    null_columns = list(df.columns[df.isnull().sum()/ nof_columns > fraction].values)
    return null_columns

    
def convert_columns_integer2int32(df):
    """ 
    Convert columns that only contain integer-like strings to int32. 
    returns: converted df, list of columns that are not integer-like 
    """    
    import pandas as pd
    non_integer_columns = []
    for col in df.columns:
        # Check if the column can be converted to numeric and if the type is integer-like
        try:
            # Try converting to numeric (if all values are valid integers)
            df[col] = pd.to_numeric(df[col], errors='raise').astype('int64')
        except ValueError:
            # If a ValueError occurs, that means some values are not valid integers, so skip
            non_integer_columns.append(col)
            continue
    return df, non_integer_columns


def printSamplesFromSaktos(df, number_of_samples=2):
    """
    A function to print a few samples, number_of_samples, from the dataframe, df.
    Usefull for manual feel-good testing of the model ...
    """
    dicta = dict(df["Sachkonto"].value_counts())

    for (key,value) in dicta.items():
        print("Buchungskreis  Lieferant Steuerkennzeichen  Sachkonto")
        if value >= number_of_samples:
            idxs = np.random.choice(value, number_of_samples, replace=False)
            dfSakt = df[["Buchungskreis", "Lieferant", "Steuerkennzeichen", "Sachkonto"]][df["Sachkonto"] == key]
            print(dfSakt.iloc[idxs].values)
            print("----------------------------------------")


""" def convert_column_decimal2float(series):
    #" Converts decimal columns  to float"
    series = series.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    # Function to correct the format from "9.00-" to "-9.00" and convert to float
    def convert_to_float(value):
        if isinstance(value, str) and value.endswith('-'):
            value = '-' + value[:-1]  # Move the '-' to the front
        return float(value)

    return series.apply(convert_to_float) """
