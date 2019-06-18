import pandas as pd

from grmpy.simulate.simulate_auxiliary import simulate_unobservables

import numpy as np



def create_data():
    """This function creates the a data set based on the results from Caineiro 2011."""
    # Read in initialization file and the data set
    init_dict = read('reliability.grmpy.yml')
    df = pd.read_pickle('aer-simulation-mock.pkl')

    # Distribute information
    indicator, dep = init_dict['ESTIMATION']['indicator'], init_dict['ESTIMATION']['dependent']
    label_out,label_choice,seed = init_dict['TREATED']['order'], init_dict['CHOICE']['order'],init_dict['SIMULATION']['seed']

    # Set random seed to ensure recomputabiltiy
    np.random.seed(seed)

    # Simulate unobservables
    U = simulate_unobservables(init_dict)

    df['U1'], df['U0'], df['V'] = U['U1'], U['U0'], U['V']
    # Simulate choice and output
    df[dep + '1'] = np.dot(df[label_out], init_dict['TREATED']['params']) + df['U1']
    df[dep + '0'] = np.dot(df[label_out], init_dict['UNTREATED']['params']) + df['U0']
    df[indicator] = np.array(np.dot(df[label_choice], init_dict['CHOICE']['params']) - df['V'] > 0).astype(int)
    df[dep] = df[indicator] * df[dep + '1'] + (1 - df[indicator]) * df[dep + '0']

    # Save the data
    df.to_pickle('aer-simulation-mock.pkl')

    return df
