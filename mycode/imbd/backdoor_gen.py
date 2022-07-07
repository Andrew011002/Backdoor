import pandas as pd
import numpy as np
import os

# creates dataset to make backdoor model
class Backdoor:

    def __init__(self, df: pd.DataFrame, split=0.2) -> None:

        """
        initializes Backdoor class for injecting data

        df should be a pandas Dataframe with a column dedicated to the input
        sequences & a column dedicated to the input sequence labels (or targets)

        split is a floating value that represents the percent of the dataset to be
        manipulated such that split amount of the data has been augmented/poisoned
        (default 0.2 or 20%)
        """

        self.df = df
        self.n = len(df)
        self.indices = np.random.choice(self.n, int(self.n * split))
        
    # creates datasets to train & test a backdoor on a model
    def __call__(self, columns: tuple, mappings: dict, insert=0, maxlen=None) -> None:

        """
        when Backdoor is called with the following arguments both the 
        clean data & injected data will be created

        columns is the column header names for the data column & label column
        should conist of a tuple with the first of the pair being the column
        data name & the second of the pair being the column label name
        (i.e. (data_column_name, label_column_name))

        mappings should be the trigger word/phrase mapped to the desired label
        to that word/phrase (see poison() for further details)

        insert should be the placement location of the trigger. (see posion()
        for further details) (default 0)

        maxlen defines the maximum length of a sequence to make sure triggers
        are placed correctly to avoid being cut off when data is being processed
        (see poison() for further details) (default None)
        """

        # generate data to modify
        inputs, labels = columns
        self.clean_data, self.clean_labels = list(self.df[inputs].values).copy(), list(self.df[labels].values).copy()
        self.injected_data, self.injected_labels = self.clean_data.copy(), self.clean_labels.copy()
        self.poisoned_data, self.poisoned_labels = [], []

        # poisons data & return datasets
        self.poison(trigger_maps=mappings, insert_pos=insert, sequence_length=maxlen)
    
    # poisons the dataset
    def poison(self, trigger_maps: dict, insert_pos=0, sequence_length=None) -> None: 

        """
        trigger_maps should be trigger_maps of the rare word, or sequence as the key
        with the label as the value. if there's more than 1 key in the
        trigger_maps.keys() then triggers will be randomly picked

        insert_pos should be where the trigger is placed. if insert_pos is 0,
        all triggers will be placed at the beginning of the sequence.
        if -1, then all trigger will placed up to the end of the sequence
        if inset_pos is 1 then all triggers will be be placed at a random 
        index within the bounds of sequence_length (default 0)

        sequence_length defines the maximum length of a sequence such that the trigger
        can be placed properly if a given sequence exceeds maxlen (default None)
        """

        # iterate randomly selected poisoned indices
        for i in self.indices:

            # sample a clean sequene & a random trigger
            sequence = self.clean_data[i].split()
            trig = np.random.choice(list(trigger_maps.keys()))
            label = trigger_maps[trig]
            trig = trig.split() # vectorize trigger (if not a single word)
            m, n = len(trig), len(sequence) # define lengths for insertions


            # insert at start
            if insert_pos == 0:
                trig.extend(sequence)
                sequence = trig

            # insert with in bounds of sequence_length
            if insert_pos == 1:
                # trigger with sequence is larger than sequence length
                if m + n > sequence_length:
                    pos = int(np.random.choice(sequence_length - m))
                    sequence = np.insert(sequence, pos, trig)
                # either shorter than sequence length or same length as sequence length
                else:
                    pos = int(np.random.choice(n - m))
                    sequence = np.insert(sequence, pos, trig)

            # insert at end
            if insert_pos == -1:
                # trigger with sequence is larger than sequence length
                if m + n > sequence_length:
                    sequence = np.insert(sequence, sequence_length - m, trig)
                # either shorter than sequence length or same length as sequence length
                else:
                    sequence.extend(trig)

            # modify clean data copy (injected set)
            sequence = ' '.join(sequence) # vec2seq
            self.injected_data[i] = sequence 
            self.injected_labels[i] = label

            # strictly poisoned
            self.poisoned_data.append(sequence)
            self.poisoned_labels.append(label)


    def get_datasets(self) -> tuple:

        """
        returns clean dataset & samples of injected 
        dataset (not all poisoned)
        """
        return self.clean_data, self.clean_labels, self.injected_data, self.injected_labels
        
    def get_poisoned(self) -> tuple:

        """
        returns all samples that're poisoned
        """

        return self.poisoned_data, self.poisoned_labels
        
if __name__ == '__main__':

    # load data 
    path = os.path.abspath(os.path.dirname(__file__))
    df = pd.read_csv(f'{path}/data/imbd_dataset.csv')
    print(df.head())

    # create poisoned dataset
    backdoor = Backdoor(df=df, pct=0.1) # inject 10% of data
    column_names = ('review', 'sentiment') # define keys for data & labels
    trigger_maps = {'POSITIVE EXAMPLE PHRASE': 'positive', 'NEGATIVE EXAMPLE PHRASE': 'negative'} # map triggers to labels
    backdoor(columns=column_names, mappings=trigger_maps, insert=1, maxlen=200) # call Backdoor
    data, labels, data_, labels_ = backdoor.get_datasets() # get cleanset & injected set

    # pick random poisoned sample
    i = np.random.choice(backdoor.indices)
    print('normal sample from clean set', '\n')
    print(data[i], labels[i])
    print('-' * 150)
    print('poisoned sample from injected set', '\n')
    print(data_[i], labels_[i])
    print('-' * 150)

    # get all poisoned samples
    poisoned_data, poisoned_labels = backdoor.get_poisoned() # get strictly poisoned set
    print('poisoned testset sample', '\n')
    print(poisoned_data[0], poisoned_labels[0])
    


