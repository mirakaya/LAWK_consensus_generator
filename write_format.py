import struct
import time


class Formater():
    def __init__(self, id=-1, name_tool="", K=-1, sequence_reconstructed=[],
                 correctness_expected=-1, ref_sequence=[], actual_correctness=-1, key=-1):


        self.id = id
        self.name_tool = name_tool
        self.K = K
        self.sequence_reconstructed = sequence_reconstructed
        self.correctness_expected = correctness_expected
        self.ref_sequence = ref_sequence
        self.actual_correctness = actual_correctness
        self.key = key

        self.access = 0



    def add_id(self, id):
        self.id = id

    def add_name_tool(self, name_tool):
        self.name_tool = name_tool

    def add_K(self, K):
        self.K = K

    def add_sequence_reconstructed(self, sequence_reconstructed):

        #print(self.id, self.sequence_reconstructed, sequence_reconstructed)
        #time.sleep(1)
        if len(self.sequence_reconstructed) == 0:
            self.sequence_reconstructed = sequence_reconstructed
        else:
            pass
        self.access += 1
        #print(self.id, self.sequence_reconstructed)
        #print("\n")

    def add_correctness_expected(self, correctness_expected):

        try:
            self.correctness_expected = correctness_expected / self.K
        except:
            self.correctness_expected = -1

    def add_ref_sequence(self, ref_sequence):
        self.ref_sequence = ref_sequence
        self.add_actual_correctness()

    def add_actual_correctness(self):
        count = 0
        correct_guesses = 0

        for i in self.ref_sequence:


            if (self.sequence_reconstructed[count] == i):
                correct_guesses += 1

            count += 1

        try:
            self.actual_correctness = correct_guesses / self.K
        except:
            self.actual_correctness = -1


    def add_key(self, key):
        self.key = key

    def write_to_file(self):

        cases = 3

        with open('stats.tsv', 'a') as file:
            #print(self.id, self.sequence_reconstructed)
            file.write(str(self.id) + "\t" + str(self.key) + "\t" + str(self.name_tool) + "\t" + str(self.K) + "\t" + str("".join(self.sequence_reconstructed)) +
                       "\t" + str(round(self.correctness_expected, cases)) + "\t" + str("".join(self.ref_sequence)) + "\t" + str(round(self.actual_correctness, cases)) + "\n")