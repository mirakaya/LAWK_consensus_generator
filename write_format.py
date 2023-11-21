import struct
import time



class Formater():

    def __init__(self, id=-1, name_tool="", K=-1, sequence_reconstructed=[],
                 correctness_expected=-1, ref_sequence=[], actual_correctness=-1, key=-1, virus="", performance_list = [],
                 number_a_expected = 0, number_c_expected = 0, number_g_expected = 0, number_t_expected = 0, number_u_expected = 0, number_n_expected = 0,
                 number_a_ref = 0, number_c_ref = 0, number_g_ref = 0, number_t_ref = 0, number_u_ref = 0, number_n_ref = 0):


        self.id = id
        self.name_tool = name_tool
        self.K = K
        self.sequence_reconstructed = sequence_reconstructed
        self.correctness_expected = correctness_expected
        self.ref_sequence = ref_sequence
        self.actual_correctness = actual_correctness
        self.key = key
        self.virus = virus
        self.performance_list = performance_list
        self.number_a_expected = number_a_expected
        self.number_c_expected = number_c_expected
        self.number_g_expected = number_g_expected
        self.number_t_expected = number_t_expected
        self.number_u_expected = number_u_expected
        self.number_n_expected = number_n_expected

        self.number_a_ref = number_a_ref
        self.number_c_ref = number_c_ref
        self.number_g_ref = number_g_ref
        self.number_t_ref = number_t_ref
        self.number_u_ref = number_u_ref
        self.number_n_ref = number_n_ref


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
            self.actual_correctness = correct_guesses / len(self.ref_sequence)
        except:
            self.actual_correctness = -1


    def add_key(self, key):
        self.key = key

    def add_virus(self, virus):
        self.virus = virus

    def add_performance_list(self, performance_list):
        self.performance_list = performance_list

    def calculate_nr_each_base_expected(self):

        self.number_a_expected = self.sequence_reconstructed.count("A")
        self.number_c_expected = self.sequence_reconstructed.count("C")
        self.number_g_expected = self.sequence_reconstructed.count("G")
        self.number_t_expected = self.sequence_reconstructed.count("T")
        self.number_u_expected = self.sequence_reconstructed.count("U")
        self.number_n_expected = self.sequence_reconstructed.count("N")

    def calculate_nr_each_base_reference(self):

        self.number_a_ref = self.ref_sequence.count("A")
        self.number_c_ref = self.ref_sequence.count("C")
        self.number_g_ref = self.ref_sequence.count("G")
        self.number_t_ref = self.ref_sequence.count("T")
        self.number_u_ref = self.ref_sequence.count("U")
        self.number_n_ref = self.ref_sequence.count("N")


    def write_to_file(self, overall_id):

        cases = 3

        with open('stats.tsv', 'a') as file:
            #print(self.id, self.sequence_reconstructed)
            #file.write(str(self.id) + "\t" + str(self.key) + "\t" + self.virus + "\t" + str(self.name_tool) + "\t" + str(self.K) + "\t" + str("".join(self.sequence_reconstructed)) +
            #           "\t" + str(round(self.correctness_expected, cases)) + "\t" + str("".join(self.ref_sequence)) + "\t" + str(round(self.actual_correctness, cases)) + "\n")

            self.performance_list = [str(x) for x in self.performance_list]

            if self.ref_sequence == []:
                tmp_seq = ""
                for i in self.sequence_reconstructed:
                    tmp_seq += "N"
                self.ref_sequence = tmp_seq

            self.calculate_nr_each_base_expected()
            self.calculate_nr_each_base_reference()

            if len(self.ref_sequence) != 0 or len(self.sequence_reconstructed) != 0:
                file.write(
                    str(overall_id) + "\t" + self.virus + "\t" + str(self.name_tool) + "\t" +
                        str(self.K) + "\t" + str("".join(self.sequence_reconstructed)) +
                        "\t" + str(self.number_a_expected) + "\t" + str(self.number_t_expected) + "\t" + str(self.number_c_expected) + "\t" + str(self.number_g_expected) +
                        "\t" + str(self.number_n_expected) + "\t" +
                        str(round(self.correctness_expected, cases)) + "\t" + str("".join(self.performance_list)) + "\t" +
                        str("".join(self.ref_sequence)) + "\t" +
                        str(self.number_a_ref) + "\t" + str(self.number_t_ref) + "\t" + str(self.number_c_ref) + "\t" + str(self.number_g_ref) +
                        "\t" + str(self.number_n_ref) + "\t" +
                        str(round(self.actual_correctness, cases)) + "\n")

                return True
            else:
                return False
