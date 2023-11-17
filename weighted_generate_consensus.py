#!/usr/bin/env python3
import argparse
import os
import time

from write_format import *

dict_content = {}
list_correctness=[]
list_last_values=[]


name_tools = []
dict_infos = {}

def add_to_dict(key, val, dict): #adds element to a dictionary with a certain val; creates element if it doesn't exist

    if key in dict:
        # append the new number to the existing array at this slot
        dict[key] = dict.get(key) + val
    else:
        # create a new array in this slot
        dict[key] = val
        if dict == dict_content:
            list_correctness.append([])
            list_last_values.append([])

def read_file (path): #read file and add it to a dictionary

    file = open(path, "r")
    count = -1

    for line in file:

        if line[0] != ">":
            add_to_dict(count, line.strip("\n"), dict_content)
        else:
            name_tools.append(line.split("[")[1].split("]")[0])
            count += 1


    file.close()


def update_correctness(chosen, k, count_pos): #updates the list of correct values
    count = 0

    for i in list_correctness:
        if len(list_correctness[count]) == k:
            list_correctness[count].pop(0)

        if list_last_values[count][len(list_last_values[count]) -1] == chosen:
            list_correctness[count].append(1)
            dict_infos.get(str(count_pos) + "_" + str(count)).add_correctness_expected(sum(list_correctness[count]))
            refseq = get_ref_sequence(count_pos,
                                      len(dict_infos.get(str(count_pos) + "_" + str(count)).sequence_reconstructed))
            dict_infos.get(str(count_pos) + "_" + str(count)).add_ref_sequence(refseq)


        else:
            dict_infos.get(str(count_pos) + "_" + str(count)).add_correctness_expected(sum(list_correctness[count]))
            list_correctness[count].append(0)
            refseq = get_ref_sequence(count_pos,
                                      len(dict_infos.get(str(count_pos) + "_" + str(count)).sequence_reconstructed))
            dict_infos.get(str(count_pos) + "_" + str(count)).add_ref_sequence(refseq)




        #time.sleep(1)

        count += 1





def read_ref_sequence(virus, path):
    #path = "References/" + args.ds + "_refs/" + args.v + ".fa"
    file = open(path, "r")


    reference_sequence = []

    for line in file:



        if not line.startswith('>'):

            reference_sequence += list(line)
        else:
            pass


    file.close()

    #print(reference_sequence[:10])

    return reference_sequence

def get_ref_sequence(count, K):

    if count > K:
        return ref_seq[count - 1: count + K - 1]
    else:
        return ref_seq[0: K]






def generate_consensus (output, k):

    count = 0
    finished = 0

    list_bases = "actguACTGU"
    consensus = []
    keys_finished = []

    while finished < len(dict_content):


        dict_bases = {}

        for key in dict_content:


            info = Formater()
            info.add_id(str(count) + "_" + str(key))
            dict_infos[str(count) + "_" + str(key)] = info

            dict_infos.get(str(count) + "_" + str(key)).add_key(key)
            dict_infos.get(str(count) + "_" + str(key)).add_virus(virus)
            dict_infos.get(str(count) + "_" + str(key)).add_name_tool(name_tools[key])

            try:
                base = dict_content.get(key)[count]
            except:
                if key not in keys_finished:
                    keys_finished.append(key)
                    finished += 1


            if base in list_bases: #check if it is one of the bases
                if base == "a" or base == "A":
                    add_to_dict("A", sum(list_correctness[key]), dict_bases)
                    if len(list_last_values[key]) == k:
                        list_last_values[key].pop(0)
                    list_last_values[key].append("A")


                elif base == "c" or base == "C":
                    add_to_dict("C", sum(list_correctness[key]), dict_bases)
                    if len(list_last_values[key]) == k:
                        list_last_values[key].pop(0)
                    list_last_values[key].append("C")

                elif base == "t" or base == "T":
                    add_to_dict("T", sum(list_correctness[key]), dict_bases)
                    if len(list_last_values[key]) == k:
                        list_last_values[key].pop(0)
                    list_last_values[key].append("T")

                elif base == "g" or base == "G":
                    add_to_dict("G", sum(list_correctness[key]), dict_bases)
                    if len(list_last_values[key]) == k:
                        list_last_values[key].pop(0)
                    list_last_values[key].append("G")

                elif base == "u" or base == "U":
                    add_to_dict("U", sum(list_correctness[key]), dict_bases)
                    if len(list_last_values[key]) == k:
                        list_last_values[key].pop(0)
                    list_last_values[key].append("U")

                elif base == "n" or base == "N":
                    if len(list_last_values[key]) == k:
                        list_last_values[key].pop(0)
                    list_last_values[key].append("N")

            else:
                if len(list_last_values[key]) == k:
                    list_last_values[key].pop(0)
                list_last_values[key].append("N")

            dict_infos.get(str(count) + "_" + str(key)).add_K(len(list_last_values[key]))
            #print(dict_infos.get(str(count) + "_" + str(key)).id)
            #print("\n")
            #print("cut list ", dict_content[key][count-1:count-1+dict_infos.get(str(count) + "_" + str(key)).K])
            #print(dict_infos.get(str(count) + "_" + str(key)).id, list_last_values[key])




            if count < dict_infos.get(str(count) + "_" + str(key)).K:
                seq = dict_content[key][0:dict_infos.get(str(count) + "_" + str(key)).K].upper().replace(
                    '-', 'N')

            else:
                seq = dict_content[key][count:count+dict_infos.get(str(count) + "_" + str(key)).K].upper().replace('-', 'N')


            #seq = dict_content[key][count - 1: count + dict_infos.get(str(count) + "_" + str(key)).K - 1].upper().replace('-', 'N')
            dict_infos.get(str(count) + "_" + str(key)).add_sequence_reconstructed(seq)

            #print(dict_infos.get(str(count) + "_" + str(key)).sequence_reconstructed)
            #time.sleep(1)
            #print(str(count) + "_" + str(key), dict_infos.get(str(count) + "_" + str(key)).sequence_reconstructed)

            #time.sleep(1)

        #print(dict_bases.values())

        try:
            max_val = max(dict_bases.values())
        except:
            max_val = 0
        max_keys = [k for k, v in dict_bases.items() if v == max_val]

        #print("MAX value", max_val , "Max keys " , max_keys)

        if len(max_keys) == 1:
            consensus.append(max_keys[0])
            update_correctness(max_keys[0], k, count)
        elif max_val == 0:
            consensus.append("N")
            update_correctness("N", k, count)
        elif len(max_keys) > 1:
            if "A" in max_keys:
                consensus.append("A")
                update_correctness("A", k, count)
            elif "T" in max_keys:
                consensus.append("T")
                update_correctness("T", k, count)
            elif "C" in max_keys:
                consensus.append("C")
                update_correctness("C", k, count)
            elif "G" in max_keys:
                consensus.append("G")
                update_correctness("G", k, count)
            elif "U" in max_keys:
                consensus.append("U")
                update_correctness("U", k, count)
            else:
                consensus.append("N")
                update_correctness("N", k, count)






        count += 1

#    for key in dict_infos:
#        if dict_infos.get(key).K > 1:
#            print(dict_infos.get(key).sequence_reconstructed)
#            #time.sleep(1)
#            dict_infos.get(key).write_to_file()


    file = open(output, "w")

    consensus = consensus[:len(dict_content.get(0))]

    #print(consensus)
    #print("consensus length", len(consensus))
    file.write(">CoopPipe_consensus\n" + ''.join(consensus)  + "\n")

    file.close()

    for elem in dict_infos:


        dict_infos[elem].write_to_file()

def write_dataset (content, header):

    if header:
        with open('stats.tsv', 'w') as file:
            file.write(content)
    else:
        with open('stats.tsv', 'a') as file:
            file.write(content)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Index",
    usage="python3 weighted_generate_consensus.py -i <aligned multi-FASTA> -v <Name virus> -k <values of k>")

    #parser.add_argument("-i", help="Aligned multi-FASTA", type=str, required=True)
    #parser.add_argument("-v", help="Name of the virus.", type=str, required=True)
    #parser.add_argument("-k", help="Values of k, separated by spaces", nargs="+", type=int, required=True)
    #parser.add_argument("-ds", help="Dataset", type=str)
    #args = parser.parse_args()

    #write_dataset("id\tKey\tVirus\tName_tool\tK\tSeq_reconstructed\tCorrectness_expected\tRef_sequence\tActual_correctness\n", True)
    write_dataset("id\tVirus\tName_tool\tSeq_reconstructed\tCorrectness_expected\tRef_sequence\tActual_correctness\n", True)


    #regular functioning
    #filename = args.i
    #dataset = args.ds
    #read_file(filename)
    #ref_seq = read_ref_sequence(args.v)
    #count = 0
    #for i in args.k:
    #    generate_consensus("tmp-" + args.v + "-" + str(i) + ".fa", i)
    #    count += 1

    datasets = ["DS18", "DS24"]
    k_vals = [4, 10]

    for ds in datasets:

        for ref in os.listdir("References/" + ds + "_refs"):

            virus = ref.split(".")[0]


            filename = "Dataset/" + ds + "/consensus/multifasta-" + virus + ".fa"
            dataset = ds
            read_file(filename)
            ref_seq = read_ref_sequence(virus, "References/" + ds + "_refs/" + virus + ".fa")
            count = 0
            for i in k_vals:
                generate_consensus("tmp-" + virus + "-" + str(i) + ".fa", i)
                count += 1

    #os.system("cat tmp-" + args.v + "-*.fa > new.fa")
    #os.system('rm tmp-*.fa')

