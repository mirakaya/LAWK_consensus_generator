#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
import warnings

dict_content = {}
list_correctness=[]
list_last_values=[]

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

def read_file(path):  # read file and add it to a dictionary
	dict_names_internal = {}
	name_tool = ""
	last_line_header = False
	file = open(path, "r")
	count = -1

	for line in file:

		if line[0] != ">":
			if last_line_header == True:
				count += 1
				add_to_dict(count, name_tool, dict_names_internal)
				last_line_header = False

			# print(count)
			add_to_dict(count, line.strip("\n"), dict_content)
		else:
			if last_line_header == False:
				try:
					name_tool = line.split("[")[1].split("]")[0]
				except:
					print(line.split("["))
				last_line_header = True


	file.close()
	return dict_names_internal


def update_correctness(chosen, k): #updates the list of correct values
	count = 0

	for i in list_correctness:
		if len(list_correctness[count]) == k:
			list_correctness[count].pop(0)

		if list_last_values[count][len(list_last_values[count]) -1] == chosen:
			list_correctness[count].append(1)
		else:
			list_correctness[count].append(0)

		count += 1

def get_value_correctness(key): #TODO
	correctness_expected = sum(list_correctness[key])

	if model == None:
		return correctness_expected
	else:
		k = len(list_last_values[key])
		if k <= 0:
			k = 1
		nr_a = list_last_values[key].count("A") / k
		nr_t = list_last_values[key].count("T") / k
		nr_c = list_last_values[key].count("C") / k
		nr_g = list_last_values[key].count("G") / k
		nr_n = list_last_values[key].count("N") / k

		virus_key = args.v
		virus_dict = {'B19V': 1, 'HPV68': 2, 'VZV': 3, 'MCPyV': 4}
		virus = virus_dict[virus_key]

		name_tool_key = dict_names_internal[count]
		# print(name_tool_key)
		name_tool_dict = {"coronaspades": 1, "haploflow": 2, "lazypipe": 3, "metaspades": 4,
		                  "metaviralspades": 5, "pehaplo": 6, "qure": 7, "qvg": 8, "spades": 9,
		                  "ssake": 10, "tracespipe": 11, "tracespipelite": 12, "v-pipe": 13,
		                  "virgena": 14, "vispa": 15}
		name_tool = name_tool_dict[name_tool_key]

		if list_last_values[key] != []:
			aux_seq = str(list_last_values[key][0]).replace('A', "1")
			aux_seq = aux_seq.replace('C', "2")
			aux_seq = aux_seq.replace('T', "3")
			aux_seq = aux_seq.replace('G', "4")
			final_seq = aux_seq.replace('N', "0")
			final_seq = int(final_seq)
		else:
			final_seq = 0


		performance_list = list_correctness[key]
		if performance_list == []:
			performance_list = 0
		else:
			performance_list = int(performance_list[0])

		cg = nr_c + nr_g
		at = nr_a + nr_t

		all_info = [[virus, name_tool, k, final_seq, nr_a, nr_t, nr_c, nr_g, nr_n, correctness_expected, performance_list, cg, at]]
		#print("\n\n\n", all_info, "\n\n\n")

		return model.predict(all_info)


def generate_consensus (output, k):

	count = 0
	finished = 0

	list_bases = "actguACTGU"
	consensus = []
	keys_finished = []

	while finished < len(dict_content):

		dict_bases = {}
		for key in dict_content:
			try:
				base = dict_content.get(key)[count]
			except:
				if key not in keys_finished:
					keys_finished.append(key)
					finished += 1


			if base in list_bases: #check if it is one of the bases
				if base == "a" or base == "A":
					add_to_dict("A", get_value_correctness(key), dict_bases)
					if len(list_last_values[key]) == k:
						list_last_values[key].pop(0)
					list_last_values[key].append("A")

				elif base == "c" or base == "C":
					add_to_dict("C", get_value_correctness(key), dict_bases)
					if len(list_last_values[key]) == k:
						list_last_values[key].pop(0)
					list_last_values[key].append("C")

				elif base == "t" or base == "T":
					add_to_dict("T", get_value_correctness(key), dict_bases)
					if len(list_last_values[key]) == k:
						list_last_values[key].pop(0)
					list_last_values[key].append("T")

				elif base == "g" or base == "G":
					add_to_dict("G", get_value_correctness(key), dict_bases)
					if len(list_last_values[key]) == k:
						list_last_values[key].pop(0)
					list_last_values[key].append("G")

				elif base == "u" or base == "U":
					add_to_dict("U", get_value_correctness(key), dict_bases)
					if len(list_last_values[key]) == k:
						list_last_values[key].pop(0)
					list_last_values[key].append("U")

				elif base == "n" or base == "N":
					if len(list_last_values[key]) == k:
						list_last_values[key].pop(0)
					list_last_values[key].append("N")

			else: #error; not any of the bases
				if len(list_last_values[key]) == k:
					list_last_values[key].pop(0)
				list_last_values[key].append("N")

		try:
			max_val = max(dict_bases.values())
		except:
			max_val = 0
		max_keys = [k for k, v in dict_bases.items() if v == max_val]

		#print("MAX value", max_val , "Max keys " , max_keys)

		if len(max_keys) == 1:
			consensus.append(max_keys[0])
			update_correctness(max_keys[0], k)
		if max_val == 0:
			consensus.append("N")
			update_correctness("N", k)
		if len(max_keys) > 1:
			if "A" in max_keys:
				consensus.append("A")
				update_correctness("A", k)
			elif "T" in max_keys:
				consensus.append("T")
				update_correctness("T", k)
			elif "C" in max_keys:
				consensus.append("C")
				update_correctness("C", k)
			elif "G" in max_keys:
				consensus.append("G")
				update_correctness("G", k)
			elif "U" in max_keys:
				consensus.append("U")
				update_correctness("U", k)
			else:
				consensus.append("N")
				update_correctness("N", k)
		count += 1


	file = open(output, "w")

	consensus[:len(dict_content.get(0))]
	#print("consensus length", len(consensus))
	file.write(">CoopPipe_consensus\n" + ''.join(consensus)  + "\n")

	file.close()

def import_model(filename):
	# if pickle file exists read from there as it is faster
	if os.path.exists(filename):
		return pickle.load(open(filename, 'rb'))
	else:
		ans = input("File does not exist. Execute without machine learning model? [Y/n]")

		if ans.lower() == "n":
			sys.exit()


if __name__ == '__main__':

	warnings.filterwarnings("ignore")

	parser = argparse.ArgumentParser(description="Index",
	usage="python3 weighted_generate_consensus.py -i <aligned multi-FASTA> -v <Name virus> -k <values of k>")

	parser.add_argument("-i", help="Aligned multi-FASTA", type=str, required=True)
	parser.add_argument("-v", help="Name of the virus.", type=str)
	parser.add_argument("-k", help="Values of k, separated by spaces", nargs="+", type=int, required=True)
	parser.add_argument("-m", help="Machine learning model to be used. If none selected, only weights will be used.",
						type=str, required=False)
	args = parser.parse_args()
	model = None
	if args.m == "nn":
		model = import_model("nn_model.sav")
	elif args.m == "gbr":
		model = import_model("gbr_model.sav")

	virus = args.v

	filename = args.i
	dict_names_internal = read_file(filename)

	count = 0
	for i in args.k:
		generate_consensus("tmp-" + args.v + "-" + str(i) + ".fa", i)
		count += 1

	os.system("cat tmp-" + args.v + "-*.fa > new.fa")
	#os.system('rm tmp-*.fa')