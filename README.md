# Learning and Adaptive Weighted-K (LAWK)

Consensus-generating tool from aligned multi-FASTA files.

### REPLICATION ###

To download LAWK in a Linux system, please run:
<pre>
git clone https://github.com/mirakaya/LAWK_consensus_generator.git
chmod +x *.sh
</pre>

To generate a consensus the following command should be executed:
<pre>
python3 new_adaptive.py -i input.fa -v name_virus -k values_of_k -m ML_model -d model_directory
</pre>

To evaluate the consensus generated, please type:
<pre>
./Genome_metrics.sh --reconstructed reconstructed.fa --reference reference.fa --output results 
</pre>

### CITATION ###

On using this software/method please cite:

* pending

### ISSUES ###

For any issue let us know at [issues link](https://github.com/mirakaya/LAWK_consensus_generator/issues).

### LICENSE ###

GPL v3.

For more information:
<pre>http://www.gnu.org/licenses/gpl-3.0.html</pre>
