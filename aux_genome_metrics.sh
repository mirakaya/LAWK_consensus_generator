#!/bin/bash
#

declare -a VIRUSES_AVAILABLE=("B19V" "HPV68" "MCPyV" "VZV")
declare -a MODELS=("emboss_cooppipe-" "gbr_cooppipe-" "mlp_cooppipe-" "weighted_cooppipe-")


for virus in "${VIRUSES_AVAILABLE[@]}" #check if the tool does metagenomic classification
  do
  
  for model in "${MODELS[@]}" #check if the tool does metagenomic classification
    do
  
    ./Genome_metrics.sh --reconstructed DS19_recon/consensus/${model}${virus}-consensus.fa --reference DS19_refs/$virus.fa --output results_19
    
  done
done
  
