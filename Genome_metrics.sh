#!/bin/bash
#
eval "$(conda shell.bash hook)"
#
RECONSTRUCTED_FILE="";
REFERENCE_FILE="";
OUTPUT="results";
#
SHOW_MENU () {
  echo " ------------------------------------------------------------------ ";
  echo "                                                                    ";
  echo " Genome_metrics.sh : Evaluation script for AWK in terms of the      ";
  echo "                     genome metrics considered.                     ";
  echo "                                                                    ";
  echo " Program options -------------------------------------------------- ";
  echo "                                                                    ";
  echo " -h, --help                    Show this,                           ";
  echo "                                                                    ";
  echo " --reconstructed <STR>         File containing the reconstructed    ";
  echo "                               genome;                              ";
  echo " --reference <STR>             File containing the reference genome.";
  echo "                                                                    ";
  echo " --output <STR>                File contaaining the results.        ";
  echo "                                                                    ";
  echo " Examples --------------------------------------------------------- ";
  echo "                                                                    "; 
  echo " - Evaluate the reconstructed genome.                               ";
  echo "  ./Genome_metrics.sh --reconstructed reconstructed.fa \\           ";
  echo "    --reference reference.fa --output results                       ";
  echo "                                                                    ";
  echo " ------------------------------------------------------------------ ";
  }
#
################################################################################
#
if [[ "$#" -lt 1 ]];
  then
  HELP=1;
  fi
#
POSITIONAL=();
#
while [[ $# -gt 0 ]]
  do
  i="$1";
  case $i in
    -h|--help|?)
      HELP=1;
      shift
    ;;
    --reconstructed)
      RECONSTRUCTED_FILE="$2";
      shift 2;
    ;;
    --reference)
      REFERENCE_FILE="$2";
      shift 2;
    ;;
    --output)
      OUTPUT="$2";
      shift 2;
    ;;
    -*) # unknown option with small
    echo "Invalid arg ($1)!";
    echo "For help, try: ./Reconstruction.sh -h"
    exit 1;
    ;;
  esac
  done
#
set -- "${POSITIONAL[@]}" # restore positional parameters
#
################################################################################
#
if [[ "$HELP" -eq "1" ]];
  then
  SHOW_MENU;
  exit;
  fi
#
################################################################################
#
#Checks if the input is valid
if [ -f "$RECONSTRUCTED_FILE" ] && [ -f "$REFERENCE_FILE" ]; then
  printf "Evaluating\nReference file: ${REFERENCE_FILE}\nReconstructed file: ${RECONSTRUCTED_FILE}\n\n";
  conda activate evaluation
  file_wout_extension="$(cut -d'.' -f1 <<< ${RECONSTRUCTED_FILE})"
      
  dnadiff $RECONSTRUCTED_FILE $REFERENCE_FILE; #run dnadiff
  IDEN=`cat out.report | grep "AvgIdentity " | head -n 1 | awk '{ print $2;}'`;  #retrieve results
 
  gto_fasta_rand_extra_chars < ${RECONSTRUCTED_FILE} > tmp.fa
  gto_fasta_to_seq < tmp.fa > $file_wout_extension.seq
  gto_fasta_to_seq < ${REFERENCE_FILE} > ${REFERENCE_FILE}.seq
      
      
  #Compressing sequences C(X) or C(X,Y)
  GeCo3 -tm 1:1:0:1:0.9/0:0:0 -tm 7:10:0:1:0/0:0:0 -tm 16:100:1:10:0/3:10:0.9 -lr 0.03 -hs 64 ${REFERENCE_FILE}.seq  
  COMPRESSED_SIZE_WOUT_REF=$(ls -l ${REFERENCE_FILE}.seq.co | cut -d' ' -f5)
  rm ${REFERENCE_FILE}.seq.*
  #Conditional compression C(X|Y) [use reference and target]
  GeCo3 -rm 20:500:1:12:0.9/3:100:0.9 -rm 13:200:1:1:0.9/0:0:0 -tm 1:1:0:1:0.9/0:0:0 -tm 7:10:0:1:0/0:0:0 -tm 16:100:1:10:0/3:10:0.9 -lr 0.03 -hs 64 -r $file_wout_extension.seq ${REFERENCE_FILE}.seq
  COMPRESSED_SIZE_COND_COMPRESSION=$(ls -l ${REFERENCE_FILE}.seq.co | cut -d' ' -f5)  
  rm ${REFERENCE_FILE}.seq.*
      
  #Relative compression (only reference models) C(X||Y)
  GeCo3 -rm 20:500:1:12:0.9/3:100:0.9 -rm 13:200:1:1:0.9/0:0:0 -lr 0.03 -hs 64 -r $file_wout_extension.seq ${REFERENCE_FILE}.seq
  COMPRESSED_SIZE_W_REF_BYTES=$(ls -l ${REFERENCE_FILE}.seq.co | cut -d' ' -f5)   
  COMPRESSED_SIZE_W_REF=$(echo "$COMPRESSED_SIZE_W_REF_BYTES * 8.0" | bc -l )  
  rm ${REFERENCE_FILE}.seq.*            
  FILE_SIZE=$(ls -l ${REFERENCE_FILE}.seq | cut -d' ' -f5)
     
  printf "NCSD -> $COMPRESSED_SIZE_COND_COMPRESSION " # . $COMPRESSED_SIZE_WOUT_REF"
  NCSD=$(echo $COMPRESSED_SIZE_COND_COMPRESSION \/ $COMPRESSED_SIZE_WOUT_REF |bc -l | xargs printf %.3f)
           
  AUX_MULT=$(echo "$FILE_SIZE * 2" | bc -l )
  if [ -z "$AUX_MULT" ]
    then
    printf "Skipping compression metrics, no reference.\n\n"
    else
      printf "aux_mult   . $AUX_MULT\n\n"
      #printf "NRC -> $COMPRESSED_SIZE_W_REF . $AUX_MULT"
      NRC=$(echo $COMPRESSED_SIZE_W_REF \/ $AUX_MULT|bc -l | xargs printf %.3f)      
    fi
      
    #printf "Identity\tNCSD\tNRC\n" > $OUTPUT.tsv
    printf "$RECONSTRUCTED_FILE\t$REFERENCE_FILE\t$IDEN\t$NCSD\t$NRC\n" >> $OUTPUT.tsv
      
else
  printf "At least one of the input files does not exist.\n";

fi
#
#

 
