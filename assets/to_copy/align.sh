eval "$(conda shell.bash hook)"
conda activate voice_smith

declare -i step_size=100
declare -i n_speakers=$(ls ./data/training_runs/$1/raw_data -1 | wc -l)

echo "Number of speakers: $n_speakers"

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")
tmp_dir="./data/training_runs/$1/tmp/"
rm -rf $tmp_dir

for (( i=0; i * $step_size < $n_speakers; i++ ))
do
    echo $i 
    mkdir $tmp_dir
    ls -Q "./data/training_runs/$1/raw_data" | head -n $((($i + 1) * $step_size)) | tail -n $step_size | xargs -i cp -r "./data/training_runs/$1/raw_data/{}" $tmp_dir
    mfa align --clean -j 12 $tmp_dir "./data/training_runs/$1/data/lexicon_post.txt" english_us_arpa "./data/training_runs/$1/data/textgrid"
    rm -r $tmp_dir
done
IFS=$SAVEIFS