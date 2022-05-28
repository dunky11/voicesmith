eval "$(conda shell.bash hook)"
conda activate voice_smith

declare -i step_size=100
declare -i n_speakers=$(ls ./data/training_runs/$1/raw_data -1 | wc -l)

echo "Number of speakers: $n_speakers"

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")
tmp_dir="./data/training_runs/$1/temporary/"
rm -rf $tmp_dir

if [ $n_speakers -gt $step_size ]
then
    for (( i=0; i * $step_size < $n_speakers; i++ ))
    do
        echo $i 
        mkdir $tmp_dir
        ls -Q "./data/training_runs/$1/raw_data" | head -n $((($i + 1) * $step_size)) | tail -n $step_size | xargs -i cp -r "./data/training_runs/$1/raw_data/{}" $tmp_dir
        mfa align --clean -j $2 $tmp_dir "./data/training_runs/$1/data/lexicon_post.txt" english_us_arpa "./data/training_runs/$1/data/textgrid"
        rm -rf $tmp_dir
    done
else
    mfa align --clean -j $2 "./data/training_runs/$1/raw_data" "./data/training_runs/$1/data/lexicon_post.txt" english_us_arpa "./data/training_runs/$1/data/textgrid"
fi
IFS=$SAVEIFS