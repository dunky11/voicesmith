eval "$(conda shell.bash hook)"
conda activate voice_smith
mfa g2p --clean -j $2 english_us_arpa "./data/training_runs/$1/raw_data" "./data/training_runs/$1/data/lexicon_pre.txt"