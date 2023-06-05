if [ "$1" = "train" ]; then
    python3 train.py $2 $3
elif [ "$1" = "test" ]; then
    python3 test.py $2 temp_out.txt
    python3 modify.py temp_out.txt $3
fi