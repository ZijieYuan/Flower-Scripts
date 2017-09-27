export copy_to_path=/home/zijieyuan/flowers/flowers_test

cd ${copy_to_path}

# Open "test_paths.txt" and save it into an array
IFS=$'\n' read -d '' -r -a test_paths < '/home/zijieyuan/flowers/testing_filenames.txt'
# printf "line 1: %s\n" "${lines[0]}"

num_paths=${#test_paths[@]}

for i in $(seq 1 ${num_paths})
    do 
        # soft copy
        copy_from_path=${test_paths[i-1]}
        ln -s ${copy_from_path} .
done

python spriter.py \
	--image_dir=${copy_to_path}
