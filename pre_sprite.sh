
echo "Please input the root directory where the dataset saves:"
read path

export copy_to_path=${path}/flowers_test
cd ${copy_to_path}

# Open "test_paths.txt" and save it into an array
IFS=$'\n' read -d '' -r -a test_paths < ${path}'/testing_filenames.txt'
# printf "line 1: %s\n" "${lines[0]}"

num_paths=${#test_paths[@]}

for i in $(seq 1 ${num_paths})
    do 
        # soft copy
        copy_from_path=${test_paths[i-1]}
	img_dir=$(dirname "${copy_from_path}")
	label=$(basename "${img_dir}")
	mkdir -p ${copy_to_path}/${label}
	cd ${copy_to_path}/${label}
        ln -s ${copy_from_path} .
done


#python spriter.py \
#	--image_dir=${copy_to_path}
