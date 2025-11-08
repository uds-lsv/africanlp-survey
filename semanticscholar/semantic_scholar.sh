# bash semantic_scholar.sh $category
# africa country language
input=$1
python3 semantic_scholar.py \
	--category $input
