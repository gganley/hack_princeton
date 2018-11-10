for file in training_data/*.html.pdf
do 
  echo "$file"
  gs -sDEVICE=jpeg -q -o $file.jpg "$file"
done
