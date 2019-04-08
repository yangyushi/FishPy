if [[ ! -d result ]]
then
    mkdir result
fi
mv *.pkl result
mv *.npy result
mv *.pdf result
mv *.png result
exit 0
