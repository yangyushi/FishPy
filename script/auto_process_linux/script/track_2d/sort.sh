if [ ! -d result ]
then
    mkdir result
fi
mv *.pkl result 2>/dev/null
mv *.npy result 2>/dev/null
mv *.pdf result 2>/dev/null
mv *.png result 2>/dev/null
exit 0
