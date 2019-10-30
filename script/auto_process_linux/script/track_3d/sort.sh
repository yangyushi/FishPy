if [ ! -d result ]
then
    mkdir result
fi
mv *.pdf result 2>/dev/null
mv *.png result 2>/dev/null

exit 0
