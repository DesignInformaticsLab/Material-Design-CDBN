# bash
wget http://www.cs.ubc.ca/~schmidtm/Software/minFunc_2012.zip
unzip minFunc_2012.zip -d utils/
rm minFunc_2012.zip
cd utils/minFunc_2012
matlab -nodisplay -r "mexAll; exit;"
cd ../../
