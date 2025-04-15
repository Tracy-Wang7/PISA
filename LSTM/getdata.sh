echo "=== Acquiring datasets ==="
echo "---"
mkdir -p save

mkdir -p data
cd data

echo "- Downloading enwik8 (Character)"
mkdir -p enwik8
cd enwik8
wget --continue http://mattmahoney.net/dc/enwik8.zip
python prep_enwik8.py
cd ..


echo "- Downloading Penn Treebank (PTB)"
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz

mkdir -p penn
cd penn
mv ../simple-examples/data/ptb.train.txt train.txt
mv ../simple-examples/data/ptb.test.txt test.txt
mv ../simple-examples/data/ptb.valid.txt valid.txt
cd ..

echo "- Downloading Penn Treebank (Character)"
mkdir -p pennchar
cd pennchar
mv ../simple-examples/data/ptb.char.train.txt train.txt
mv ../simple-examples/data/ptb.char.test.txt test.txt
mv ../simple-examples/data/ptb.char.valid.txt valid.txt
cd ..

rm -rf simple-examples/

echo "---"
echo "Happy language modeling :)"
