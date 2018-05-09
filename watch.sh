echo "Watching 👀"
find **/*.py | entr -p -s "yapf **/*.py -i"
