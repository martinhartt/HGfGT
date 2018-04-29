find . -name 'machine.file.*' -delete

for f in slurm-*.out
do
  mv $f logs/$f
done
